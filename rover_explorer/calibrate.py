from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from .localize import Localizer, RoverPose
from .motion import Action, pulse


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


class CalibrationError(RuntimeError):
    """Calibration failure carrying the last frame for offline diagnosis."""

    def __init__(self, message: str, frame=None):
        super().__init__(message)
        self.frame = frame


async def _locate_until(camera, localizer: Localizer, timeout: float, min_confidence: float):
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout)
    frame = camera.read()
    pose = await asyncio.to_thread(localizer.locate, frame)
    while (pose is None or pose.confidence < min_confidence) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.1)
        frame = camera.read()
        pose = await asyncio.to_thread(localizer.locate, frame)
    return frame, pose


async def _calibration_pulse(
    ble,
    action: Action,
    speed: int,
    duration_ms: int,
    settle_ms: int,
    on_pulse_completed: Callable[[], None] | None,
) -> None:
    await pulse(ble, action, speed, duration_ms, settle_ms)
    if on_pulse_completed is not None:
        on_pulse_completed()


def _inverse_action(action: Action) -> Action:
    inverse = {
        Action.FORWARD: Action.BACKWARD,
        Action.BACKWARD: Action.FORWARD,
        Action.TURN_LEFT: Action.TURN_RIGHT,
        Action.TURN_RIGHT: Action.TURN_LEFT,
    }
    if action not in inverse:
        raise ValueError(f"No exact calibration inverse for {action.value}")
    return inverse[action]


async def _recover_after_lost_motion(
    ble,
    camera,
    localizer: Localizer,
    *,
    failed_action: Action,
    speed: int,
    duration_ms: int,
    settle_ms: int,
    recovery_pulses: int,
    localization_timeout: float,
    min_confidence: float,
    on_pulse_completed: Callable[[], None] | None,
):
    """Undo the known last calibration action, then reacquire while stationary."""
    inverse = _inverse_action(failed_action)
    frame = camera.read()
    pose = None
    for _ in range(max(1, recovery_pulses)):
        await _calibration_pulse(
            ble, inverse, speed, duration_ms, settle_ms, on_pulse_completed
        )
        frame, pose = await _locate_until(
            camera, localizer, localization_timeout, min_confidence
        )
        if pose is not None and pose.confidence >= min_confidence:
            return frame, pose
    return frame, pose


@dataclass(slots=True)
class BodyToImage:
    px_per_forward_pulse: float
    radians_per_turn_pulse: float
    forward_axis_in_image: tuple[float, float]

    def direction(self, pose: RoverPose) -> tuple[float, float]:
        if pose.heading is not None:
            return math.cos(pose.heading), math.sin(pose.heading)
        axis = np.asarray(self.forward_axis_in_image, dtype=float)
        norm = float(np.linalg.norm(axis))
        return tuple(axis / norm) if norm else (1.0, 0.0)

    def predict(self, pose: RoverPose, action: Action) -> tuple[float, float]:
        x, y = pose.centre
        dx, dy = self.direction(pose)
        distance = self.px_per_forward_pulse
        if action == Action.FORWARD:
            return x + dx * distance, y + dy * distance
        if action == Action.BACKWARD:
            return x - dx * distance, y - dy * distance
        if action in (Action.ARC_LEFT, Action.ARC_RIGHT):
            sign = 1 if action == Action.ARC_LEFT else -1
            angle = sign * self.radians_per_turn_pulse * 0.5
            c, s = math.cos(angle), math.sin(angle)
            adx, ady = c * dx - s * dy, s * dx + c * dy
            return x + adx * distance * 0.75, y + ady * distance * 0.75
        return x, y


async def calibrate(
    ble,
    camera,
    localizer: Localizer,
    *,
    speed: int = 150,
    translation_ms: int = 400,
    turn_ms: int = 250,
    settle_ms: int = 250,
    repetitions: int = 3,
    noise_threshold_px: float = 2.0,
    minimum_margin_frac: float = 0.2,
    localization_timeout_seconds: float = 10.0,
    post_motion_timeout_seconds: float = 2.0,
    min_confidence: float = 0.25,
    return_to_start_each_sample: bool = False,
    minimum_valid_samples: int = 2,
    sample_retries: int = 1,
    angular_noise_threshold_degrees: float = 1.0,
    lost_recovery_pulses: int = 1,
    retry_duration_scale: float = 0.65,
    minimum_retry_pulse_ms: int = 150,
    on_pulse_completed: Callable[[], None] | None = None,
) -> BodyToImage:
    initial_frame, pose = await _locate_until(
        camera, localizer, localization_timeout_seconds, min_confidence
    )
    if pose is None or pose.confidence < min_confidence:
        detected_ids = getattr(localizer, "last_detected_ids", [])
        detail = f" Detected other marker IDs: {detected_ids}." if detected_ids else " No ArUco markers were detected."
        raise CalibrationError(
            "Calibration aborted after waiting for localization." + detail,
            initial_frame,
        )
    height, width = initial_frame.shape[:2]
    mx, my = width * minimum_margin_frac, height * minimum_margin_frac
    if not (mx <= pose.centre[0] <= width - mx and my <= pose.centre[1] <= height - my):
        x, y = pose.centre
        raise CalibrationError(
            "Calibration aborted: rover must be well inside the frame. "
            f"Marker centre=({x:.1f}, {y:.1f}); required x={mx:.1f}..{width-mx:.1f}, "
            f"y={my:.1f}..{height-my:.1f}.",
            initial_frame,
        )

    displacements: list[np.ndarray] = []
    displacement_norms: list[float] = []
    current = pose
    translation_duration = translation_ms
    for _ in range(repetitions + max(0, sample_retries)):
        before = current
        await _calibration_pulse(
            ble, Action.FORWARD, speed, translation_duration, settle_ms, on_pulse_completed
        )
        next_frame, next_pose = await _locate_until(
            camera, localizer, post_motion_timeout_seconds, min_confidence
        )
        if next_pose is None or next_pose.confidence < min_confidence:
            recovery_frame, recovery_pose = await _recover_after_lost_motion(
                ble, camera, localizer,
                failed_action=Action.FORWARD, speed=speed,
                duration_ms=translation_duration, settle_ms=settle_ms,
                recovery_pulses=lost_recovery_pulses,
                localization_timeout=post_motion_timeout_seconds,
                min_confidence=min_confidence,
                on_pulse_completed=on_pulse_completed,
            )
            if recovery_pose is None or recovery_pose.confidence < min_confidence:
                raise CalibrationError(
                    "Calibration aborted: marker stayed lost after reversing the forward pulse",
                    recovery_frame,
                )
            current = recovery_pose
            translation_duration = max(
                minimum_retry_pulse_ms,
                round(translation_duration * retry_duration_scale),
            )
            continue
        displacement = np.subtract(next_pose.centre, before.centre)
        # A shortened retry is scaled back to the configured pulse duration so
        # predictions remain valid for normal runtime actions.
        displacement = displacement * (translation_ms / translation_duration)
        displacement_norm = float(np.linalg.norm(displacement))
        displacement_norms.append(displacement_norm)
        if displacement_norm >= noise_threshold_px:
            displacements.append(displacement)
        current = next_pose
        if return_to_start_each_sample:
            await _calibration_pulse(
                ble, Action.BACKWARD, speed, translation_duration, settle_ms, on_pulse_completed
            )
            rollback_frame, rollback_pose = await _locate_until(
                camera, localizer, post_motion_timeout_seconds, min_confidence
            )
            if rollback_pose is None or rollback_pose.confidence < min_confidence:
                raise CalibrationError("Calibration aborted: marker lost while returning after forward sample", rollback_frame)
            current = rollback_pose
        if len(displacements) >= repetitions:
            break
    if len(displacements) < min(minimum_valid_samples, repetitions):
        raise CalibrationError(
            f"Calibration aborted: only {len(displacements)} valid forward samples; "
            f"need {min(minimum_valid_samples, repetitions)}. Measured pixels={displacement_norms}.",
            camera.read(),
        )
    mean_displacement = np.median(np.asarray(displacements), axis=0)
    pixels = float(np.linalg.norm(mean_displacement))
    if pixels < noise_threshold_px:
        raise CalibrationError(
            f"Calibration aborted: robust forward displacement {pixels:.2f}px is below "
            f"the {noise_threshold_px:.2f}px noise threshold; samples={displacement_norms}.",
            camera.read(),
        )
    axis = mean_displacement / pixels

    turn_angles: list[float] = []
    measured_turn_degrees: list[float] = []
    angular_threshold = math.radians(angular_noise_threshold_degrees)
    turn_duration = turn_ms
    for _ in range(repetitions + max(0, sample_retries)):
        before = current
        await _calibration_pulse(
            ble, Action.TURN_LEFT, speed, turn_duration, settle_ms, on_pulse_completed
        )
        turn_frame, current = await _locate_until(
            camera, localizer, post_motion_timeout_seconds, min_confidence
        )
        if current is None or current.confidence < min_confidence:
            recovery_frame, recovery_pose = await _recover_after_lost_motion(
                ble, camera, localizer,
                failed_action=Action.TURN_LEFT, speed=speed,
                duration_ms=turn_duration, settle_ms=settle_ms,
                recovery_pulses=lost_recovery_pulses,
                localization_timeout=post_motion_timeout_seconds,
                min_confidence=min_confidence,
                on_pulse_completed=on_pulse_completed,
            )
            if recovery_pose is None or recovery_pose.confidence < min_confidence:
                raise CalibrationError(
                    "Calibration aborted: marker stayed lost after reversing the left turn",
                    recovery_frame,
                )
            current = recovery_pose
            turn_duration = max(
                minimum_retry_pulse_ms,
                round(turn_duration * retry_duration_scale),
            )
            continue
        if before.heading is not None and current.heading is not None:
            angle = _wrap_angle(current.heading - before.heading)
            angle *= turn_ms / turn_duration
            measured_turn_degrees.append(math.degrees(angle))
            if abs(angle) >= angular_threshold:
                turn_angles.append(angle)
        if return_to_start_each_sample:
            await _calibration_pulse(
                ble, Action.TURN_RIGHT, speed, turn_duration, settle_ms, on_pulse_completed
            )
            rollback_frame, rollback_pose = await _locate_until(
                camera, localizer, post_motion_timeout_seconds, min_confidence
            )
            if rollback_pose is None or rollback_pose.confidence < min_confidence:
                raise CalibrationError("Calibration aborted: marker lost while returning after turn sample", rollback_frame)
            current = rollback_pose
        if len(turn_angles) >= repetitions:
            break
    if current.heading is not None and len(turn_angles) < min(minimum_valid_samples, repetitions):
        raise CalibrationError(
            f"Calibration aborted: only {len(turn_angles)} valid turn samples; "
            f"need {min(minimum_valid_samples, repetitions)}. "
            f"Measured degrees={measured_turn_degrees}.",
            camera.read(),
        )
    radians = float(np.median(turn_angles)) if turn_angles else math.radians(20)
    return BodyToImage(pixels, radians, (float(axis[0]), float(axis[1])))
