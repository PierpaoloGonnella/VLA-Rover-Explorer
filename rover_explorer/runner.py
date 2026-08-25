from __future__ import annotations

import sys

# If a dependency imports pythoncom later, request MTA before that import. The
# real BLE console has no Windows GUI message pump, so STA would deadlock WinRT.
if sys.platform == "win32":
    sys.coinit_flags = 0

import argparse
import asyncio
import contextlib
import time
from pathlib import Path

import cv2

from .annotate import draw_arrows, draw_grid, highlight_action
from .ble import MockBle, RoverBle
from .calibrate import BodyToImage, CalibrationError, calibrate
from .camera import ReplaySource, SimulatedSource, WebcamSource
from .config import AppConfig, load_config
from .coverage import CoverageTracker
from .guard import TRANSLATING_ACTIONS, ULTRASONIC_BLOCKED_ACTIONS, allowed_actions, apply_ultrasonic_guard
from .localize import ArucoLocalizer, ColorBlobLocalizer, VlmLocalizer
from .logger import SessionLogger
from .motion import Action, MotionWatchdog, pulse
from .policy import CoverageSweep, FrontierGreedy, ObstacleSweep, RandomWalk, VlmExplorer
from .simulator import RoverSimulator


async def _offload(function, /, *args):
    """Run blocking vision/HTTP work without starving safety tasks."""
    return await asyncio.to_thread(function, *args)


def _nominal_transform(config: AppConfig) -> BodyToImage:
    pixels = (
        config.simulator.px_per_second_at_full_speed
        * config.motion.speed / 255
        * config.motion.translation_ms / 1000
    )
    radians = (
        config.simulator.radians_per_second_at_full_speed
        * config.motion.speed / 255
        * config.motion.turn_ms / 1000
    )
    return BodyToImage(pixels, radians, (1.0, 0.0))


async def run(
    config: AppConfig,
    *,
    transport: str = "mock",
    localizer_name: str = "aruco",
    policy_name: str = "random",
    annotation_style: str = "arrows",
    cycles: int | None = None,
    session_dir: str | Path | None = None,
    do_calibrate: bool = True,
) -> dict:
    simulator = None
    if transport == "mock":
        simulator = RoverSimulator(
            config.camera.width, config.camera.height,
            config.simulator.wheel_slip, config.simulator.ble_latency_ms,
            config.simulator.px_per_second_at_full_speed,
            config.simulator.radians_per_second_at_full_speed,
            config.simulator.seed, config.localization.aruco_marker_id,
        )
        ble = MockBle(simulator)
        camera = SimulatedSource(simulator)
    elif transport == "ble":
        ble = RoverBle(
            config.ble.device_name, config.ble.characteristic_uuid,
            config.ble.reconnect_attempts, config.ble.backoff_seconds,
        )
        camera = WebcamSource(config.camera.index, config.camera.width, config.camera.height)
    elif transport == "replay":
        if session_dir is None:
            raise ValueError("--session-dir is required for replay")
        simulator = RoverSimulator(config.camera.width, config.camera.height, ble_latency_ms=0)
        ble = MockBle(simulator)
        camera = ReplaySource(session_dir)
        do_calibrate = False
    else:
        raise ValueError(f"Unknown transport {transport}")

    localizers = {
        "aruco": ArucoLocalizer(config.localization.aruco_marker_id),
        "color": ColorBlobLocalizer(config.localization.color_hsv_low, config.localization.color_hsv_high),
        "vlm": VlmLocalizer(config.ollama.url, config.ollama.model, config.localization.vlm_grid_cols, config.localization.vlm_grid_rows, config.ollama.timeout_seconds),
    }
    localizer = localizers[localizer_name]
    output = Path(session_dir or config.runner.session_dir) / time.strftime("%Y%m%d-%H%M%S")
    if transport == "replay":
        output = Path(str(session_dir) + "-replay-" + time.strftime("%Y%m%d-%H%M%S"))
    logger = SessionLogger(output)
    await ble.connect()
    watchdog = MotionWatchdog(ble, config.motion.watchdog_seconds)
    await watchdog.start()
    transform = _nominal_transform(config)
    try:
        if do_calibrate:
            try:
                transform = await calibrate(
                    ble, camera, localizer, speed=config.motion.speed,
                    translation_ms=config.motion.translation_ms, turn_ms=config.motion.turn_ms,
                    settle_ms=config.motion.settle_ms, repetitions=config.calibration.repetitions,
                    noise_threshold_px=config.calibration.noise_threshold_px,
                    minimum_margin_frac=config.calibration.minimum_margin_frac,
                    localization_timeout_seconds=config.calibration.localization_timeout_seconds,
                    post_motion_timeout_seconds=config.calibration.post_motion_timeout_seconds,
                    min_confidence=config.localization.min_confidence,
                    return_to_start_each_sample=config.calibration.return_to_start_each_sample,
                    minimum_valid_samples=config.calibration.minimum_valid_samples,
                    sample_retries=config.calibration.sample_retries,
                    angular_noise_threshold_degrees=config.calibration.angular_noise_threshold_degrees,
                    lost_recovery_pulses=config.calibration.lost_recovery_pulses,
                    retry_duration_scale=config.calibration.retry_duration_scale,
                    minimum_retry_pulse_ms=config.calibration.minimum_retry_pulse_ms,
                    on_pulse_completed=watchdog.pulse_completed,
                )
            except CalibrationError as exc:
                diagnostic = output / "calibration_failed.jpg"
                detail = str(exc)
                if exc.frame is not None:
                    cv2.imwrite(str(diagnostic), exc.frame)
                    height, width = exc.frame.shape[:2]
                    detail += f" Captured frame resolution={width}x{height}."
                (output / "calibration_error.txt").write_text(detail, encoding="utf-8")
                raise RuntimeError(f"{detail} Diagnostic frame saved to: {diagnostic}") from exc
            watchdog.pulse_completed()
        if policy_name == "random":
            policy = RandomWalk(config.simulator.seed)
        elif policy_name == "frontier":
            policy = FrontierGreedy(transform)
        elif policy_name == "vlm":
            policy = VlmExplorer(transform, config.ollama.url, config.ollama.model, config.ollama.timeout_seconds, annotation_style)
        elif policy_name == "sweep":
            policy = CoverageSweep(transform, config.guard.margin_frac)
        elif policy_name == "obstacle_sweep":
            ultrasonic = config.ultrasonic
            policy = ObstacleSweep(
                transform,
                config.guard.margin_frac,
                map_cols=ultrasonic.map_cols,
                map_rows=ultrasonic.map_rows,
                cm_per_translation_pulse=ultrasonic.cm_per_translation_pulse,
                rover_radius_cm=ultrasonic.rover_radius_cm,
                scan_angle_degrees=ultrasonic.scan_angle_degrees,
                maximum_mapping_distance_cm=ultrasonic.maximum_mapping_distance_cm,
                obstacle_ttl_cycles=ultrasonic.obstacle_ttl_cycles,
            )
        else:
            raise ValueError(f"Unknown policy {policy_name}")

        coverage: CoverageTracker | None = None
        cycle_limit = cycles if cycles is not None else config.runner.cycles
        for cycle in range(cycle_limit):
            cycle_started = time.monotonic()
            try:
                frame = camera.read()
            except EOFError:
                break
            if coverage is None:
                # Cameras may silently negotiate a different resolution. Safety
                # geometry and coverage must use the delivered frame dimensions.
                coverage = CoverageTracker(frame.shape, config.coverage.cols, config.coverage.rows)
            # VLM localization performs a blocking HTTP request; even classical
            # OpenCV localization is kept off the event loop for consistent timing.
            pose = await _offload(localizer.locate, frame)
            coverage.update(pose)
            if isinstance(policy, ObstacleSweep):
                policy.update_obstacles(
                    pose,
                    frame.shape,
                    coverage,
                    centre_cm=ble.sonar_cm,
                    left_cm=ble.sonar_left_cm,
                    right_cm=ble.sonar_right_cm,
                    blocked=ble.obstacle_blocked,
                    scan_sequence=ble.sonar_scan_sequence,
                )
            allowed = allowed_actions(pose, transform, frame.shape, config.guard.margin_frac)
            all_count = len(TRANSLATING_ACTIONS) + 3
            coverage.add_vetoes(all_count - len(allowed))
            if ble.obstacle_blocked:
                before_sonar_guard = len(allowed)
                allowed = apply_ultrasonic_guard(allowed, True)
                coverage.add_ultrasonic_vetoes(before_sonar_guard - len(allowed))
            if pose is not None and annotation_style == "arrows":
                annotated, _ = draw_arrows(frame, pose, transform, allowed, coverage, config.guard.margin_frac)
            else:
                annotated, _ = draw_grid(frame, pose, coverage)
            # Keep the event loop alive while Ollama thinks so the independent
            # motor watchdog can continue issuing fail-safe stops.
            decision = await _offload(
                policy.choose, frame, pose, allowed, coverage
            )
            if isinstance(policy, ObstacleSweep):
                annotated = policy.annotate(annotated)
            emergency_file = output / "EMERGENCY_STOP"
            if emergency_file.exists():
                decision.action = Action.STOP
                decision.reason = "Emergency stop requested by viewer."
            # The sonar state may change while a VLM request is in flight. Check
            # it again at the final decision boundary; the firmware independently
            # applies the same veto with much lower latency.
            if ble.obstacle_blocked and decision.action in ULTRASONIC_BLOCKED_ACTIONS:
                decision.action = Action.STOP
                decision.reason = "Ultrasonic obstacle stop; remote replanning required."
            duration = config.motion.turn_ms if decision.action in (Action.TURN_LEFT, Action.TURN_RIGHT) else config.motion.translation_ms
            if decision.action == Action.STOP:
                duration = 0
            await pulse(ble, decision.action, config.motion.speed, duration, config.motion.settle_ms)
            watchdog.pulse_completed()
            if pose is not None:
                annotated = highlight_action(annotated, pose, transform, decision.action)
            elapsed = time.monotonic() - cycle_started
            logger.log_cycle(
                cycle, frame, annotated,
                frame_timestamp=camera.timestamp,
                poses={localizer_name: pose.as_dict() if pose else None},
                allowed_actions=[a.value for a in allowed],
                decision=decision,
                raw_vlm_response=decision.raw_response,
                vlm_latency=decision.latency_seconds,
                pulse_sent={"action": decision.action.value, "speed": config.motion.speed, "duration_ms": duration},
                coverage=coverage.report(), battery_mv=ble.battery_mv,
                sonar_cm=ble.sonar_cm, sonar_left_cm=ble.sonar_left_cm,
                sonar_right_cm=ble.sonar_right_cm,
                sonar_scan_sequence=ble.sonar_scan_sequence,
                obstacle_blocked=ble.obstacle_blocked,
                cycle_latency_seconds=elapsed,
            )
            if coverage.fraction >= config.coverage.target:
                break
            remaining = config.runner.target_cycle_seconds - (time.monotonic() - cycle_started)
            if remaining > 0:
                await asyncio.sleep(remaining)
        if coverage is None:
            coverage = CoverageTracker(
                (config.camera.height, config.camera.width, 3),
                config.coverage.cols,
                config.coverage.rows,
            )
        return {**coverage.report(), "session_dir": str(output), "transform": transform}
    finally:
        with contextlib.suppress(Exception):
            await watchdog.stop()
        with contextlib.suppress(Exception):
            await ble.disconnect()
        camera.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe discrete VLA rover explorer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--transport", choices=("ble", "mock", "replay"), default="mock")
    parser.add_argument("--localizer", choices=("aruco", "color", "vlm"), default="aruco")
    parser.add_argument("--policy", choices=("vlm", "random", "frontier", "sweep", "obstacle_sweep"), default="random")
    parser.add_argument("--annotate", choices=("arrows", "grid"), default="arrows")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--session-dir")
    parser.add_argument("--no-calibrate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    try:
        report = asyncio.run(run(config, transport=args.transport, localizer_name=args.localizer, policy_name=args.policy, annotation_style=args.annotate, cycles=args.cycles, session_dir=args.session_dir, do_calibrate=not args.no_calibrate))
        print(report)
    except KeyboardInterrupt:
        print("Stopped safely.")
    except (RuntimeError, ConnectionError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
