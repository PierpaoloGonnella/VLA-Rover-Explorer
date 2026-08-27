from __future__ import annotations

import base64
import json
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np
import requests

from .annotate import draw_arrows, draw_grid
from .calibrate import BodyToImage
from .coverage import CoverageTracker
from .guard import safe_rectangle
from .localize import RoverPose
from .motion import Action
from .obstacle import ObstacleGrid


@dataclass(slots=True)
class Decision:
    action: Action
    reason: str
    confidence: float
    raw_response: str
    latency_seconds: float = 0.0
    target_cell: tuple[int, int] | None = None
    safe_to_advance: bool = False
    hazard_type: str = "unknown"
    hazard_cells: tuple[tuple[int, int], ...] = ()
    safe_cells: tuple[tuple[int, int], ...] = ()
    scene_description: str = ""
    objective: str = ""
    plan_steps: tuple[str, ...] = ()
    maneuver: str = "hold"


def semantic_clearance_is_valid(
    current_pose: RoverPose | None,
    source_pose: RoverPose | None,
    *,
    safe_to_advance: bool,
    source_age_seconds: float,
    maximum_age_seconds: float,
    maximum_motion_px: float,
    maximum_heading_change_radians: float,
) -> bool:
    """Reject semantic clearance that is unsafe, stale, or from another pose."""
    if not safe_to_advance or current_pose is None or source_pose is None:
        return False
    if source_age_seconds < 0 or source_age_seconds > maximum_age_seconds:
        return False
    if math.dist(current_pose.centre, source_pose.centre) > maximum_motion_px:
        return False
    if current_pose.heading is None or source_pose.heading is None:
        return False
    heading_change = abs(
        (current_pose.heading - source_pose.heading + math.pi) % (2 * math.pi) - math.pi
    )
    return heading_change <= maximum_heading_change_radians


def semantic_path_cells(
    pose: RoverPose | None,
    action: Action,
    transform: BodyToImage,
    frame_shape: tuple[int, ...],
    grid_cols: int,
    grid_rows: int,
    *,
    lookahead_pulses: float = 2.0,
    samples: int = 12,
) -> set[tuple[int, int]]:
    """Project a bounded translation into cells of a camera-fixed semantic map."""
    translating = {Action.FORWARD, Action.BACKWARD, Action.ARC_LEFT, Action.ARC_RIGHT}
    if pose is None or action not in translating:
        return set()
    height, width = frame_shape[:2]
    predicted = transform.predict(pose, action)
    factor = max(1.0, float(lookahead_pulses))
    end = (
        pose.centre[0] + (predicted[0] - pose.centre[0]) * factor,
        pose.centre[1] + (predicted[1] - pose.centre[1]) * factor,
    )
    count = max(2, int(samples))
    result: set[tuple[int, int]] = set()
    for index in range(count + 1):
        ratio = index / count
        x = pose.centre[0] + (end[0] - pose.centre[0]) * ratio
        y = pose.centre[1] + (end[1] - pose.centre[1]) * ratio
        if not (0 <= x < width and 0 <= y < height):
            return {(-1, -1)}
        result.add((int(x * grid_cols / width), int(y * grid_rows / height)))
    return result


def merge_semantic_hazards(
    previous: set[tuple[int, int]],
    previous_misses: dict[tuple[int, int], int],
    incoming: set[tuple[int, int]],
    clear_confirmations: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    """Add hazards immediately, but require repeated clean maps to remove them."""
    confirmations = max(1, int(clear_confirmations))
    retained = set(incoming)
    misses = {cell: 0 for cell in incoming}
    for cell in previous - incoming:
        count = previous_misses.get(cell, 0) + 1
        if count < confirmations:
            retained.add(cell)
            misses[cell] = count
    return retained, misses


def scene_report_supports_hazard(report: str, hazard_type: str) -> bool:
    """Require an unnegated raw-scene observation before accepting a hazard."""
    groups = {
        "stairs": ("stairs", "staircase", "steps", "stair risers"),
        "drop_off": ("drop-off", "drop off", "ledge", "lower level"),
        "ramp": ("ramp", "slope", "incline"),
        "wall": ("wall",),
        "obstacle": ("obstacle", "blocked", "furniture"),
        "person": ("person", "people", "leg", "human"),
    }
    requested = hazard_type.strip().lower()
    terms = groups.get(requested, ())
    if requested in {"stairs", "drop_off", "ramp"}:
        # Small VLMs often confuse these three terrain labels, but the raw
        # observer must still have seen some explicit terrain discontinuity.
        terms = groups["stairs"] + groups["drop_off"] + groups["ramp"]
    for sentence in re.split(r"[.!?;\n]+", report.lower()):
        for term in terms:
            start = sentence.find(term)
            while start >= 0:
                prefix = sentence[max(0, start - 48):start]
                negated = re.search(
                    r"\b(no|not|without|neither|absent|absence of)\b", prefix
                )
                if not negated:
                    return True
                start = sentence.find(term, start + len(term))
    return False


def scene_report_supports_clear_ground(report: str) -> bool:
    """Accept clearance only when the raw observer positively sees the floor."""
    text = " ".join(report.lower().split())
    if not text or not re.search(r"\b(floor|ground|terrain)\b", text):
        return False
    uncertain = (
        "floor structure is not clearly visible",
        "floor is not clearly visible",
        "floor is not visible",
        "ground is not clearly visible",
        "ground is not visible",
        "cannot see the floor",
        "can't see the floor",
        "unclear floor",
        "uncertain floor",
    )
    if any(phrase in text for phrase in uncertain):
        return False
    if any(
        scene_report_supports_hazard(text, kind)
        for kind in ("stairs", "drop_off", "ramp")
    ):
        return False
    return True


def update_severe_hazard_latch(
    latched: bool,
    clean_count: int,
    hazard_type: str,
    safe_to_advance: bool,
    clear_confirmations: int,
) -> tuple[bool, int]:
    """Latch terrain discontinuities until several consecutive clear reports."""
    if hazard_type in {"stairs", "drop_off", "ramp"}:
        return True, 0
    if not latched:
        return False, 0
    if hazard_type == "none" and safe_to_advance:
        count = clean_count + 1
        if count >= max(1, int(clear_confirmations)):
            return False, 0
        return True, count
    return True, 0


def bounded_pose_recovery_action(
    legal_reason: str,
    allowed: list[Action],
    now: float,
    next_allowed: float,
    retry_seconds: float,
) -> tuple[Action | None, float]:
    """Stop on stale localization; blind translation is unsafe near edges."""
    if "pose stale/lost" not in legal_reason:
        return None, 0.0
    return Action.STOP, now + max(0.5, float(retry_seconds))


class RandomWalk:
    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)

    def choose(self, frame, pose, allowed: list[Action], coverage: CoverageTracker) -> Decision:
        action = self.random.choice(allowed) if allowed else Action.STOP
        return Decision(action, "Uniform random legal action baseline.", 1.0, "")


class FrontierGreedy:
    def __init__(self, transform: BodyToImage):
        self.transform = transform

    def choose(self, frame, pose: RoverPose | None, allowed: list[Action], coverage: CoverageTracker) -> Decision:
        if pose is None:
            action = Action.BACKWARD if Action.BACKWARD in allowed else Action.STOP
            return Decision(action, "Recovering conservatively after localization loss.", 1.0, "")
        h, w = frame.shape[:2]
        candidates = [(c, r) for r in range(coverage.rows) for c in range(coverage.cols) if (c, r) not in coverage.visited]
        if not candidates:
            return Decision(Action.STOP, "All coverage cells have been visited.", 1.0, "")
        target = min(candidates, key=lambda cell: math.dist(pose.centre, ((cell[0]+.5)*w/coverage.cols, (cell[1]+.5)*h/coverage.rows)))
        target_px = ((target[0]+.5)*w/coverage.cols, (target[1]+.5)*h/coverage.rows)
        mobile = [a for a in allowed if a != Action.STOP]
        action = min(mobile or [Action.STOP], key=lambda a: math.dist(self.transform.predict(pose, a), target_px))
        return Decision(action, f"Moving toward nearest unvisited cell {chr(65+target[0])}{target[1]+1}.", 1.0, "")


class CoverageSweep:
    """Closed-loop boustrophedon (lawnmower) coverage in image coordinates."""

    def __init__(self, transform: BodyToImage, margin_frac: float = 0.12):
        self.transform = transform
        self.margin_frac = margin_frac
        self._next_waypoint = 0
        self._turn_action: Action | None = None
        self._turn_waypoint: int | None = None

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (value + math.pi) % (2 * math.pi) - math.pi

    def _waypoints(self, frame, coverage: CoverageTracker) -> list[tuple[float, float, tuple[int, int]]]:
        h, w = frame.shape[:2]
        # Stay one calibrated pulse inside the guard boundary, leaving room
        # for timing error and wheel slip.
        inset_x = w * self.margin_frac + self.transform.px_per_forward_pulse
        inset_y = h * self.margin_frac + self.transform.px_per_forward_pulse
        left, right = inset_x, w - inset_x
        top, bottom = inset_y, h - inset_y
        if left > right:
            left = right = w / 2
        if top > bottom:
            top = bottom = h / 2
        xs = np.linspace(left, right, coverage.cols)
        ys = np.linspace(top, bottom, coverage.rows)
        result: list[tuple[float, float, tuple[int, int]]] = []
        for row, y in enumerate(ys):
            row_xs = xs if row % 2 == 0 else xs[::-1]
            for x in row_xs:
                point = (float(x), float(y))
                result.append((point[0], point[1], coverage.cell_for(point)))
        return result

    def choose(self, frame, pose: RoverPose | None, allowed: list[Action], coverage: CoverageTracker) -> Decision:
        if pose is None:
            self._turn_action = None
            self._turn_waypoint = None
            return Decision(Action.STOP, "Pose lost during sweep; stopping.", 0.0, "")

        waypoints = self._waypoints(frame, coverage)
        # Move monotonically through the serpentine route, skipping cells that
        # were already crossed on the way to an earlier target.
        while self._next_waypoint < len(waypoints) and (
            waypoints[self._next_waypoint][2] in coverage.visited
            or waypoints[self._next_waypoint][2] in coverage.excluded
        ):
            self._next_waypoint += 1
        if self._next_waypoint >= len(waypoints):
            self._turn_action = None
            self._turn_waypoint = None
            return Decision(Action.STOP, "Coverage sweep complete.", 1.0, "")

        if self._turn_waypoint != self._next_waypoint:
            self._turn_action = None
            self._turn_waypoint = self._next_waypoint

        tx, ty, cell = waypoints[self._next_waypoint]
        label = f"{chr(65 + cell[0])}{cell[1] + 1}"
        dx, dy = tx - pose.centre[0], ty - pose.centre[1]
        distance = math.hypot(dx, dy)

        if pose.heading is not None:
            desired = math.atan2(dy, dx)
            error = self._wrap_angle(desired - pose.heading)
            turn_size = max(abs(self.transform.radians_per_turn_pulse), math.radians(2))
            tolerance = max(math.radians(8), turn_size * 0.55)
            if abs(error) > tolerance:
                # The calibrated value records the observed TURN_LEFT sign in
                # image coordinates, so this also handles inverted image axes.
                left_sign = 1 if self.transform.radians_per_turn_pulse >= 0 else -1
                preferred = Action.TURN_LEFT if error * left_sign > 0 else Action.TURN_RIGHT
                # Near +/-pi, tiny heading noise flips the shortest-turn sign.
                # Keep the previous direction only inside that wraparound band.
                # Everywhere else, allow a reversal after overshooting the
                # target so a single large pulse cannot become a full circle.
                wrap_band = max(tolerance, math.radians(15))
                near_wrap = abs(error) >= math.pi - wrap_band
                if self._turn_action is None or (
                    preferred != self._turn_action and not near_wrap
                ):
                    self._turn_action = preferred
                action = self._turn_action
                if action in allowed:
                    return Decision(action, f"Aligning to sweep waypoint {label} ({math.degrees(error):.1f} deg).", 1.0, "")
            else:
                self._turn_action = None

        if Action.FORWARD in allowed:
            return Decision(Action.FORWARD, f"Sweeping toward waypoint {label} ({distance:.0f}px away).", 1.0, "")

        turn = Action.TURN_LEFT if Action.TURN_LEFT in allowed else Action.TURN_RIGHT
        return Decision(turn if turn in allowed else Action.STOP, f"Boundary recovery while targeting {label}.", 1.0, "")


class WaypointFollower:
    """Fast closed-loop follower for a slowly changing semantic waypoint."""

    def __init__(self, transform: BodyToImage):
        self.transform = transform

    def choose(
        self,
        pose: RoverPose | None,
        target: tuple[float, float],
        allowed: list[Action],
        label: str,
    ) -> Decision:
        if pose is None or pose.heading is None:
            return Decision(Action.STOP, f"Pose unavailable for semantic waypoint {label}.", 0.0, "")
        desired = math.atan2(target[1] - pose.centre[1], target[0] - pose.centre[0])
        error = (desired - pose.heading + math.pi) % (2 * math.pi) - math.pi
        turn_size = max(abs(self.transform.radians_per_turn_pulse), math.radians(2))
        tolerance = max(math.radians(8), turn_size * .55)
        if abs(error) > tolerance:
            left_sign = 1 if self.transform.radians_per_turn_pulse >= 0 else -1
            action = Action.TURN_LEFT if error * left_sign > 0 else Action.TURN_RIGHT
            if action in allowed:
                return Decision(
                    action,
                    f"Aligning to semantic waypoint {label} ({math.degrees(error):.1f} deg).",
                    1.0,
                    "",
                )
        if Action.FORWARD in allowed:
            return Decision(Action.FORWARD, f"Following semantic waypoint {label}.", 1.0, "")
        return Decision(Action.STOP, f"Semantic waypoint {label} is temporarily guarded.", 0.0, "")


class BottomCenterKeeper:
    """Closed-loop policy that tracks a safe bottom-centre image waypoint."""

    def __init__(
        self,
        transform: BodyToImage | None = None,
        *,
        margin_frac: float = 0.12,
        safety_offset_px: float = 8.0,
        rover_rear_extent_px: float = 95.0,
        hold_enter_radius_px: float = 30.0,
        hold_exit_radius_px: float = 60.0,
        maximum_data_age_seconds: float = 1.0,
    ) -> None:
        self.transform = transform
        self.margin_frac = max(0.0, min(0.45, float(margin_frac)))
        self.safety_offset_px = max(2.0, float(safety_offset_px))
        self.rover_rear_extent_px = max(0.0, float(rover_rear_extent_px))
        self.hold_enter_radius_px = max(2.0, float(hold_enter_radius_px))
        self.hold_exit_radius_px = max(
            self.hold_enter_radius_px + 1.0, float(hold_exit_radius_px)
        )
        self.maximum_data_age_seconds = max(0.05, float(maximum_data_age_seconds))
        self._turn_action: Action | None = None
        self._holding = False

    @staticmethod
    def _fallback(reason: str, allowed: list[Action]) -> Decision:
        action = Action.STOP if Action.STOP in allowed else (
            Action.BACKWARD if Action.BACKWARD in allowed else Action.STOP
        )
        return Decision(
            action=action,
            reason=reason,
            confidence=0.0,
            raw_response="",
            safe_to_advance=False,
            hazard_type="unknown",
            objective="stay_bottom_center",
            plan_steps=("Wait for fresh localization and calibration data.",),
            maneuver="hold" if action == Action.STOP else "reverse",
        )

    @staticmethod
    def _transform_valid(transform: BodyToImage) -> bool:
        values = (
            transform.px_per_forward_pulse,
            transform.radians_per_turn_pulse,
            *transform.forward_axis_in_image,
        )
        return (
            all(math.isfinite(float(value)) for value in values)
            and float(transform.px_per_forward_pulse) > 0
            and abs(float(transform.radians_per_turn_pulse)) >= math.radians(1)
        )

    def choose(
        self,
        frame,
        pose: RoverPose | None,
        allowed: list[Action],
        coverage: CoverageTracker,
        transform: BodyToImage | None = None,
        *,
        localization_age_seconds: float = 0.0,
        transform_age_seconds: float = 0.0,
    ) -> Decision:
        active_transform = transform or self.transform
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return self._fallback("Frame unavailable for bottom-centre tracking.", allowed)
        if not allowed:
            return self._fallback("Guard supplied no allowed actions.", [Action.STOP])
        if pose is None or pose.heading is None or pose.confidence <= 0:
            return self._fallback("Localization unavailable for bottom-centre tracking.", allowed)
        if (
            not all(math.isfinite(float(value)) for value in (*pose.centre, pose.heading))
            or localization_age_seconds < 0
            or localization_age_seconds > self.maximum_data_age_seconds
        ):
            return self._fallback("Localization is invalid or stale; bottom-centre plan rejected.", allowed)
        if (
            active_transform is None
            or not self._transform_valid(active_transform)
            or transform_age_seconds < 0
            or transform_age_seconds > self.maximum_data_age_seconds
        ):
            return self._fallback("Body-to-image calibration is invalid or stale.", allowed)

        height, width = frame.shape[:2]
        pulse_px = float(active_transform.px_per_forward_pulse)
        # Keep the whole chassis in view, not only the centre of the ArUco
        # marker. The rear extent was measured from the real 1280x720 logs.
        margin_px = pulse_px + self.rover_rear_extent_px + self.safety_offset_px
        raw_target = (width / 2.0, height - margin_px)
        left, top, right, bottom = safe_rectangle(frame.shape, self.margin_frac)
        inset = self.safety_offset_px
        safe_left, safe_right = left + inset, right - inset
        safe_top, safe_bottom = top + inset, bottom - inset
        if safe_left > safe_right or safe_top > safe_bottom:
            return self._fallback("Safe guard rectangle is too small for the calibrated rover pulse.", allowed)
        target = (
            min(safe_right, max(safe_left, raw_target[0])),
            min(safe_bottom, max(safe_top, raw_target[1])),
        )
        target_was_clamped = target != raw_target
        target_cell = (
            min(coverage.cols - 1, max(0, int(target[0] * coverage.cols / width))),
            min(coverage.rows - 1, max(0, int(target[1] * coverage.rows / height))),
        )

        dx, dy = target[0] - pose.centre[0], target[1] - pose.centre[1]
        distance = math.hypot(dx, dy)
        if self._holding:
            if distance <= self.hold_exit_radius_px:
                self._turn_action = None
                return Decision(
                    action=Action.STOP,
                    reason=(
                        f"Holding bottom-centre waypoint inside exit radius; "
                        f"distance={distance:.0f}px, exit={self.hold_exit_radius_px:.0f}px."
                    ),
                    confidence=max(0.0, min(1.0, pose.confidence)),
                    raw_response="",
                    target_cell=target_cell,
                    safe_to_advance=False,
                    hazard_type="none",
                    objective="stay_bottom_center",
                    plan_steps=("Hold position.", "Resume only outside the exit radius."),
                    maneuver="hold",
                )
            self._holding = False
        if distance <= self.hold_enter_radius_px:
            self._holding = True
            self._turn_action = None
            return Decision(
                action=Action.STOP,
                reason=(
                    f"Entered bottom-centre hold radius; distance={distance:.0f}px, "
                    f"enter={self.hold_enter_radius_px:.0f}px."
                ),
                confidence=max(0.0, min(1.0, pose.confidence)),
                raw_response="",
                target_cell=target_cell,
                safe_to_advance=False,
                hazard_type="none",
                objective="stay_bottom_center",
                plan_steps=("Hold position.", "Ignore heading noise while inside hysteresis."),
                maneuver="hold",
            )

        desired = math.atan2(dy, dx)
        error = (desired - pose.heading + math.pi) % (2 * math.pi) - math.pi
        turn_size = max(abs(active_transform.radians_per_turn_pulse), math.radians(2))
        tolerance = max(math.radians(8), turn_size * 0.55)
        wrap_band = max(tolerance * 1.5, math.radians(15))
        clamped_note = " Target clamped to the nearest safe inset." if target_was_clamped else ""

        if abs(error) > tolerance:
            left_sign = 1 if active_transform.radians_per_turn_pulse >= 0 else -1
            preferred = Action.TURN_LEFT if error * left_sign > 0 else Action.TURN_RIGHT
            near_wrap = abs(error) >= math.pi - wrap_band
            if self._turn_action is None or (
                preferred != self._turn_action and not near_wrap
            ):
                self._turn_action = preferred
            action = self._turn_action
            predicted = active_transform.predict(pose, action)
            if action in allowed and (
                left <= predicted[0] <= right and top <= predicted[1] <= bottom
            ):
                return Decision(
                    action=action,
                    reason=(
                        f"Aligning with bottom-centre waypoint at ({target[0]:.0f},"
                        f" {target[1]:.0f}); heading error={math.degrees(error):.1f} deg."
                        f"{clamped_note}"
                    ),
                    confidence=max(0.0, min(1.0, pose.confidence)),
                    raw_response="",
                    target_cell=target_cell,
                    safe_to_advance=False,
                    hazard_type="none",
                    objective="stay_bottom_center",
                    plan_steps=("Align heading with the waypoint.", "Advance after alignment."),
                    maneuver="turn_left" if action == Action.TURN_LEFT else "turn_right",
                )
        else:
            self._turn_action = None

        predicted_forward = active_transform.predict(pose, Action.FORWARD)
        forward_improves = math.dist(predicted_forward, target) + 1.0 < distance
        forward_inside = (
            left <= predicted_forward[0] <= right and top <= predicted_forward[1] <= bottom
        )
        if Action.FORWARD in allowed and forward_improves and forward_inside:
            return Decision(
                action=Action.FORWARD,
                reason=(
                    f"Advancing toward bottom-centre waypoint at ({target[0]:.0f},"
                    f" {target[1]:.0f}); predicted remaining distance="
                    f"{math.dist(predicted_forward, target):.0f}px.{clamped_note}"
                ),
                confidence=max(0.0, min(1.0, pose.confidence)),
                raw_response="",
                target_cell=target_cell,
                safe_to_advance=True,
                hazard_type="none",
                objective="stay_bottom_center",
                plan_steps=("Advance one calibrated pulse.", "Recompute from the next pose."),
                maneuver="advance",
            )

        guarded_turn = next(
            (candidate for candidate in (self._turn_action, Action.TURN_LEFT, Action.TURN_RIGHT)
             if candidate is not None and candidate in allowed),
            Action.STOP,
        )
        obstacle_blocked = Action.FORWARD not in allowed
        return Decision(
            action=guarded_turn,
            reason=(
                "Forward progress toward bottom-centre is blocked by the latest guard; "
                f"yielding {guarded_turn.value}."
            ),
            confidence=max(0.0, min(1.0, pose.confidence)) if guarded_turn != Action.STOP else 0.0,
            raw_response="",
            target_cell=target_cell,
            safe_to_advance=False,
            hazard_type="obstacle" if obstacle_blocked else "unknown",
            objective="stay_bottom_center",
            plan_steps=("Respect the current guard.", "Turn if legal; otherwise hold."),
            maneuver=(guarded_turn.value if guarded_turn != Action.STOP else "hold"),
        )


class ObstacleSweep(CoverageSweep):
    """Lawnmower coverage with ultrasonic occupancy mapping and A* detours."""

    def __init__(
        self,
        transform: BodyToImage,
        margin_frac: float = .12,
        *,
        map_cols: int = 12,
        map_rows: int = 8,
        cm_per_translation_pulse: float = 10.0,
        rover_radius_cm: float = 12.0,
        scan_angle_degrees: float = 50.0,
        maximum_mapping_distance_cm: int = 150,
        obstacle_ttl_cycles: int = 40,
    ):
        super().__init__(transform, margin_frac)
        self.map_cols = map_cols
        self.map_rows = map_rows
        self.cm_per_translation_pulse = cm_per_translation_pulse
        self.rover_radius_cm = rover_radius_cm
        self.scan_angle = math.radians(scan_angle_degrees)
        self.maximum_mapping_distance_cm = maximum_mapping_distance_cm
        self.obstacle_ttl_cycles = obstacle_ttl_cycles
        self.grid: ObstacleGrid | None = None
        self._map_cycle = 0
        self._last_side_scan_sequence = 0
        self._required_scan_sequence = 0
        self._was_blocked = False
        self._obstacle_blocked = False
        self._scan_ready = True
        self._last_path: list[tuple[int, int]] = []

    def _ensure_grid(self, frame_shape: tuple[int, ...]) -> ObstacleGrid:
        if self.grid is None or self.grid.frame_shape[:2] != frame_shape[:2]:
            self.grid = ObstacleGrid(
                frame_shape,
                self.map_cols,
                self.map_rows,
                self.obstacle_ttl_cycles,
            )
        return self.grid

    def update_obstacles(
        self,
        pose: RoverPose | None,
        frame_shape: tuple[int, ...],
        coverage: CoverageTracker,
        *,
        centre_cm: int | None,
        left_cm: int | None,
        right_cm: int | None,
        blocked: bool,
        scan_sequence: int,
    ) -> None:
        grid = self._ensure_grid(frame_shape)
        self._map_cycle += 1
        grid.prune(self._map_cycle)
        if blocked and not self._was_blocked:
            self._required_scan_sequence = scan_sequence + 1
        self._obstacle_blocked = blocked
        self._scan_ready = not blocked or scan_sequence >= self._required_scan_sequence
        self._was_blocked = blocked
        if pose is None:
            return

        pixels_per_cm = self.transform.px_per_forward_pulse / self.cm_per_translation_pulse
        left_sign = 1 if self.transform.radians_per_turn_pulse >= 0 else -1

        def observe(distance_cm: int | None, angle: float) -> None:
            if distance_cm is None or distance_cm <= 0:
                return
            clipped_cm = min(distance_cm, self.maximum_mapping_distance_cm)
            distance_px = clipped_cm * pixels_per_cm
            endpoint = (
                pose.centre[0] + math.cos(angle) * distance_px,
                pose.centre[1] + math.sin(angle) * distance_px,
            )
            endpoint_visible = 0 <= endpoint[0] < frame_shape[1] and 0 <= endpoint[1] < frame_shape[0]
            hit = distance_cm < self.maximum_mapping_distance_cm and endpoint_visible
            hit_cell = grid.observe_ray(
                pose.centre,
                angle,
                distance_px,
                hit=hit,
                cycle=self._map_cycle,
            )
            if hit and hit_cell is not None:
                coverage.exclude(coverage.cell_for(grid.centre(hit_cell)))

        heading = pose.heading
        if heading is None:
            direction = self.transform.direction(pose)
            heading = math.atan2(direction[1], direction[0])
        observe(centre_cm, heading)
        if scan_sequence > 0 and scan_sequence != self._last_side_scan_sequence:
            observe(left_cm, heading + left_sign * self.scan_angle)
            observe(right_cm, heading - left_sign * self.scan_angle)
            self._last_side_scan_sequence = scan_sequence

    def _inflation_cells(self) -> int:
        if self.grid is None:
            return 0
        pixels_per_cm = self.transform.px_per_forward_pulse / self.cm_per_translation_pulse
        cell_size = min(self.grid.width / self.grid.cols, self.grid.height / self.grid.rows)
        return max(1, math.ceil(self.rover_radius_cm * pixels_per_cm / cell_size))

    def _blocked_cells(self, frame_shape: tuple[int, ...]) -> set[tuple[int, int]]:
        assert self.grid is not None
        blocked = self.grid.occupied(self._inflation_cells())
        left, top, right, bottom = safe_rectangle(frame_shape, self.margin_frac)
        uncertainty = max(2.0, self.transform.px_per_forward_pulse)
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                x, y = self.grid.centre((col, row))
                if not (
                    left + uncertainty <= x <= right - uncertainty
                    and top + uncertainty <= y <= bottom - uncertainty
                ):
                    blocked.add((col, row))
        return blocked

    def choose(self, frame, pose: RoverPose | None, allowed: list[Action], coverage: CoverageTracker) -> Decision:
        if pose is None:
            return Decision(Action.STOP, "Pose lost during obstacle-aware sweep; stopping.", 0.0, "")
        if self._obstacle_blocked and not self._scan_ready:
            return Decision(Action.STOP, "Obstacle detected; waiting for stationary left-centre-right scan.", 1.0, "")

        grid = self._ensure_grid(frame.shape)
        waypoints = self._waypoints(frame, coverage)
        while self._next_waypoint < len(waypoints) and (
            waypoints[self._next_waypoint][2] in coverage.visited
            or waypoints[self._next_waypoint][2] in coverage.excluded
        ):
            self._next_waypoint += 1
        if self._next_waypoint >= len(waypoints):
            self._last_path = []
            return Decision(Action.STOP, "All reachable sweep cells are complete.", 1.0, "")

        tx, ty, coverage_cell = waypoints[self._next_waypoint]
        start = grid.cell_for(pose.centre)
        goal = grid.cell_for((tx, ty))
        blocked_cells = self._blocked_cells(frame.shape)
        if goal in blocked_cells:
            free_cells = [
                (col, row)
                for row in range(grid.rows)
                for col in range(grid.cols)
                if (col, row) not in blocked_cells
            ]
            if free_cells:
                goal = min(free_cells, key=lambda cell: math.dist(grid.centre(cell), (tx, ty)))
        path = grid.astar(start, goal, blocked_cells)
        if path is None:
            self._last_path = []
            label = f"{chr(65 + coverage_cell[0])}{coverage_cell[1] + 1}"
            return Decision(Action.STOP, f"No safe A* route to {label}; waiting for a new scan.", 1.0, "")
        self._last_path = path
        target = grid.centre(path[1]) if len(path) > 1 else (tx, ty)
        label = f"{chr(65 + coverage_cell[0])}{coverage_cell[1] + 1}"
        dx, dy = target[0] - pose.centre[0], target[1] - pose.centre[1]
        desired = math.atan2(dy, dx)
        heading = pose.heading
        if heading is None:
            return Decision(Action.STOP, f"Heading unavailable while planning A* route to {label}.", 0.0, "")
        error = self._wrap_angle(desired - heading)
        turn_size = max(abs(self.transform.radians_per_turn_pulse), math.radians(2))
        tolerance = max(math.radians(8), turn_size * .55)
        if abs(error) > tolerance:
            left_sign = 1 if self.transform.radians_per_turn_pulse >= 0 else -1
            action = Action.TURN_LEFT if error * left_sign > 0 else Action.TURN_RIGHT
            if action in allowed:
                return Decision(action, f"A* alignment toward {label} ({math.degrees(error):.1f} deg).", 1.0, "")
        if Action.FORWARD in allowed:
            return Decision(Action.FORWARD, f"Following A* detour toward sweep waypoint {label}.", 1.0, "")
        return Decision(Action.STOP, f"A* route to {label} is temporarily blocked by sonar.", 1.0, "")

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        if self.grid is None:
            return frame
        result = frame.copy()
        overlay = result.copy()
        for cell in self.grid.occupied(self._inflation_cells()):
            col, row = cell
            p1 = (round(col * self.grid.width / self.grid.cols), round(row * self.grid.height / self.grid.rows))
            p2 = (round((col + 1) * self.grid.width / self.grid.cols), round((row + 1) * self.grid.height / self.grid.rows))
            cv2.rectangle(overlay, p1, p2, (0, 0, 255), -1)
        cv2.addWeighted(overlay, .28, result, .72, 0, result)
        if len(self._last_path) > 1:
            points = np.asarray([self.grid.centre(cell) for cell in self._last_path], dtype=np.int32)
            cv2.polylines(result, [points], False, (255, 255, 0), 3, cv2.LINE_AA)
        return result


class VlmExplorer:
    SYSTEM_PROMPT = (
        "A separate observer has already inspected the unmodified camera frame. Use its scene report as the "
        "source for physical geometry. The provided image contains a synthetic coordinate overlay used only "
        "to locate the rover, direction, target, and grid cells. Never interpret overlay lines, labels, arrows, "
        "or colours as physical scene geometry. Do not invent a hazard absent from the scene report. "
        "Follow the response schema exactly. "
        "Interpret visible walls, furniture, people, objects, and traversable floor. "
        "Treat stairs, downward ramps, ledges, balconies, and any floor discontinuity as a severe hazard. "
        "Never direct the rover toward a drop-off. If the immediate corridor ahead is hazardous or uncertain, "
        "set safe_to_advance=false and cell=STOP. "
        "Maintain an explicit navigation objective and a short plan. Compare the present scene with the temporal "
        "history and the observed outcome of earlier motor pulses. If progress is negligible, actions repeat, or "
        "motors were commanded without displacement, diagnose a stall and change target or maneuver; never repeat "
        "a failed plan indefinitely. Treat the listed legal actions as hard constraints. Mark stairs and drop-offs "
        "as non-traversable cells and choose a target whose route avoids them. For an unfamiliar situation, prefer "
        "a reversible information-gathering maneuver or HOLD over an unsupported translation. Visit unvisited grid "
        "cells; shaded cells are already visited. Prefer open space and use sensor context to avoid blocked sides. "
        "Never invent a choice."
    )

    def __init__(
        self,
        transform: BodyToImage,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5vl:3b",
        timeout: float = 20.0,
        annotation_style: str = "arrows",
        requester: Callable[..., Any] | None = None,
        keep_alive: str = "30m",
    ):
        self.transform = transform
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.annotation_style = annotation_style
        self.requester = requester or requests.post
        self.keep_alive = keep_alive
        self.fallback_count = 0
        self.last_annotated_frame = None
        self.last_scene_description = ""

    @staticmethod
    def _jpeg_512(frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        scale = min(1.0, 512 / max(h, w))
        resized = cv2.resize(frame, (round(w*scale), round(h*scale))) if scale < 1 else frame
        ok, data = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def _fallback(allowed: list[Action], reason: str, raw: str, latency: float) -> Decision:
        # A malformed answer or network timeout is not spatial uncertainty: the
        # safest response is to remain stationary. A turn is used only when STOP
        # is unexpectedly absent from the guard-approved set.
        turns = [a for a in (Action.TURN_LEFT, Action.TURN_RIGHT) if a in allowed]
        action = Action.STOP if Action.STOP in allowed else (turns[0] if turns else allowed[0])
        return Decision(action, reason, 0.0, raw, latency)

    def choose(
        self,
        frame,
        pose: RoverPose | None,
        allowed: list[Action],
        coverage: CoverageTracker,
        sensor_context: str = "",
        temporal_context: str = "",
        prior_frames: list[np.ndarray] | None = None,
    ) -> Decision:
        if pose is None or not allowed:
            return self._fallback(allowed or [Action.STOP], "No reliable pose; conservative fallback.", "", 0.0)
        if self.annotation_style == "grid":
            self.last_scene_description = ""
            annotated, cells = draw_grid(frame, pose, coverage)
            self.last_annotated_frame = annotated.copy()
            cell_labels = sorted(cells)
            properties = {
                "cell": {"type": "string", "enum": [*cell_labels, "STOP"]},
                "safe_to_advance": {"type": "boolean"},
                "hazard_type": {
                    "type": "string",
                    "enum": ["none", "stairs", "drop_off", "ramp", "wall", "obstacle", "person", "unknown"],
                },
                "hazard_cells": {"type": "array", "items": {"type": "string", "enum": cell_labels}},
                "safe_cells": {"type": "array", "items": {"type": "string", "enum": cell_labels}},
                "objective": {"type": "string"},
                "plan_steps": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 1, "maxItems": 5,
                },
                "maneuver": {
                    "type": "string",
                    "enum": ["advance", "reverse", "turn_left", "turn_right", "hold", "replan"],
                },
                "reason": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            }
            required = [
                "cell", "safe_to_advance", "hazard_type", "hazard_cells",
                "safe_cells", "objective", "plan_steps", "maneuver", "reason", "confidence",
            ]
            instruction = (
                "Use the scene observer report below for geometry and this overlay only for coordinates. "
                "Assess whether the arrow from the rover shows a safe immediate corridor. "
                "Mark every grid cell containing stairs, a drop-off, downward ramp, ledge, or obstacle. "
                "hazard_type describes the immediate corridor in front, not hazards elsewhere in the image. "
                "A drop-off requires visible geometric evidence such as a floor edge, missing floor continuation, "
                "or repeated stair risers. Tile joints, grout lines, shadows, blur, and furniture edges are not "
                "drop-offs. Do not infer a hazard only because the prompt mentions it. Always return a persistent "
                "objective, 1-5 ordered plan steps, and the next bounded maneuver. If temporal evidence says the "
                "previous maneuver stalled or repeated, explicitly replan to a different target/maneuver. Choose "
                "a safe reachable target cell, or STOP if no safe route exists. Return JSON only."
            )
        else:
            annotated, mapping = draw_arrows(frame, pose, self.transform, allowed, coverage)
            self.last_annotated_frame = annotated.copy()
            # Constrain the choice to the labels that are actually present in
            # the image.  An unconstrained integer lets some vision models emit
            # internal-looking visual token IDs instead of an arrow number.
            legal_choices = sorted(mapping)
            properties = {
                "choice": {"type": "integer", "enum": legal_choices},
                "reason": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            }
            required = ["choice", "reason", "confidence"]
            legend = ", ".join(f"{number}={action.value}" for number, action in mapping.items())
            instruction = (
                f"Choose exactly one numbered arrow. Legal choices: {legend}. Return JSON only."
            )
        if sensor_context:
            instruction = f"{instruction} Sensor context: {sensor_context}"
        if temporal_context:
            instruction = f"{instruction}\nTEMPORAL MEMORY AND EXECUTION FEEDBACK:\n{temporal_context}"
        started = time.monotonic()
        raw = ""
        scene_description = ""
        if self.annotation_style == "grid":
            history = list(prior_frames or [])[-2:]
            observer_images = [self._jpeg_512(value) for value in history] + [self._jpeg_512(frame)]
            observer_payload = {
                "model": self.model,
                "stream": False,
                "keep_alive": self.keep_alive,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"You receive {len(observer_images)} unmodified frames ordered oldest to newest; the final "
                        "frame is the present. Compare them and describe changes, the current room and visible floor "
                        "structure. Mention stairs or a drop-off only if actually visible. State whether commanded "
                        "motion appears to have produced progress. Do not plan motion."
                    ),
                    "images": observer_images,
                }],
            }
            try:
                observer_response = self.requester(
                    f"{self.url}/api/chat", json=observer_payload, timeout=self.timeout
                )
                observer_response.raise_for_status()
                scene_description = str(
                    observer_response.json()["message"]["content"]
                ).strip()
                if not scene_description:
                    raise ValueError("empty scene description")
            except (requests.RequestException, TimeoutError, KeyError, TypeError, ValueError) as exc:
                self.fallback_count += 1
                return self._fallback(
                    allowed,
                    f"VLM scene observer failure ({type(exc).__name__}); conservative fallback.",
                    "",
                    time.monotonic() - started,
                )
            self.last_scene_description = scene_description
            instruction = f"Scene observer report: {scene_description}\n{instruction}"
        payload = {
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "format": {"type": "object", "properties": properties, "required": required},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": instruction,
                    "images": [self._jpeg_512(annotated)],
                },
            ],
        }
        try:
            response = self.requester(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            outer = response.json()
            raw_value = outer.get("message", {}).get("content", outer.get("response", ""))
            raw = raw_value if isinstance(raw_value, str) else json.dumps(raw_value)
            answer = raw_value if isinstance(raw_value, dict) else json.loads(raw_value)
            if self.annotation_style == "grid":
                cell = str(answer["cell"]).upper()
                if cell != "STOP" and cell not in cells:
                    raise ValueError("out-of-range grid cell")
                hazard_labels = {str(value).upper() for value in answer["hazard_cells"]}
                safe_labels = {str(value).upper() for value in answer["safe_cells"]}
                if not hazard_labels.issubset(cells) or not safe_labels.issubset(cells):
                    raise ValueError("out-of-range semantic cell")
                hazard_cells = tuple(cells[label] for label in sorted(hazard_labels))
                safe_cells = tuple(
                    cells[label] for label in sorted(safe_labels - hazard_labels)
                )
                hazard_type = str(answer["hazard_type"]).lower()
                known_hazards = {
                    "none", "stairs", "drop_off", "ramp", "wall",
                    "obstacle", "person", "unknown",
                }
                if hazard_type not in known_hazards:
                    raise ValueError("unknown hazard type")
                reason = str(answer["reason"])
                objective = str(answer["objective"]).strip()
                plan_steps = tuple(str(step).strip() for step in answer["plan_steps"] if str(step).strip())
                maneuver = str(answer["maneuver"]).lower()
                if not objective or not plan_steps:
                    raise ValueError("missing objective or plan")
                unsupported_hazard = False
                observer_clear = scene_report_supports_clear_ground(scene_description)
                if hazard_type not in {"none", "unknown"} and not scene_report_supports_hazard(
                    scene_description, hazard_type
                ):
                    reason = (
                        f"Rejected unsupported {hazard_type} from coordinate planner; "
                        f"raw-scene observer reported: {scene_description}"
                    )
                    hazard_type = "none"
                    hazard_cells = ()
                    unsupported_hazard = True
                safe_to_advance = bool(answer["safe_to_advance"])
                force_stop = False
                if hazard_type == "none" and not observer_clear:
                    hazard_type = "unknown"
                    hazard_cells = ()
                    safe_to_advance = False
                    force_stop = True
                    reason = (
                        "Raw-scene observer did not positively confirm visible clear ground; "
                        f"reported: {scene_description}"
                    )
                elif hazard_type != "none":
                    safe_to_advance = False
                    force_stop = hazard_type in {"stairs", "drop_off", "ramp", "unknown"}
                elif unsupported_hazard:
                    safe_to_advance = True
                elif cell == "STOP":
                    safe_to_advance = False
                if cell == "STOP" or force_stop:
                    action = Action.STOP
                    target_cell = None
                else:
                    col, row = cells[cell]
                    h, w = frame.shape[:2]
                    target = ((col+.5)*w/coverage.cols, (row+.5)*h/coverage.rows)
                    action = min(allowed, key=lambda a: math.dist(self.transform.predict(pose, a), target))
                    target_cell = (col, row)
            else:
                choice = int(answer["choice"])
                if choice not in mapping:
                    raise ValueError("out-of-range choice")
                action = mapping[choice]
                target_cell = None
                safe_to_advance = False
                hazard_type = "unknown"
                hazard_cells = ()
                safe_cells = ()
                reason = str(answer["reason"])
                objective = "Choose one safe legal motion."
                plan_steps = (reason,)
                maneuver = action.value
            latency = time.monotonic() - started
            return Decision(
                action,
                reason,
                max(0.0, min(1.0, float(answer["confidence"]))),
                raw,
                latency,
                target_cell,
                safe_to_advance,
                hazard_type,
                hazard_cells,
                safe_cells,
                scene_description,
                objective,
                plan_steps,
                maneuver,
            )
        except (requests.RequestException, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.fallback_count += 1
            return self._fallback(allowed, f"VLM failure ({type(exc).__name__}); conservative fallback.", raw, time.monotonic()-started)
