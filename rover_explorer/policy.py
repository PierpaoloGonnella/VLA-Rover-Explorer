from __future__ import annotations

import base64
import json
import math
import random
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
            return Decision(Action.STOP, "Coverage sweep complete.", 1.0, "")

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
                action = Action.TURN_LEFT if error * left_sign > 0 else Action.TURN_RIGHT
                if action in allowed:
                    return Decision(action, f"Aligning to sweep waypoint {label} ({math.degrees(error):.1f} deg).", 1.0, "")

        if Action.FORWARD in allowed:
            return Decision(Action.FORWARD, f"Sweeping toward waypoint {label} ({distance:.0f}px away).", 1.0, "")

        turn = Action.TURN_LEFT if Action.TURN_LEFT in allowed else Action.TURN_RIGHT
        return Decision(turn if turn in allowed else Action.STOP, f"Boundary recovery while targeting {label}.", 1.0, "")


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
        "The rover is the object marked by the overlay. Numbered arrows are the ONLY legal choices. "
        "Visit unvisited grid cells; shaded cells are already visited. Prefer open space. "
        "If uncertain, choose a turn rather than a translation. Never invent a choice."
    )

    def __init__(
        self,
        transform: BodyToImage,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5vl:3b",
        timeout: float = 20.0,
        annotation_style: str = "arrows",
        requester: Callable[..., Any] | None = None,
    ):
        self.transform = transform
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.annotation_style = annotation_style
        self.requester = requester or requests.post
        self.fallback_count = 0

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

    def choose(self, frame, pose: RoverPose | None, allowed: list[Action], coverage: CoverageTracker) -> Decision:
        if pose is None or not allowed:
            return self._fallback(allowed or [Action.STOP], "No reliable pose; conservative fallback.", "", 0.0)
        if self.annotation_style == "grid":
            annotated, cells = draw_grid(frame, pose, coverage)
            properties = {"cell": {"type": "string"}, "reason": {"type": "string"}, "confidence": {"type": "number"}}
            required = ["cell", "reason", "confidence"]
            instruction = "Name one labelled target grid cell. Return JSON only."
        else:
            annotated, mapping = draw_arrows(frame, pose, self.transform, allowed, coverage)
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
            instruction = f"Choose exactly one numbered arrow. Legal choices: {legend}. Return JSON only."
        payload = {
            "model": self.model,
            "stream": False,
            "format": {"type": "object", "properties": properties, "required": required},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": instruction, "images": [self._jpeg_512(annotated)]},
            ],
        }
        started = time.monotonic()
        raw = ""
        try:
            response = self.requester(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            outer = response.json()
            raw_value = outer.get("message", {}).get("content", outer.get("response", ""))
            raw = raw_value if isinstance(raw_value, str) else json.dumps(raw_value)
            answer = raw_value if isinstance(raw_value, dict) else json.loads(raw_value)
            if self.annotation_style == "grid":
                cell = str(answer["cell"]).upper()
                if cell not in cells:
                    raise ValueError("out-of-range grid cell")
                col, row = cells[cell]
                h, w = frame.shape[:2]
                target = ((col+.5)*w/coverage.cols, (row+.5)*h/coverage.rows)
                action = min(allowed, key=lambda a: math.dist(self.transform.predict(pose, a), target))
            else:
                choice = int(answer["choice"])
                if choice not in mapping:
                    raise ValueError("out-of-range choice")
                action = mapping[choice]
            latency = time.monotonic() - started
            return Decision(action, str(answer["reason"]), max(0.0, min(1.0, float(answer["confidence"]))), raw, latency)
        except (requests.RequestException, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.fallback_count += 1
            return self._fallback(allowed, f"VLM failure ({type(exc).__name__}); conservative fallback.", raw, time.monotonic()-started)
