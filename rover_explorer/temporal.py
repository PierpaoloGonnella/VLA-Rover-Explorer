from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

from .localize import RoverPose
from .motion import Action


@dataclass(frozen=True, slots=True)
class TemporalStatus:
    stall_detected: bool
    repeated_maneuver: bool
    displacement_px: float
    action_sequence: tuple[str, ...]
    elapsed_seconds: float


class TemporalMemory:
    """Small bounded memory of observed outcomes for the asynchronous VLM."""

    def __init__(
        self,
        window_seconds: float = 20.0,
        stall_translation_attempts: int = 2,
        stall_displacement_px: float = 12.0,
        repeated_turn_attempts: int = 3,
        max_scenes: int = 5,
    ) -> None:
        self.window_seconds = max(2.0, float(window_seconds))
        self.stall_translation_attempts = max(1, int(stall_translation_attempts))
        self.stall_displacement_px = max(1.0, float(stall_displacement_px))
        self.repeated_turn_attempts = max(2, int(repeated_turn_attempts))
        self._poses: deque[tuple[float, RoverPose]] = deque(maxlen=200)
        self._actions: deque[tuple[float, Action]] = deque(maxlen=80)
        self._scenes: deque[str] = deque(maxlen=max(1, int(max_scenes)))
        self.objective = "Explore unvisited reachable floor safely."
        self.plan_steps: tuple[str, ...] = (
            "Observe current hazards.",
            "Select a reachable unvisited cell.",
            "Move while checking progress.",
        )
        self.target = "none"
        self.last_outcome = "No previous VLM plan has completed yet."

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._poses and self._poses[0][0] < cutoff:
            self._poses.popleft()
        while self._actions and self._actions[0][0] < cutoff:
            self._actions.popleft()

    def add_pose(self, pose: RoverPose, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        self._poses.append((timestamp, pose))
        self._trim(timestamp)

    def add_action(self, action: Action, now: float | None = None) -> None:
        if action == Action.STOP:
            return
        timestamp = time.monotonic() if now is None else float(now)
        self._actions.append((timestamp, action))
        self._trim(timestamp)

    def add_scene(self, description: str) -> None:
        normalized = " ".join(str(description).split())
        if normalized and (not self._scenes or normalized != self._scenes[-1]):
            self._scenes.append(normalized[:600])

    def record_outcome(self, outcome: str) -> None:
        normalized = " ".join(str(outcome).split())
        if normalized:
            self.last_outcome = normalized[:500]

    def update_plan(
        self,
        objective: str,
        plan_steps: tuple[str, ...],
        target: str,
        outcome: str,
    ) -> None:
        if objective.strip():
            self.objective = objective.strip()[:300]
        if plan_steps:
            self.plan_steps = tuple(step.strip()[:240] for step in plan_steps if step.strip())[:5]
        self.target = target or "none"
        self.last_outcome = outcome.strip()[:500] if outcome.strip() else self.last_outcome

    def status(self, now: float | None = None) -> TemporalStatus:
        timestamp = time.monotonic() if now is None else float(now)
        self._trim(timestamp)
        sequence = tuple(action.value for _, action in self._actions)
        translations = sum(
            action in {Action.FORWARD, Action.BACKWARD, Action.ARC_LEFT, Action.ARC_RIGHT}
            for _, action in self._actions
        )
        turns = [
            action for _, action in self._actions
            if action in {Action.TURN_LEFT, Action.TURN_RIGHT}
        ]
        displacement = 0.0
        elapsed = 0.0
        if len(self._poses) >= 2:
            elapsed = self._poses[-1][0] - self._poses[0][0]
            displacement = math.dist(self._poses[0][1].centre, self._poses[-1][1].centre)
        stall = (
            translations >= self.stall_translation_attempts
            and len(self._poses) >= 2
            and displacement < self.stall_displacement_px
        )
        repeated = len(turns) >= self.repeated_turn_attempts or (
            len(sequence) >= 4 and len(set(sequence[-4:])) == 1
        )
        return TemporalStatus(stall, repeated, displacement, sequence[-10:], elapsed)

    def context(
        self,
        *,
        legal_actions: list[Action],
        sonar_context: str,
        now: float | None = None,
    ) -> tuple[str, TemporalStatus]:
        status = self.status(now)
        scenes = " | ".join(
            f"t-{len(self._scenes) - index}: {scene}"
            for index, scene in enumerate(self._scenes)
        ) or "none"
        actions = ", ".join(status.action_sequence) or "none"
        steps = " -> ".join(self.plan_steps) or "none"
        text = (
            f"Persistent objective: {self.objective}\n"
            f"Previous plan: {steps}\n"
            f"Previous target: {self.target}\n"
            f"Previous outcome: {self.last_outcome}\n"
            f"Recent scene summaries (oldest to newest): {scenes}\n"
            f"Executed motor-pulse sequence: {actions}\n"
            f"Observed displacement over {status.elapsed_seconds:.1f}s: "
            f"{status.displacement_px:.1f}px\n"
            f"stall_detected={status.stall_detected}; "
            f"repeated_maneuver={status.repeated_maneuver}\n"
            f"Legal actions now: {', '.join(action.value for action in legal_actions)}\n"
            f"{sonar_context}"
        )
        return text, status
