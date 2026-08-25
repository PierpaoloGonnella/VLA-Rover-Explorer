from __future__ import annotations

import logging

from .calibrate import BodyToImage
from .localize import RoverPose
from .motion import Action

LOGGER = logging.getLogger(__name__)
TRANSLATING_ACTIONS = (Action.FORWARD, Action.BACKWARD, Action.ARC_LEFT, Action.ARC_RIGHT)
TURN_ACTIONS = (Action.TURN_LEFT, Action.TURN_RIGHT)
ULTRASONIC_BLOCKED_ACTIONS = (Action.FORWARD, Action.ARC_LEFT, Action.ARC_RIGHT)


def safe_rectangle(frame_shape: tuple[int, ...], margin_frac: float) -> tuple[float, float, float, float]:
    height, width = frame_shape[:2]
    margin_x = width * margin_frac
    margin_y = height * margin_frac
    return margin_x, margin_y, width - margin_x, height - margin_y


def allowed_actions(
    pose: RoverPose | None,
    transform: BodyToImage,
    frame_shape: tuple[int, ...],
    margin_frac: float = 0.12,
) -> list[Action]:
    if pose is None:
        LOGGER.warning("LOST: rover not localized; entering conservative recovery")
        return [Action.BACKWARD, Action.STOP]
    left, top, right, bottom = safe_rectangle(frame_shape, margin_frac)
    # Timed asyncio pulses and wheel slip introduce small errors around the calibrated
    # mean.  Reserve a fraction of one pulse inside the nominal safety envelope so an
    # action that is mathematically tangent to the boundary cannot overshoot it.
    uncertainty = max(2.0, transform.px_per_forward_pulse)
    allowed: list[Action] = []
    for action in TRANSLATING_ACTIONS:
        x, y = transform.predict(pose, action)
        if left + uncertainty <= x <= right - uncertainty and top + uncertainty <= y <= bottom - uncertainty:
            allowed.append(action)
    allowed.extend(TURN_ACTIONS)
    allowed.append(Action.STOP)
    return allowed


def apply_ultrasonic_guard(actions: list[Action], blocked: bool) -> list[Action]:
    """Veto motion toward a front obstacle while preserving escape actions."""
    if not blocked:
        return actions
    return [action for action in actions if action not in ULTRASONIC_BLOCKED_ACTIONS]


def recovery_sequence() -> list[tuple[Action, int]]:
    """Two short reverse pulses; the zero-duration STOP entry represents the pause."""
    return [(Action.BACKWARD, 200), (Action.STOP, 250), (Action.BACKWARD, 200)]
