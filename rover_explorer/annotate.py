from __future__ import annotations

import math

import cv2
import numpy as np

from .calibrate import BodyToImage
from .coverage import CoverageTracker
from .guard import safe_rectangle
from .localize import RoverPose
from .motion import Action


def _outlined_text(image, text: str, origin: tuple[int, int], scale: float = 0.65) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_coverage(frame: np.ndarray, coverage: CoverageTracker | None) -> None:
    if coverage is None:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for col, row in coverage.visited:
        p1 = (round(col * w / coverage.cols), round(row * h / coverage.rows))
        p2 = (round((col + 1) * w / coverage.cols), round((row + 1) * h / coverage.rows))
        cv2.rectangle(overlay, p1, p2, (90, 90, 90), -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    for col in range(1, coverage.cols):
        cv2.line(frame, (round(col*w/coverage.cols), 0), (round(col*w/coverage.cols), h), (120, 120, 120), 1)
    for row in range(1, coverage.rows):
        cv2.line(frame, (0, round(row*h/coverage.rows)), (w, round(row*h/coverage.rows)), (120, 120, 120), 1)


def draw_arrows(
    frame: np.ndarray,
    pose: RoverPose,
    transform: BodyToImage,
    actions: list[Action],
    coverage: CoverageTracker | None = None,
    margin_frac: float = 0.12,
) -> tuple[np.ndarray, dict[int, Action]]:
    result = frame.copy()
    _draw_coverage(result, coverage)
    left, top, right, bottom = safe_rectangle(frame.shape, margin_frac)
    cv2.rectangle(result, (round(left), round(top)), (round(right), round(bottom)), (0, 180, 255), 2)
    mapping: dict[int, Action] = {}
    start = tuple(round(v) for v in pose.centre)
    for index, action in enumerate(actions, 1):
        mapping[index] = action
        destination = transform.predict(pose, action)
        end = tuple(round(v) for v in destination)
        if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            sign = -1 if action == Action.TURN_LEFT else 1
            end = (start[0] + sign * 24, start[1] - 24)
        elif action == Action.STOP:
            end = start
        color = (40, 40, 240) if action == Action.STOP else (20, 220, 20)
        cv2.arrowedLine(result, start, end, (0, 0, 0), 5, tipLength=0.25)
        cv2.arrowedLine(result, start, end, color, 2, tipLength=0.25)
        _outlined_text(result, str(index), (end[0] + 4, end[1] - 4))
    cv2.circle(result, start, 6, (255, 0, 255), -1)
    return result, mapping


def draw_grid(
    frame: np.ndarray,
    pose: RoverPose | None,
    coverage: CoverageTracker,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    result = frame.copy()
    _draw_coverage(result, coverage)
    h, w = result.shape[:2]
    mapping: dict[str, tuple[int, int]] = {}
    for row in range(coverage.rows):
        for col in range(coverage.cols):
            label = f"{chr(65 + col)}{row + 1}"
            mapping[label] = (col, row)
            _outlined_text(result, label, (round(col*w/coverage.cols)+5, round(row*h/coverage.rows)+22), .55)
    if pose is not None:
        cv2.circle(result, tuple(round(v) for v in pose.centre), 8, (255, 0, 255), 2)
    return result, mapping


def highlight_action(frame: np.ndarray, pose: RoverPose, transform: BodyToImage, action: Action) -> np.ndarray:
    result = frame.copy()
    start = tuple(round(v) for v in pose.centre)
    end = tuple(round(v) for v in transform.predict(pose, action))
    cv2.arrowedLine(result, start, end, (0, 255, 255), 5, tipLength=.25)
    return result

