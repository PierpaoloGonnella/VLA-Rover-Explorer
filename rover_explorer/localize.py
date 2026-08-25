from __future__ import annotations

import base64
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable

import cv2
import numpy as np
import requests


@dataclass(slots=True)
class RoverPose:
    centre: tuple[float, float]
    heading: float | None
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Localizer(ABC):
    @abstractmethod
    def locate(self, frame: np.ndarray) -> RoverPose | None:
        raise NotImplementedError


class ArucoLocalizer(Localizer):
    def __init__(self, marker_id: int = 0):
        self.marker_id = marker_id
        self.last_detected_ids: list[int] = []
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, parameters)

    def locate(self, frame: np.ndarray) -> RoverPose | None:
        corners, ids, _ = self.detector.detectMarkers(frame)
        self.last_detected_ids = [] if ids is None else [int(value) for value in ids.ravel()]
        if ids is None:
            return None
        matches = np.flatnonzero(ids.ravel() == self.marker_id)
        if not len(matches):
            return None
        points = corners[int(matches[0])].reshape(4, 2)
        centre = points.mean(axis=0)
        edge = points[1] - points[0]
        heading = math.atan2(float(edge[1]), float(edge[0]))
        perimeter = float(cv2.arcLength(points.astype(np.float32), True))
        confidence = min(1.0, perimeter / max(32.0, 0.2 * min(frame.shape[:2])))
        return RoverPose((float(centre[0]), float(centre[1])), heading, confidence)


class ColorBlobLocalizer(Localizer):
    def __init__(
        self,
        hsv_low: tuple[int, int, int] = (35, 80, 80),
        hsv_high: tuple[int, int, int] = (90, 255, 255),
        min_area: float = 30.0,
    ):
        self.low = np.array(hsv_low, np.uint8)
        self.high = np.array(hsv_high, np.uint8)
        self.min_area = min_area

    def locate(self, frame: np.ndarray) -> RoverPose | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.low, self.high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if area < self.min_area or moments["m00"] == 0:
            return None
        centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        confidence = min(1.0, area / (frame.shape[0] * frame.shape[1] * 0.01))
        return RoverPose(centre, None, confidence)


def cell_to_centre(label: str, frame_shape: tuple[int, ...], cols: int = 6, rows: int = 4) -> tuple[float, float]:
    label = label.strip().upper()
    if len(label) < 2 or not label[0].isalpha() or not label[1:].isdigit():
        raise ValueError(f"Invalid grid cell {label!r}")
    col, row = ord(label[0]) - ord("A"), int(label[1:]) - 1
    if not (0 <= col < cols and 0 <= row < rows):
        raise ValueError(f"Grid cell {label!r} is outside the {cols}x{rows} grid")
    height, width = frame_shape[:2]
    return ((col + 0.5) * width / cols, (row + 0.5) * height / rows)


class VlmLocalizer(Localizer):
    """Deliberately coarse localizer used to quantify VLM spatial error."""

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5vl:3b",
        cols: int = 6,
        rows: int = 4,
        timeout: float = 20.0,
        requester: Callable[..., Any] | None = None,
    ):
        self.url, self.model = url.rstrip("/"), model
        self.cols, self.rows, self.timeout = cols, rows, timeout
        self.requester = requester or requests.post

    def _grid_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = result.shape[:2]
        for col in range(self.cols + 1):
            cv2.line(result, (round(col * w / self.cols), 0), (round(col * w / self.cols), h), (255, 255, 255), 2)
        for row in range(self.rows + 1):
            cv2.line(result, (0, round(row * h / self.rows)), (w, round(row * h / self.rows)), (255, 255, 255), 2)
        for row in range(self.rows):
            for col in range(self.cols):
                cv2.putText(result, f"{chr(65 + col)}{row + 1}", (int(col*w/self.cols)+5, int(row*h/self.rows)+22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
                cv2.putText(result, f"{chr(65 + col)}{row + 1}", (int(col*w/self.cols)+5, int(row*h/self.rows)+22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        return result

    def locate(self, frame: np.ndarray) -> RoverPose | None:
        annotated = self._grid_frame(frame)
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            return None
        payload = {
            "model": self.model,
            "stream": False,
            "format": {"type": "object", "properties": {"cell": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["cell", "confidence"]},
            "messages": [{"role": "user", "content": "Which labelled grid cell contains the rover? Return JSON only.", "images": [base64.b64encode(encoded).decode("ascii")]}],
        }
        try:
            response = self.requester(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            outer = response.json()
            content = outer.get("message", {}).get("content", outer.get("response", "{}"))
            answer = content if isinstance(content, dict) else json.loads(content)
            centre = cell_to_centre(answer["cell"], frame.shape, self.cols, self.rows)
            return RoverPose(centre, None, max(0.0, min(1.0, float(answer.get("confidence", 0.25)))))
        except (requests.RequestException, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None
