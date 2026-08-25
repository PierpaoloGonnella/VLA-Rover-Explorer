from __future__ import annotations

import asyncio
import math
import time

import cv2
import numpy as np


class RoverSimulator:
    """Differential-drive physics and a one-to-one synthetic overhead camera."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        wheel_slip: float = 0.03,
        ble_latency_ms: int = 10,
        px_per_second_at_full_speed: float = 180.0,
        radians_per_second_at_full_speed: float = 3.0,
        seed: int = 7,
        marker_id: int = 0,
    ):
        self.width, self.height = width, height
        self.x, self.y = width / 2, height / 2
        self.heading = 0.0
        self.left_speed = self.right_speed = 0
        self.wheel_slip = wheel_slip
        self.ble_latency_ms = ble_latency_ms
        self.linear_scale = px_per_second_at_full_speed
        self.angular_scale = radians_per_second_at_full_speed
        self.rng = np.random.default_rng(seed)
        self.marker_id = marker_id
        self.battery_mv = 7400
        self._last_update = time.monotonic()

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last_update)
        self._last_update = now
        left = self.left_speed / 255.0
        right = self.right_speed / 255.0
        slip_l, slip_r = self.rng.normal(1.0, self.wheel_slip, 2)
        left *= slip_l
        right *= slip_r
        linear = (left + right) * 0.5 * self.linear_scale
        angular = (right - left) * 0.5 * self.angular_scale
        if abs(angular) > 1e-8:
            mid = self.heading + angular * dt / 2
            self.x += linear * dt * math.cos(mid)
            self.y += linear * dt * math.sin(mid)
            self.heading += angular * dt
        else:
            self.x += linear * dt * math.cos(self.heading)
            self.y += linear * dt * math.sin(self.heading)
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

    async def command(self, command: str) -> None:
        await asyncio.sleep(self.ble_latency_ms / 1000)
        self._integrate()
        parts = command.strip().split("#")
        if parts and parts[0] == "A" and len(parts) >= 3:
            self.left_speed = max(-255, min(255, int(parts[1])))
            self.right_speed = max(-255, min(255, int(parts[2])))
        elif parts and parts[0] == "C":
            pass
        # H# is intentionally unsupported: manual-mode vision only.

    def render(self) -> np.ndarray:
        self._integrate()
        frame = np.full((self.height, self.width, 3), (225, 230, 225), np.uint8)
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), (210, 215, 210), 1)
        for y in range(0, self.height, 80):
            cv2.line(frame, (0, y), (self.width, y), (210, 215, 210), 1)
        c, s = math.cos(self.heading), math.sin(self.heading)
        local = np.array([[-30, -22], [30, -22], [30, 22], [-30, 22]], np.float32)
        rotation = np.array([[c, -s], [s, c]], np.float32)
        body = local @ rotation.T + np.array([self.x, self.y])
        cv2.fillConvexPoly(frame, np.rint(body).astype(np.int32), (40, 115, 190))
        nose = np.array([self.x + 34 * c, self.y + 34 * s], dtype=np.int32)
        cv2.circle(frame, tuple(nose), 5, (0, 255, 0), -1)
        self._draw_marker(frame, size=34)
        return frame

    def _draw_marker(self, frame: np.ndarray, size: int) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(dictionary, self.marker_id, 100)
        else:  # pragma: no cover - older OpenCV
            marker = np.zeros((100, 100), np.uint8)
            cv2.aruco.drawMarker(dictionary, self.marker_id, 100, marker, 1)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        c, s = math.cos(self.heading), math.sin(self.heading)
        half = size / 2
        local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], np.float32)
        rotation = np.array([[c, -s], [s, c]], np.float32)
        destination = (local @ rotation.T + [self.x, self.y]).astype(np.float32)
        source = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32)
        matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(marker_bgr, matrix, (self.width, self.height), borderValue=(255, 255, 255))
        mask = cv2.warpPerspective(np.full((100, 100), 255, np.uint8), matrix, (self.width, self.height))
        frame[mask > 0] = warped[mask > 0]

