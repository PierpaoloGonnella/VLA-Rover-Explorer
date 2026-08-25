from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class CameraSource(ABC):
    timestamp: float

    @abstractmethod
    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class WebcamSource(CameraSource):
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.timestamp = 0.0
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}")
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, name="webcam-latest-frame", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=3.0):
            self.close()
            raise RuntimeError(f"Webcam index {index} opened but produced no frames")

    def _capture_loop(self) -> None:
        """Continuously consume the backend queue so callers never receive stale frames."""
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            captured_at = time.time()
            with self._lock:
                self._latest = frame
                self.timestamp = captured_at
            self._ready.set()

    def read(self) -> np.ndarray:
        if not self._ready.wait(timeout=1.0):
            raise RuntimeError("Webcam frame capture timed out")
        with self._lock:
            if self._latest is None:
                raise RuntimeError("Webcam frame capture failed")
            return self._latest.copy()

    def close(self) -> None:
        self._stop.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        self.capture.release()


class SimulatedSource(CameraSource):
    def __init__(self, simulator):
        self.simulator = simulator
        self.timestamp = 0.0

    def read(self) -> np.ndarray:
        self.timestamp = time.time()
        return self.simulator.render()


class ReplaySource(CameraSource):
    def __init__(self, session_dir: str | Path):
        root = Path(session_dir)
        self.frames = sorted((root / "raw").glob("*.jpg"))
        if not self.frames:
            self.frames = sorted(root.glob("raw_*.jpg"))
        if not self.frames:
            raise FileNotFoundError(f"No replay frames in {root}")
        self.index = 0
        self.timestamp = 0.0

    def read(self) -> np.ndarray:
        if self.index >= len(self.frames):
            raise EOFError("Replay session exhausted")
        path = self.frames[self.index]
        self.index += 1
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read replay frame {path}")
        self.timestamp = path.stat().st_mtime
        return frame
