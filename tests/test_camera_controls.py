import time

import numpy as np
import pytest

from rover_explorer import camera


class FakeCapture:
    def __init__(self, index, backend):
        self.index = index
        self.backend = backend
        self.values = {}
        self.open = True

    def set(self, property_id, value):
        if property_id == camera.cv2.CAP_PROP_GAIN:
            return False
        self.values[property_id] = value
        return True

    def get(self, property_id):
        return self.values.get(property_id, -1.0)

    def getBackendName(self):
        return "FAKE"

    def isOpened(self):
        return self.open

    def read(self):
        time.sleep(0.001)
        return True, np.zeros((2, 3, 3), np.uint8)

    def release(self):
        self.open = False


def test_unset_controls_preserve_defaults_and_effective_values_are_reported(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", FakeCapture)
    source = camera.WebcamSource(2, 640, 480, fps=10.0, backend="auto", controls={})
    try:
        assert source.backend_name == "FAKE"
        assert set(source.control_results) == {"width", "height", "fps", "buffer_size"}
        assert source.control_results["fps"].effective == 10.0
    finally:
        source.close()


def test_unsupported_camera_property_is_reported_without_crashing(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", FakeCapture)
    source = camera.WebcamSource(0, controls={"gain": 12.0, "exposure": -6.0})
    try:
        assert not source.control_results["gain"].accepted
        assert source.control_results["exposure"].accepted
        assert source.control_results["exposure"].effective == -6.0
    finally:
        source.close()


def test_invalid_backend_and_control_are_rejected_clearly():
    with pytest.raises(ValueError, match="Unsupported camera backend"):
        camera.camera_backend("made-up")
