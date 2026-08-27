from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image

from rover_explorer.calibrate import BodyToImage
from rover_explorer.motion import Action


ALL_ACTIONS = {action.value: action for action in Action}
STOP_COMMAND = "A#0#0#"


try:
    from cv_bridge import CvBridge as ImageBridge
except ImportError:
    class ImageBridge:
        """Small bgr8 bridge for binary ROS distributions without cv_bridge."""

        def cv2_to_imgmsg(self, frame: np.ndarray, encoding: str = "bgr8") -> Image:
            if encoding != "bgr8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("The fallback image bridge supports bgr8 frames only")
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            message = Image()
            message.height, message.width = frame.shape[:2]
            message.encoding = encoding
            message.is_bigendian = False
            message.step = message.width * 3
            message.data = frame.tobytes()
            return message

        def imgmsg_to_cv2(self, message: Image, desired_encoding: str = "bgr8") -> np.ndarray:
            if desired_encoding != "bgr8" or message.encoding not in ("bgr8", "8UC3"):
                raise ValueError("The fallback image bridge supports bgr8 frames only")
            row_bytes = int(message.step) if message.step else int(message.width) * 3
            raw = np.frombuffer(message.data, dtype=np.uint8)
            rows = raw.reshape(int(message.height), row_bytes)
            return rows[:, : int(message.width) * 3].reshape(
                int(message.height), int(message.width), 3
            ).copy()


def action_from_string(value: str) -> Action:
    return ALL_ACTIONS.get(value.strip().lower(), Action.STOP)


def transform_from_node(node) -> BodyToImage:
    return BodyToImage(
        float(node.get_parameter("px_per_forward_pulse").value),
        float(node.get_parameter("radians_per_turn_pulse").value),
        (
            float(node.get_parameter("forward_axis_x").value),
            float(node.get_parameter("forward_axis_y").value),
        ),
    )


def declare_transform_parameters(node) -> None:
    node.declare_parameter("px_per_forward_pulse", 35.0)
    node.declare_parameter("radians_per_turn_pulse", 0.35)
    node.declare_parameter("forward_axis_x", 1.0)
    node.declare_parameter("forward_axis_y", 0.0)


def seconds_since(stamp) -> float:
    if stamp is None:
        return math.inf
    return time.monotonic() - stamp


def quaternion_from_yaw(yaw: float) -> Quaternion:
    result = Quaternion()
    result.z = math.sin(yaw / 2.0)
    result.w = math.cos(yaw / 2.0)
    return result


def package_share_file(package: str, relative: str) -> str:
    from ament_index_python.packages import get_package_share_directory

    return str(Path(get_package_share_directory(package)) / relative)
