from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image, Range

from rover_explorer_ros2.msg import LegalActions, PolicyDecision, RoverPose, VlmAdvisory

from .common import ImageBridge


class LoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("logger_node")
        self.declare_parameter("log_directory", "sessions/ros2")
        root = Path(str(self.get_parameter("log_directory").value)) / time.strftime("%Y%m%d-%H%M%S")
        self._images = root / "raw"
        self._vlm_images = root / "vlm"
        self._images.mkdir(parents=True, exist_ok=True)
        self._vlm_images.mkdir(parents=True, exist_ok=True)
        self._events = root / "events.jsonl"
        self._bridge = ImageBridge()
        self._lock = threading.Lock()
        self._image_index = 0
        self._vlm_image_index = 0
        subscriptions = (
            (RoverPose, "/rover/pose"),
            (LegalActions, "/rover/legal_actions"),
            (VlmAdvisory, "/rover/vlm/advisory"),
            (PolicyDecision, "/rover/policy/classic_decision"),
            (BatteryState, "/rover/battery"),
            (Range, "/rover/sonar"),
            (OccupancyGrid, "/rover/occupancy_grid"),
            (OccupancyGrid, "/rover/coverage_map"),
            (Twist, "/cmd_vel"),
        )
        for message_type, topic in subscriptions:
            self.create_subscription(message_type, topic, lambda msg, name=topic: self._record(name, msg), 10)
        self.create_subscription(Image, "/rover/image_raw", self._on_image, 10)
        self.create_subscription(Image, "/rover/vlm/debug_image", self._on_vlm_image, 2)

    @staticmethod
    def _simple(message):
        if isinstance(message, dict):
            return message
        from rosidl_runtime_py.convert import message_to_ordereddict

        return message_to_ordereddict(message)

    def _record(self, topic: str, message) -> None:
        record = {"timestamp": time.time(), "topic": topic, "message": self._simple(message)}
        with self._lock, self._events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=True) + "\n")

    def _on_image(self, message: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        path = self._images / f"{self._image_index:06d}.jpg"
        self._image_index += 1
        cv2.imwrite(str(path), frame)
        self._record("/rover/image_raw", {"path": str(path)})

    def _on_vlm_image(self, message: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        path = self._vlm_images / f"{self._vlm_image_index:06d}.jpg"
        self._vlm_image_index += 1
        cv2.imwrite(str(path), frame)
        self._record("/rover/vlm/debug_image", {"path": str(path)})


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
