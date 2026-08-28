from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import cv2
import rclpy
from geometry_msgs.msg import Twist
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image, Range

from rover_explorer_ros2.msg import LegalActions, PolicyDecision, RoverPose, VlmAdvisory

from .common import ImageBridge


class LoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("logger_node")
        self.declare_parameter("log_directory", "sessions/ros2")
        self.declare_parameter("failure_image_queue_size", 4)
        self.declare_parameter("failure_image_jpeg_quality", 85)
        root = Path(str(self.get_parameter("log_directory").value)) / time.strftime("%Y%m%d-%H%M%S")
        self._root = root
        self._images = root / "raw"
        self._vlm_images = root / "vlm"
        self._failure_images = root / "localization_failures"
        self._images.mkdir(parents=True, exist_ok=True)
        self._vlm_images.mkdir(parents=True, exist_ok=True)
        self._failure_images.mkdir(parents=True, exist_ok=True)
        self._events = root / "events.jsonl"
        self._bridge = ImageBridge()
        self._lock = threading.Lock()
        self._image_index = 0
        self._vlm_image_index = 0
        self._failure_image_index = 0
        self._failure_queue: queue.Queue[tuple[Path, object] | None] = queue.Queue(
            maxsize=max(1, int(self.get_parameter("failure_image_queue_size").value))
        )
        self._failure_worker = threading.Thread(
            target=self._failure_writer, name="localization-failure-writer", daemon=True
        )
        self._failure_worker.start()
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
            (DiagnosticArray, "/rover/localization/diagnostics"),
            (DiagnosticArray, "/rover/camera/diagnostics"),
            (DiagnosticArray, "/rover/battery/diagnostics"),
        )
        for message_type, topic in subscriptions:
            self.create_subscription(message_type, topic, lambda msg, name=topic: self._record(name, msg), 10)
        self.create_subscription(Image, "/rover/image_raw", self._on_image, 10)
        self.create_subscription(
            Image, "/rover/localization/failure_image", self._on_failure_image, 2
        )
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
        self._record(
            "/rover/image_raw",
            {"path": str(path), "header": self._simple(message.header)},
        )

    def _on_failure_image(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            reason = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in (message.header.frame_id or "failure")
            )
            path = self._failure_images / f"{self._failure_image_index:04d}_{reason}.jpg"
            self._failure_image_index += 1
            self._failure_queue.put_nowait((path, frame))
        except (ValueError, cv2.error, queue.Full) as exc:
            self.get_logger().warning(f"Dropped localization failure image: {exc}")

    def _failure_writer(self) -> None:
        quality = max(1, min(100, int(self.get_parameter("failure_image_jpeg_quality").value)))
        while True:
            item = self._failure_queue.get()
            try:
                if item is None:
                    return
                path, frame = item
                ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok:
                    self._record(
                        "/rover/localization/failure_image",
                        {"path": str(path), "jpeg_quality": quality},
                    )
            finally:
                self._failure_queue.task_done()

    def _on_vlm_image(self, message: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        path = self._vlm_images / f"{self._vlm_image_index:06d}.jpg"
        self._vlm_image_index += 1
        cv2.imwrite(str(path), frame)
        self._record("/rover/vlm/debug_image", {"path": str(path)})

    def destroy_node(self):
        try:
            self._failure_queue.put_nowait(None)
        except queue.Full:
            try:
                self._failure_queue.get_nowait()
                self._failure_queue.task_done()
            except queue.Empty:
                pass
            self._failure_queue.put_nowait(None)
        self._failure_worker.join(timeout=3.0)
        return super().destroy_node()


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
