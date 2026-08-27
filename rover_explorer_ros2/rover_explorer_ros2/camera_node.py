from __future__ import annotations

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from rover_explorer.camera import WebcamSource

from .common import ImageBridge


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 10.0)
        self.declare_parameter("frame_id", "rover_camera")
        self._source = WebcamSource(
            int(self.get_parameter("camera_index").value),
            int(self.get_parameter("camera_width").value),
            int(self.get_parameter("camera_height").value),
        )
        self._bridge = ImageBridge()
        self._publisher = self.create_publisher(Image, "/rover/image_raw", 10)
        fps = max(0.5, float(self.get_parameter("camera_fps").value))
        self.create_timer(1.0 / fps, self._publish)

    def _publish(self) -> None:
        try:
            frame = self._source.read()
            message = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = str(self.get_parameter("frame_id").value)
            self._publisher.publish(message)
        except (RuntimeError, cv2.error) as exc:
            self.get_logger().error(f"Camera read failed: {exc}")

    def destroy_node(self):
        self._source.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
