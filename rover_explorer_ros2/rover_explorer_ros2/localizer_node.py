from __future__ import annotations

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from rover_explorer.localize import ArucoLocalizer, ColorBlobLocalizer
from rover_explorer_ros2.msg import RoverPose

from .common import ImageBridge


class LocalizerNode(Node):
    def __init__(self) -> None:
        super().__init__("localizer_node")
        self.declare_parameter("aruco_marker_id", 0)
        self.declare_parameter("aruco_heading_offset_degrees", 0.0)
        self.declare_parameter("localization_backend", "aruco_custom")
        self.declare_parameter("min_confidence", 0.25)
        self.declare_parameter("aruco_compare_topic", "/aruco/markers")
        self.declare_parameter("camera_fx", 0.0)
        self.declare_parameter("camera_fy", 0.0)
        self.declare_parameter("camera_cx", 0.0)
        self.declare_parameter("camera_cy", 0.0)
        backend = str(self.get_parameter("localization_backend").value)
        self._localizer = ColorBlobLocalizer() if backend == "color" else ArucoLocalizer(
            int(self.get_parameter("aruco_marker_id").value),
            math.radians(float(self.get_parameter("aruco_heading_offset_degrees").value)),
        )
        self._bridge = ImageBridge()
        self._publisher = self.create_publisher(RoverPose, "/rover/pose", 10)
        self._comparison_publisher = self.create_publisher(Float32, "/rover/localization/aruco_error_px", 10)
        self._last_custom = None
        self.create_subscription(Image, "/rover/image_raw", self._on_image, 10)
        self._configure_ros2_aruco_comparison()

    def _configure_ros2_aruco_comparison(self) -> None:
        try:
            from ros2_aruco_interfaces.msg import ArucoMarkers
        except ImportError:
            self.get_logger().info("ros2_aruco interfaces unavailable; custom ArUco remains active")
            return
        self.create_subscription(
            ArucoMarkers,
            str(self.get_parameter("aruco_compare_topic").value),
            self._on_ros2_aruco,
            10,
        )

    def _on_image(self, message: Image) -> None:
        pose = self._localizer.locate(self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"))
        if pose is None or pose.confidence < float(self.get_parameter("min_confidence").value):
            return
        result = RoverPose()
        result.header = message.header
        result.centre.x, result.centre.y = pose.centre
        result.centre.z = 0.0
        result.has_heading = pose.heading is not None
        result.heading = 0.0 if pose.heading is None else pose.heading
        result.confidence = pose.confidence
        self._last_custom = pose
        self._publisher.publish(result)

    def _on_ros2_aruco(self, message) -> None:
        """Publishes comparison error; never replaces the validated custom pose."""
        if self._last_custom is None or not getattr(message, "poses", None):
            return
        ids = list(getattr(message, "marker_ids", []))
        marker_id = int(self.get_parameter("aruco_marker_id").value)
        try:
            index = ids.index(marker_id)
        except ValueError:
            return
        external = message.poses[index].position
        fx = float(self.get_parameter("camera_fx").value)
        fy = float(self.get_parameter("camera_fy").value)
        if fx <= 0.0 or fy <= 0.0 or abs(external.z) < 1e-9:
            return
        projected = (
            fx * external.x / external.z + float(self.get_parameter("camera_cx").value),
            fy * external.y / external.z + float(self.get_parameter("camera_cy").value),
        )
        error = math.dist(self._last_custom.centre, projected)
        output = Float32()
        output.data = float(error)
        self._comparison_publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
