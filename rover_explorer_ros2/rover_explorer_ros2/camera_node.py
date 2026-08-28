from __future__ import annotations

import cv2
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
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
        self.declare_parameter("camera_backend", "auto")
        self.declare_parameter("camera_buffer_size", 1)
        for name in (
            "autofocus",
            "focus",
            "auto_exposure",
            "exposure",
            "gain",
            "brightness",
            "contrast",
            "white_balance",
        ):
            self.declare_parameter(f"camera_{name}", "")
        self.declare_parameter("frame_id", "rover_camera")
        fps = max(0.5, float(self.get_parameter("camera_fps").value))
        controls = {}
        for name in (
            "autofocus",
            "focus",
            "auto_exposure",
            "exposure",
            "gain",
            "brightness",
            "contrast",
            "white_balance",
        ):
            raw = str(self.get_parameter(f"camera_{name}").value).strip()
            if raw:
                try:
                    controls[name] = float(raw)
                except ValueError as exc:
                    raise ValueError(f"camera_{name} must be empty or numeric") from exc
        self._source = WebcamSource(
            int(self.get_parameter("camera_index").value),
            int(self.get_parameter("camera_width").value),
            int(self.get_parameter("camera_height").value),
            fps=fps,
            backend=str(self.get_parameter("camera_backend").value),
            buffer_size=int(self.get_parameter("camera_buffer_size").value),
            controls=controls,
        )
        self._bridge = ImageBridge()
        self._publisher = self.create_publisher(Image, "/rover/image_raw", 10)
        diagnostic_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._diagnostics = self.create_publisher(
            DiagnosticArray, "/rover/camera/diagnostics", diagnostic_qos
        )
        self._publish_configuration()
        self.create_timer(5.0, self._publish_configuration)
        self.create_timer(1.0 / fps, self._publish)

    def _publish_configuration(self) -> None:
        status = DiagnosticStatus()
        status.name = "rover/camera/configuration"
        status.hardware_id = self._source.backend_name
        status.level = DiagnosticStatus.OK
        status.message = "effective camera configuration"
        status.values = [KeyValue(key="backend", value=self._source.backend_name)]
        for name, result in self._source.control_results.items():
            status.values.extend(
                [
                    KeyValue(key=f"{name}.requested", value=str(result.requested)),
                    KeyValue(key=f"{name}.effective", value=str(result.effective)),
                    KeyValue(key=f"{name}.accepted", value=str(result.accepted).lower()),
                ]
            )
            if not result.accepted:
                status.level = max(status.level, DiagnosticStatus.WARN)
                self.get_logger().warning(
                    f"Camera backend did not accept {name}={result.requested}; "
                    f"effective={result.effective}"
                )
            else:
                self.get_logger().info(
                    f"Camera {name}: requested={result.requested}, effective={result.effective}"
                )
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [status]
        self._diagnostics.publish(array)

    def _publish(self) -> None:
        try:
            frame = self._source.read()
            message = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            # Source time is the actual backend capture time, not timer publication time.
            message.header.stamp = Time(seconds=self._source.timestamp).to_msg()
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
