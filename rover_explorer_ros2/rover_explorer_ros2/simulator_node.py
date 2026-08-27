from __future__ import annotations

import asyncio
import math
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image, Range

from rover_explorer.simulator import RoverSimulator

from .common import ImageBridge


class SimulatorNode(Node):
    """ROS wrapper around the existing simulator; no BLE transport is created."""

    def __init__(self) -> None:
        super().__init__("simulator_node")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 10.0)
        self.declare_parameter("frame_id", "rover_camera")
        self.declare_parameter("speed", 150)
        self.declare_parameter("watchdog_seconds", 2.0)
        self._simulator = RoverSimulator(
            int(self.get_parameter("camera_width").value),
            int(self.get_parameter("camera_height").value),
            ble_latency_ms=0,
        )
        self._bridge = ImageBridge()
        self._last_command = time.monotonic()
        self._lock = threading.Lock()
        self._image_publisher = self.create_publisher(Image, "/rover/image_raw", 10)
        self._battery_publisher = self.create_publisher(BatteryState, "/rover/battery", 10)
        self._sonar_publishers = {
            name: self.create_publisher(Range, f"/rover/sonar/{name}", 10)
            for name in ("front", "left", "right")
        }
        self._sonar_publisher = self.create_publisher(Range, "/rover/sonar", 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        fps = max(0.5, float(self.get_parameter("camera_fps").value))
        self.create_timer(1.0 / fps, self._publish)
        self.create_timer(0.05, self._watchdog)

    def _apply(self, command: str) -> None:
        with self._lock:
            asyncio.run(self._simulator.command(command))

    def _on_cmd_vel(self, message: Twist) -> None:
        self._last_command = time.monotonic()
        scale = max(0, min(255, int(self.get_parameter("speed").value)))
        left = max(-1.0, min(1.0, message.linear.x - message.angular.z))
        right = max(-1.0, min(1.0, message.linear.x + message.angular.z))
        self._apply(f"A#{round(left * scale)}#{round(right * scale)}#")

    def _watchdog(self) -> None:
        if time.monotonic() - self._last_command > float(self.get_parameter("watchdog_seconds").value):
            self._apply("A#0#0#")

    def _publish(self) -> None:
        with self._lock:
            frame = self._simulator.render()
        now = self.get_clock().now().to_msg()
        image = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image.header.stamp = now
        image.header.frame_id = str(self.get_parameter("frame_id").value)
        self._image_publisher.publish(image)
        battery = BatteryState()
        battery.header.stamp = now
        battery.voltage = self._simulator.battery_mv / 1000.0
        battery.present = True
        self._battery_publisher.publish(battery)
        for name, publisher in self._sonar_publishers.items():
            reading = Range()
            reading.header.stamp = now
            reading.header.frame_id = f"sonar_{name}"
            reading.radiation_type = Range.ULTRASOUND
            reading.field_of_view = math.radians(30)
            reading.min_range = 0.02
            reading.max_range = 3.0
            reading.range = math.inf
            publisher.publish(reading)
            self._sonar_publisher.publish(reading)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
