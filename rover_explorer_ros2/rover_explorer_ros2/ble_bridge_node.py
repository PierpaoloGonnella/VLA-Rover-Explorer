from __future__ import annotations

import asyncio
import math
import queue
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState, Range
from std_msgs.msg import Bool, UInt32

from rover_explorer.ble import RoverBle

from .common import STOP_COMMAND


class BleWorker:
    """Owns the only BLE transport instance and serializes every write."""

    def __init__(self, rover: RoverBle, on_error):
        self.rover = rover
        self.on_error = on_error
        self.commands: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._thread_main, daemon=True, name="rover-ble")
        self.thread.start()

    def submit(self, command: str) -> None:
        while True:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                break
        self.commands.put_nowait(command)

    def close(self) -> None:
        while True:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                break
        self.commands.put_nowait(STOP_COMMAND)
        self.commands.put_nowait(None)
        self.thread.join(timeout=5.0)

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        try:
            await self.rover.connect()
            while True:
                command = await asyncio.to_thread(self.commands.get)
                if command is None:
                    break
                await self.rover.send(command)
        except Exception as exc:  # the ROS timer continues commanding STOP
            self.on_error(exc)
        finally:
            try:
                await self.rover.disconnect()
            except Exception as exc:
                self.on_error(exc)


class BleBridgeNode(Node):
    """Sole BLE writer, with an independent fail-closed command watchdog."""

    def __init__(self) -> None:
        super().__init__("ble_bridge_node")
        self.declare_parameter("device_name", "BT05")
        self.declare_parameter("characteristic_uuid", "0000ffe1-0000-1000-8000-00805f9b34fb")
        self.declare_parameter("reconnect_attempts", 4)
        self.declare_parameter("backoff_seconds", 0.5)
        self.declare_parameter("watchdog_seconds", 2.0)
        self.declare_parameter("sonar_publish_hz", 10.0)
        self.declare_parameter("sonar_stop_distance_m", 0.25)
        self.declare_parameter("speed", 170)

        self._last_command = time.monotonic()
        self._emergency_stop = False
        self._transport_error = False
        self._last_sent = STOP_COMMAND
        self._rover = RoverBle(
            str(self.get_parameter("device_name").value),
            str(self.get_parameter("characteristic_uuid").value),
            int(self.get_parameter("reconnect_attempts").value),
            float(self.get_parameter("backoff_seconds").value),
        )
        self._worker = BleWorker(self._rover, self._on_transport_error)

        command_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, command_qos)
        self.create_subscription(Bool, "/rover/emergency_stop", self._on_emergency_stop, command_qos)
        self._range_publishers = {
            "front": self.create_publisher(Range, "/rover/sonar/front", 10),
            "left": self.create_publisher(Range, "/rover/sonar/left", 10),
            "right": self.create_publisher(Range, "/rover/sonar/right", 10),
        }
        self._sonar_publisher = self.create_publisher(Range, "/rover/sonar", 10)
        self._battery_publisher = self.create_publisher(BatteryState, "/rover/battery", 10)
        self._scan_sequence_publisher = self.create_publisher(
            UInt32, "/rover/sonar/scan_sequence", 10
        )
        hz = max(1.0, float(self.get_parameter("sonar_publish_hz").value))
        self.create_timer(1.0 / hz, self._publish_telemetry)
        self.create_timer(0.05, self._watchdog)

    def _on_transport_error(self, exc: Exception) -> None:
        self._transport_error = True
        self.get_logger().error(f"BLE transport failed: {exc}")

    def _on_emergency_stop(self, message: Bool) -> None:
        self._emergency_stop = bool(message.data)
        if self._emergency_stop:
            self._send_stop("emergency stop")

    def _on_cmd_vel(self, message: Twist) -> None:
        self._last_command = time.monotonic()
        front_m = None if self._rover.sonar_cm is None else self._rover.sonar_cm / 100.0
        approaching = message.linear.x > 0.0
        sonar_veto = self._rover.obstacle_blocked or front_m is None or (
            front_m <= float(self.get_parameter("sonar_stop_distance_m").value)
        )
        if self._emergency_stop or self._transport_error or (approaching and sonar_veto):
            self._send_stop("final BLE safety veto")
            return
        scale = max(0, min(255, int(self.get_parameter("speed").value)))
        left = max(-1.0, min(1.0, message.linear.x - message.angular.z))
        right = max(-1.0, min(1.0, message.linear.x + message.angular.z))
        command = f"A#{round(left * scale)}#{round(right * scale)}#"
        self._last_sent = command
        self._worker.submit(command)

    def _send_stop(self, reason: str) -> None:
        if self._last_sent != STOP_COMMAND:
            self.get_logger().warning(f"Motors stopped: {reason}")
        self._last_sent = STOP_COMMAND
        self._worker.submit(STOP_COMMAND)

    def _watchdog(self) -> None:
        if time.monotonic() - self._last_command > float(self.get_parameter("watchdog_seconds").value):
            self._send_stop("command watchdog expired")

    def _publish_range(self, direction: str, centimetres: int | None, yaw: float) -> None:
        message = Range()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = f"sonar_{direction}"
        message.radiation_type = Range.ULTRASOUND
        message.field_of_view = math.radians(30.0)
        message.min_range = 0.02
        message.max_range = 3.0
        message.range = math.inf if centimetres is None else centimetres / 100.0
        self._range_publishers[direction].publish(message)
        self._sonar_publisher.publish(message)

    def _publish_telemetry(self) -> None:
        self._publish_range("front", self._rover.sonar_cm, 0.0)
        self._publish_range("left", self._rover.sonar_left_cm, math.pi / 4)
        self._publish_range("right", self._rover.sonar_right_cm, -math.pi / 4)
        battery = BatteryState()
        battery.header.stamp = self.get_clock().now().to_msg()
        battery.voltage = math.nan if self._rover.battery_mv is None else self._rover.battery_mv / 1000.0
        battery.present = self._rover.connected
        self._battery_publisher.publish(battery)
        scan = UInt32()
        scan.data = self._rover.sonar_scan_sequence
        self._scan_sequence_publisher.publish(scan)

    def destroy_node(self):
        self._worker.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BleBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
