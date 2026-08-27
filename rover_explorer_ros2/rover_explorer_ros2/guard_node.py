from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Bool

from rover_explorer.guard import allowed_actions, apply_ultrasonic_guard
from rover_explorer.localize import RoverPose as CorePose

from rover_explorer_ros2.msg import LegalActions, RoverPose

from .common import declare_transform_parameters, transform_from_node


class GuardNode(Node):
    def __init__(self) -> None:
        super().__init__("guard_node")
        declare_transform_parameters(self)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("margin_frac", 0.12)
        self.declare_parameter("pose_timeout_seconds", 1.0)
        self.declare_parameter("sonar_timeout_seconds", 1.0)
        self.declare_parameter("sonar_stop_distance_m", 0.25)
        self._pose: RoverPose | None = None
        self._pose_received = 0.0
        self._front_range = math.inf
        self._sonar_received = 0.0
        self._emergency = False
        self._publisher = self.create_publisher(LegalActions, "/rover/legal_actions", 10)
        self.create_subscription(RoverPose, "/rover/pose", self._on_pose, 10)
        self.create_subscription(Range, "/rover/sonar", self._on_sonar, 10)
        self.create_subscription(Bool, "/rover/emergency_stop", self._on_emergency, 10)
        self.create_timer(0.05, self._recalculate)

    def _on_pose(self, message: RoverPose) -> None:
        self._pose = message
        self._pose_received = time.monotonic()

    def _on_sonar(self, message: Range) -> None:
        if message.header.frame_id and not message.header.frame_id.endswith("front"):
            return
        self._front_range = float(message.range)
        self._sonar_received = time.monotonic()

    def _on_emergency(self, message: Bool) -> None:
        self._emergency = bool(message.data)
        self._recalculate()

    def _recalculate(self) -> None:
        now = time.monotonic()
        pose_fresh = now - self._pose_received <= float(self.get_parameter("pose_timeout_seconds").value)
        sonar_fresh = now - self._sonar_received <= float(self.get_parameter("sonar_timeout_seconds").value)
        message = LegalActions()
        message.header.stamp = self.get_clock().now().to_msg()
        message.emergency_stop = self._emergency
        message.sonar_blocked = (not sonar_fresh) or self._front_range <= float(
            self.get_parameter("sonar_stop_distance_m").value
        )
        if self._emergency:
            message.actions = ["stop"]
            message.reason = "Emergency stop is active."
        else:
            core_pose = None
            if pose_fresh and self._pose is not None:
                core_pose = CorePose(
                    (self._pose.centre.x, self._pose.centre.y),
                    self._pose.heading if self._pose.has_heading else None,
                    self._pose.confidence,
                )
            shape = (
                int(self.get_parameter("camera_height").value),
                int(self.get_parameter("camera_width").value),
                3,
            )
            actions = allowed_actions(
                core_pose,
                transform_from_node(self),
                shape,
                float(self.get_parameter("margin_frac").value),
            )
            actions = apply_ultrasonic_guard(actions, message.sonar_blocked)
            message.actions = [action.value for action in actions]
            message.reason = "fresh guard calculation" if pose_fresh else "pose stale/lost; conservative recovery"
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
