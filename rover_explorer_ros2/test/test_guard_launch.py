import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import rclpy
from sensor_msgs.msg import Range
from std_msgs.msg import Bool

from rover_explorer_ros2.msg import LegalActions, RoverPose


def generate_test_description():
    guard = launch_ros.actions.Node(
        package="rover_explorer_ros2",
        executable="guard_node.exe" if os.name == "nt" else "guard_node",
        parameters=[{
            "camera_width": 640,
            "camera_height": 480,
            "pose_timeout_seconds": 0.2,
            "sonar_timeout_seconds": 0.2,
            "sonar_stop_distance_m": 0.25,
            "px_per_forward_pulse": 30.0,
        }],
    )
    return launch.LaunchDescription([guard, launch_testing.actions.ReadyToTest()]), {"guard": guard}


class TestGuardCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("guard_compatibility_test")
        self.legal_messages = []
        self.legal_subscription = self.node.create_subscription(
            LegalActions, "/rover/legal_actions", self.legal_messages.append, 10
        )
        self.poses = self.node.create_publisher(RoverPose, "/rover/pose", 10)
        self.sonar = self.node.create_publisher(Range, "/rover/sonar", 10)
        self.emergency = self.node.create_publisher(Bool, "/rover/emergency_stop", 10)

    def tearDown(self):
        self.node.destroy_node()

    def _wait_for(self, predicate, publish, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            publish()
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.legal_messages and predicate(self.legal_messages[-1]):
                return self.legal_messages[-1]
        self.fail("guard did not publish the expected compatible LegalActions message")

    def test_front_sonar_veto_and_emergency_stop(self):
        pose = RoverPose()
        pose.centre.x = 320.0
        pose.centre.y = 240.0
        pose.has_heading = True
        pose.heading = 0.0
        sonar = Range()
        sonar.header.frame_id = "sonar_front"
        sonar.range = 0.1

        blocked = self._wait_for(
            lambda message: message.sonar_blocked and "forward" not in message.actions,
            lambda: (self.poses.publish(pose), self.sonar.publish(sonar)),
        )
        self.assertIn("backward", blocked.actions)
        self.assertIn("stop", blocked.actions)

        emergency = Bool()
        emergency.data = True
        stopped = self._wait_for(
            lambda message: message.emergency_stop,
            lambda: self.emergency.publish(emergency),
        )
        self.assertEqual(stopped.actions, ["stop"])

    def test_stale_pose_and_sonar_fail_closed(self):
        clear_emergency = Bool()
        clear_emergency.data = False
        self._wait_for(
            lambda message: (
                message.sonar_blocked
                and message.actions == ["backward", "stop"]
                and "stale/lost" in message.reason
            ),
            lambda: self.emergency.publish(clear_emergency),
        )
