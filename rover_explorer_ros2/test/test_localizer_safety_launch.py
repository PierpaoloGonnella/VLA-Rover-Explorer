import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.asserts
import numpy as np
import rclpy
from sensor_msgs.msg import Image, Range

from rover_explorer.simulator import RoverSimulator
from rover_explorer_ros2.msg import LegalActions


def _executable(name):
    return f"{name}.exe" if os.name == "nt" else name


def generate_test_description():
    localizer = launch_ros.actions.Node(
        package="rover_explorer_ros2",
        executable=_executable("localizer_node"),
        name="localizer_safety",
        parameters=[{"min_confidence": 0.0}],
        remappings=[
            ("/rover/image_raw", "/safety/image"),
            ("/rover/pose", "/safety/pose"),
        ],
    )
    guard = launch_ros.actions.Node(
        package="rover_explorer_ros2",
        executable=_executable("guard_node"),
        name="guard_localizer_safety",
        parameters=[{
            "camera_width": 640,
            "camera_height": 480,
            "pose_timeout_seconds": 0.2,
            "sonar_timeout_seconds": 0.2,
            "sonar_stop_distance_m": 0.25,
            "px_per_forward_pulse": 30.0,
        }],
        remappings=[
            ("/rover/pose", "/safety/pose"),
            ("/rover/sonar", "/safety/sonar"),
            ("/rover/legal_actions", "/safety/legal_actions"),
        ],
    )
    return launch.LaunchDescription([localizer, guard, launch_testing.actions.ReadyToTest()])


class TestLocalizationLossSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("localizer_loss_test")
        self.legal = []
        self.subscription = self.node.create_subscription(
            LegalActions, "/safety/legal_actions", self.legal.append, 10)
        self.images = self.node.create_publisher(Image, "/safety/image", 10)
        self.sonar = self.node.create_publisher(Range, "/safety/sonar", 10)
        self.frame = RoverSimulator(width=640, height=480, wheel_slip=0).render()

    def tearDown(self):
        self.node.destroy_node()

    def _image(self, valid=True):
        message = Image()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = "safety_camera"
        message.height, message.width = self.frame.shape[:2]
        message.encoding = "bgr8"
        message.step = message.width * 3
        message.data = np.ascontiguousarray(self.frame).tobytes()
        if not valid:
            message.data = message.data[:-1]
        return message

    @staticmethod
    def _sonar():
        message = Range()
        message.header.frame_id = "sonar_front"
        message.range = 1.0
        return message

    def _wait(self, predicate, publish, timeout=4.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            publish()
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.legal and predicate(self.legal[-1]):
                return self.legal[-1]
        self.fail("guard did not reach the expected localization safety state")

    def _wait_fresh(self):
        return self._wait(
            lambda message: "forward" in message.actions and "fresh" in message.reason,
            lambda: (self.images.publish(self._image()), self.sonar.publish(self._sonar())),
        )

    def _wait_stale(self, publish_image=None):
        def publish():
            if publish_image is not None:
                self.images.publish(publish_image())
            self.sonar.publish(self._sonar())

        return self._wait(
            lambda message: (
                message.actions == ["backward", "stop"]
                and "stale/lost" in message.reason
            ),
            publish,
        )

    def test_stopped_and_invalid_image_streams_fail_closed(self):
        self._wait_fresh()
        self._wait_stale()
        self._wait_fresh()
        self._wait_stale(lambda: self._image(valid=False))
        self.assertEqual(len(self.node.get_publishers_info_by_topic("/safety/pose")), 1)


@launch_testing.post_shutdown_test()
class TestLocalizationLossShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
