import math
import os
import time
import unittest

import cv2
import launch
import launch_ros.actions
import launch_testing
import launch_testing.asserts
import numpy as np
import rclpy
from sensor_msgs.msg import Image

from rover_explorer.simulator import RoverSimulator
from rover_explorer_ros2.msg import RoverPose


def _executable(name):
    return f"{name}.exe" if os.name == "nt" else name


def _node(executable, name, backend, image_topic, pose_topic, error_topic):
    return launch_ros.actions.Node(
        package="rover_explorer_ros2",
        executable=_executable(executable),
        name=name,
        parameters=[{
            "aruco_marker_id": 0,
            "aruco_heading_offset_degrees": -21.0,
            "localization_backend": backend,
            "min_confidence": 0.0,
        }],
        remappings=[
            ("/rover/image_raw", image_topic),
            ("/rover/pose", pose_topic),
            ("/rover/localization/aruco_error_px", error_topic),
        ],
    )


def generate_test_description():
    processes = [
        _node("localizer_node", "localizer_cpp_parity", "aruco_custom",
              "/parity/aruco_image", "/parity/aruco_cpp_pose", "/parity/aruco_cpp_error"),
        _node("localizer_node_python", "localizer_python_parity", "aruco_custom",
              "/parity/aruco_image", "/parity/aruco_python_pose", "/parity/aruco_python_error"),
        _node("localizer_node", "color_cpp_parity", "color",
              "/parity/color_image", "/parity/color_cpp_pose", "/parity/color_cpp_error"),
        _node("localizer_node_python", "color_python_parity", "color",
              "/parity/color_image", "/parity/color_python_pose", "/parity/color_python_error"),
    ]
    return launch.LaunchDescription([*processes, launch_testing.actions.ReadyToTest()])


class TestLocalizerParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node(f"localizer_parity_{self._testMethodName}")
        self.aruco_cpp = []
        self.aruco_python = []
        self.color_cpp = []
        self.color_python = []
        self.subscriptions = [
            self.node.create_subscription(
                RoverPose, "/parity/aruco_cpp_pose", self.aruco_cpp.append, 10),
            self.node.create_subscription(
                RoverPose, "/parity/aruco_python_pose", self.aruco_python.append, 10),
            self.node.create_subscription(
                RoverPose, "/parity/color_cpp_pose", self.color_cpp.append, 10),
            self.node.create_subscription(
                RoverPose, "/parity/color_python_pose", self.color_python.append, 10),
        ]
        self.aruco_images = self.node.create_publisher(Image, "/parity/aruco_image", 10)
        self.color_images = self.node.create_publisher(Image, "/parity/color_image", 10)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not (
            self.aruco_images.get_subscription_count() >= 2
            and self.color_images.get_subscription_count() >= 2
        ):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertGreaterEqual(self.aruco_images.get_subscription_count(), 2)
        self.assertGreaterEqual(self.color_images.get_subscription_count(), 2)

    def tearDown(self):
        self.node.destroy_node()

    @staticmethod
    def _message(frame, sequence=1, encoding="bgr8"):
        message = Image()
        message.header.stamp.sec = 1000 + sequence
        message.header.stamp.nanosec = 123456789
        message.header.frame_id = f"parity_camera_{sequence}"
        message.height, message.width = frame.shape[:2]
        message.encoding = encoding
        message.is_bigendian = False
        message.step = message.width * 3
        message.data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        return message

    def _wait_for_pair(self, publisher, message, cpp, python, timeout=4.0):
        cpp.clear()
        python.clear()
        publisher.publish(message)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            cpp_match = next((pose for pose in cpp if pose.header == message.header), None)
            python_match = next(
                (pose for pose in python if pose.header == message.header), None)
            if cpp_match is not None and python_match is not None:
                return cpp_match, python_match
        self.fail("native and Python localizers did not publish matching-frame poses")

    def _assert_pose_parity(self, cpp, python, message, heading=True):
        # Both bindings execute OpenCV 4.10 corner refinement on float32 points.
        # 0.05 px and 1e-4 rad cover only binding/serialization rounding.
        self.assertLess(math.dist(
            (cpp.centre.x, cpp.centre.y),
            (python.centre.x, python.centre.y)), 0.05)
        self.assertEqual(cpp.has_heading, heading)
        self.assertEqual(cpp.has_heading, python.has_heading)
        if heading:
            error = (cpp.heading - python.heading + math.pi) % (2 * math.pi) - math.pi
            self.assertLess(abs(error), 1e-4)
        else:
            self.assertEqual(cpp.heading, 0.0)
            self.assertEqual(python.heading, 0.0)
        self.assertAlmostEqual(cpp.confidence, python.confidence, delta=1e-5)
        self.assertEqual(cpp.header, message.header)
        self.assertEqual(python.header, message.header)

    def test_aruco_rotations_boundary_offset_headers_and_repeated_frames(self):
        simulator = RoverSimulator(width=640, height=480, wheel_slip=0)
        cases = [
            (320.0, 240.0, -2.2),
            (320.0, 240.0, -0.7),
            (320.0, 240.0, 0.0),
            (320.0, 240.0, 0.85),
            (34.0, 34.0, 0.15),
        ]
        for sequence, (x, y, heading) in enumerate(cases, 1):
            simulator.x, simulator.y, simulator.heading = x, y, heading
            message = self._message(simulator.render(), sequence)
            cpp, python = self._wait_for_pair(
                self.aruco_images, message, self.aruco_cpp, self.aruco_python)
            self._assert_pose_parity(cpp, python, message)

        repeated = self._message(simulator.render(), 99)
        first = self._wait_for_pair(
            self.aruco_images, repeated, self.aruco_cpp, self.aruco_python)
        second = self._wait_for_pair(
            self.aruco_images, repeated, self.aruco_cpp, self.aruco_python)
        self._assert_pose_parity(*first, repeated)
        self._assert_pose_parity(*second, repeated)

    def test_blank_occluded_wrong_marker_unsupported_and_malformed_publish_nothing(self):
        frames = [np.zeros((480, 640, 3), np.uint8)]
        simulator = RoverSimulator(width=640, height=480, wheel_slip=0)
        occluded = simulator.render()
        cv2.rectangle(occluded, (300, 215), (355, 270), (225, 230, 225), -1)
        frames.append(occluded)
        wrong = RoverSimulator(width=640, height=480, wheel_slip=0, marker_id=1)
        frames.append(wrong.render())

        unsupported = self._message(frames[0], 201, encoding="rgb8")
        malformed = self._message(frames[0], 202)
        malformed.data = malformed.data[:-1]
        messages = [self._message(frame, 200 + index) for index, frame in enumerate(frames)]
        messages.extend([unsupported, malformed])
        self.aruco_cpp.clear()
        self.aruco_python.clear()
        for message in messages:
            self.aruco_images.publish(message)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        rejected_headers = [message.header for message in messages]
        self.assertFalse(any(pose.header in rejected_headers for pose in self.aruco_cpp))
        self.assertFalse(any(pose.header in rejected_headers for pose in self.aruco_python))

    def test_color_backend_position_only_parity(self):
        frame = np.zeros((200, 300, 3), np.uint8)
        cv2.rectangle(frame, (100, 60), (139, 89), (0, 255, 0), -1)
        message = self._message(frame, 301)
        cpp, python = self._wait_for_pair(
            self.color_images, message, self.color_cpp, self.color_python)
        self._assert_pose_parity(cpp, python, message, heading=False)

    def test_stale_frame_is_not_republished_and_diagnostic_topics_exist(self):
        simulator = RoverSimulator(width=640, height=480, wheel_slip=0)
        message = self._message(simulator.render(), 401)
        self._wait_for_pair(
            self.aruco_images, message, self.aruco_cpp, self.aruco_python)
        self.aruco_cpp.clear()
        self.aruco_python.clear()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertFalse(self.aruco_cpp)
        self.assertFalse(self.aruco_python)
        self.assertEqual(
            len(self.node.get_publishers_info_by_topic("/parity/aruco_cpp_error")), 1)
        self.assertEqual(
            len(self.node.get_publishers_info_by_topic("/parity/aruco_python_error")), 1)


@launch_testing.post_shutdown_test()
class TestLocalizerParityShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
