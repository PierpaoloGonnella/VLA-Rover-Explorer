import time
import unittest

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing
import launch_testing.asserts
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Range

from rover_explorer_ros2.msg import LegalActions, RoverPose


def generate_test_description():
    launch_file = str(
        get_package_share_directory("rover_explorer_ros2") + "/launch/sim.launch.py"
    )
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments={"policy": "sweep", "planner": "astar_legacy"}.items(),
    )
    return launch.LaunchDescription([
        simulation,
        launch_testing.actions.ReadyToTest(),
    ])


class TestHybridSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_python_cpp_topic_interoperability_and_single_motion_owner(self):
        node = rclpy.create_node("hybrid_simulation_test")
        poses = []
        legal = []
        coverage = []
        occupancy = []
        commands = []
        subscriptions = [
            node.create_subscription(RoverPose, "/rover/pose", poses.append, 10),
            node.create_subscription(LegalActions, "/rover/legal_actions", legal.append, 10),
            node.create_subscription(OccupancyGrid, "/rover/coverage_map", coverage.append, 10),
            node.create_subscription(OccupancyGrid, "/rover/occupancy_grid", occupancy.append, 10),
            node.create_subscription(Twist, "/cmd_vel", commands.append, 10),
        ]
        sonar = node.create_publisher(Range, "/rover/sonar", 10)
        finite_front = Range()
        finite_front.header.frame_id = "sonar_front"
        finite_front.range = 1.0
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline and not all(
            (poses, legal, coverage, occupancy, commands)
        ):
            sonar.publish(finite_front)
            rclpy.spin_once(node, timeout_sec=0.1)

        self.assertTrue(poses, "C++ localizer did not interoperate through RoverPose")
        self.assertTrue(legal, "C++ guard did not publish LegalActions")
        self.assertTrue(coverage, "C++ coverage map was not published")
        self.assertTrue(occupancy, "C++ obstacle map was not published")
        self.assertTrue(commands, "C++ motion node did not publish /cmd_vel")
        self.assertEqual(len(node.get_publishers_info_by_topic("/cmd_vel")), 1)
        self.assertEqual(len(node.get_publishers_info_by_topic("/rover/pose")), 1)
        self.assertTrue(subscriptions)  # Keep subscriptions alive through all assertions.
        node.destroy_node()


@launch_testing.post_shutdown_test()
class TestHybridSimulationShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
