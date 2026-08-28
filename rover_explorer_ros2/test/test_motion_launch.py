import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from std_msgs.msg import UInt32

from rover_explorer_ros2.msg import LegalActions, PolicyDecision, RoverPose


def generate_test_description():
    motion = launch_ros.actions.Node(
        package="rover_explorer_ros2",
        executable="motion_node.exe" if os.name == "nt" else "motion_node",
        parameters=[{
            "policy": "sweep",
            "legal_actions_timeout_seconds": 1.0,
            "translation_ms": 300,
            "turn_ms": 50,
            "settle_ms": 80,
            "recovery_scan_timeout_seconds": 1.0,
            "recovery_cooldown_ms": 0,
            "max_consecutive_turn_pulses": 2,
            "turn_burst_recheck_ms": 50,
            "turn_rearm_min_progress_degrees": 5.0,
            "max_continuous_turn_degrees": 220.0,
            "turn_pose_timeout_seconds": 1.0,
            "radians_per_turn_pulse": -0.48,
        }],
    )
    return launch.LaunchDescription([motion, launch_testing.actions.ReadyToTest()]), {"motion": motion}


class TestMotionSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("motion_safety_test")
        self.commands = []
        self.cmd_subscription = self.node.create_subscription(Twist, "/cmd_vel", self.commands.append, 10)
        self.legal = self.node.create_publisher(LegalActions, "/rover/legal_actions", 10)
        self.decisions = self.node.create_publisher(PolicyDecision, "/rover/policy/classic_decision", 10)
        self.poses = self.node.create_publisher(RoverPose, "/rover/pose", 10)
        self.left_range = self.node.create_publisher(Range, "/rover/sonar/left", 10)
        self.right_range = self.node.create_publisher(Range, "/rover/sonar/right", 10)
        self.scan_sequence = self.node.create_publisher(UInt32, "/rover/sonar/scan_sequence", 10)

    def tearDown(self):
        self.node.destroy_node()

    def test_illegal_policy_output_becomes_stop(self):
        # DDS discovery is noticeably slower on a cold Windows/Fast DDS start.
        # Keep publishing until both endpoints have discovered the motion node.
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and not self.commands:
            legal = LegalActions()
            legal.actions = ["stop"]
            decision = PolicyDecision()
            decision.action = "forward"
            self.legal.publish(legal)
            self.decisions.publish(decision)
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertTrue(self.commands)
        self.assertEqual(self.commands[-1].linear.x, 0.0)
        self.assertEqual(self.commands[-1].angular.z, 0.0)

    def test_exactly_one_cmd_vel_publisher(self):
        deadline = time.monotonic() + 12.0
        publishers = []
        while time.monotonic() < deadline:
            publishers = self.node.get_publishers_info_by_topic("/cmd_vel")
            if publishers:
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertEqual(len(publishers), 1)

    def test_bounded_translation_is_followed_by_explicit_stop(self):
        legal = LegalActions()
        legal.actions = ["forward", "stop"]
        decision = PolicyDecision()
        decision.action = "forward"
        moving = False
        stopped_after_motion = False
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not stopped_after_motion:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            rclpy.spin_once(self.node, timeout_sec=0.03)
            if not self.commands:
                continue
            command = self.commands[-1]
            moving = moving or command.linear.x > 0.0
            stopped_after_motion = moving and command.linear.x == 0.0 and command.angular.z == 0.0
        self.assertTrue(moving)
        self.assertTrue(stopped_after_motion, "bounded translation was not followed by STOP")

    def test_emergency_legal_state_preempts_motion(self):
        legal = LegalActions()
        legal.actions = ["stop"]
        legal.emergency_stop = True
        decision = PolicyDecision()
        decision.action = "forward"
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            rclpy.spin_once(self.node, timeout_sec=0.03)
            if self.commands and self.commands[-1].linear.x == 0.0:
                break
        self.assertTrue(self.commands)
        self.assertEqual(self.commands[-1].linear.x, 0.0)
        self.assertEqual(self.commands[-1].angular.z, 0.0)

    def test_stale_guard_or_policy_data_forces_stop(self):
        legal = LegalActions()
        legal.actions = ["forward", "stop"]
        decision = PolicyDecision()
        decision.action = "forward"
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            rclpy.spin_once(self.node, timeout_sec=0.03)
            if self.commands and self.commands[-1].linear.x > 0.0:
                break
        self.assertTrue(self.commands and self.commands[-1].linear.x > 0.0)

        deadline = time.monotonic() + 1.4
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertEqual(self.commands[-1].linear.x, 0.0)
        self.assertEqual(self.commands[-1].angular.z, 0.0)

    def test_policy_stop_interrupts_active_translation(self):
        legal = LegalActions()
        legal.actions = ["forward", "stop"]
        decision = PolicyDecision()
        decision.action = "forward"

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            rclpy.spin_once(self.node, timeout_sec=0.03)
            if self.commands and self.commands[-1].linear.x > 0.0:
                break
        self.assertTrue(self.commands and self.commands[-1].linear.x > 0.0)

        # The configured pulse lasts 300 ms. A policy STOP must reach cmd_vel
        # well before that natural deadline instead of completing the pulse.
        decision.action = "stop"
        stop_requested = time.monotonic()
        stopped = False
        deadline = stop_requested + 0.18
        while time.monotonic() < deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            rclpy.spin_once(self.node, timeout_sec=0.02)
            stopped = bool(
                self.commands
                and self.commands[-1].linear.x == 0.0
                and self.commands[-1].angular.z == 0.0
            )
            if stopped:
                break
        self.assertTrue(stopped, "policy STOP did not preempt the active translation pulse")
        self.assertLess(time.monotonic() - stop_requested, 0.18)

    def test_sonar_block_runs_reverse_then_turn_recovery(self):
        legal = LegalActions()
        legal.actions = ["backward", "turn_left", "turn_right", "stop"]
        legal.sonar_blocked = True
        decision = PolicyDecision()
        decision.action = "forward"
        left = Range()
        left.header.frame_id = "sonar_left"
        left.range = 0.8
        right = Range()
        right.header.frame_id = "sonar_right"
        right.range = 0.2

        # A newly blocked front must hold STOP until a new completed scan is
        # observed (or the bounded timeout expires).
        self.commands.clear()
        wait_deadline = time.monotonic() + 0.35
        while time.monotonic() < wait_deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            self.left_range.publish(left)
            self.right_range.publish(right)
            rclpy.spin_once(self.node, timeout_sec=0.03)
        self.assertFalse(any(command.linear.x < 0 for command in self.commands))
        self.assertFalse(any(command.angular.z != 0 for command in self.commands))

        scan = UInt32()
        scan.data = 1
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            self.left_range.publish(left)
            self.right_range.publish(right)
            self.scan_sequence.publish(scan)
            rclpy.spin_once(self.node, timeout_sec=0.05)
            reversed_once = any(command.linear.x < 0 for command in self.commands)
            turned_once = any(command.angular.z != 0 for command in self.commands)
            if reversed_once and turned_once:
                break
        self.assertTrue(any(command.linear.x < 0 for command in self.commands))
        self.assertTrue(any(command.angular.z > 0 for command in self.commands))

    def test_turn_burst_rearms_on_pose_progress_then_blocks_a_stall(self):
        legal = LegalActions()
        legal.actions = ["turn_left", "turn_right", "stop"]
        decision = PolicyDecision()
        pose = RoverPose()
        pose.has_heading = True

        # Clear any recovery or turn state left by an earlier integration test.
        decision.action = "stop"
        reset_deadline = time.monotonic() + 0.4
        while time.monotonic() < reset_deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            self.poses.publish(pose)
            rclpy.spin_once(self.node, timeout_sec=0.03)

        self.commands.clear()
        decision.action = "turn_left"
        first_turn_seen = False
        second_burst_seen = False
        previous_turning = False
        turn_starts = 0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not second_burst_seen:
            # A physical left pulse has negative calibrated image-heading delta.
            # Publish progress after the first command, then keep the pose fresh
            # through the mandatory stationary verification window.
            pose.heading = -0.25 if first_turn_seen else 0.0
            self.legal.publish(legal)
            self.decisions.publish(decision)
            self.poses.publish(pose)
            rclpy.spin_once(self.node, timeout_sec=0.03)
            if self.commands:
                turning = self.commands[-1].angular.z > 0.0
                if turning and not previous_turning:
                    turn_starts += 1
                    first_turn_seen = True
                    second_burst_seen = turn_starts >= 2
                previous_turning = turning

        self.assertTrue(second_burst_seen, "verified heading progress did not rearm turning")

        # Leave the heading unchanged during the second burst. Its verification
        # must detect zero progress and keep the rover stopped instead of allowing
        # a third, potentially endless burst.
        # Let the already-started second burst finish and enter its stationary
        # verification phase. Individual pulses inside that burst are expected;
        # only motion after the failed verification would be a safety defect.
        verification_deadline = time.monotonic() + 0.65
        while time.monotonic() < verification_deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            self.poses.publish(pose)
            rclpy.spin_once(self.node, timeout_sec=0.03)

        self.commands.clear()
        blocked_deadline = time.monotonic() + 0.6
        while time.monotonic() < blocked_deadline:
            self.legal.publish(legal)
            self.decisions.publish(decision)
            self.poses.publish(pose)
            rclpy.spin_once(self.node, timeout_sec=0.03)
        self.assertTrue(self.commands)
        self.assertTrue(
            all(command.angular.z == 0.0 for command in self.commands),
            "stalled turn burst was incorrectly rearmed",
        )
