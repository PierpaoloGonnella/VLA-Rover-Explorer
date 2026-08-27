from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import UInt32

from rover_explorer_ros2.msg import LegalActions, PolicyDecision, RoverPose


class MotionNode(Node):
    """Arbitrates advisory policy outputs and republishes only fresh legal motion."""

    def __init__(self) -> None:
        super().__init__("motion_node")
        self.declare_parameter("policy", "sweep")
        self.declare_parameter("decision_timeout_seconds", 1.0)
        self.declare_parameter("legal_actions_timeout_seconds", 0.25)
        self.declare_parameter("translation_ms", 250)
        self.declare_parameter("turn_ms", 180)
        self.declare_parameter("settle_ms", 500)
        self.declare_parameter("turn_scale", 0.55)
        self.declare_parameter("recovery_scan_timeout_seconds", 2.5)
        self.declare_parameter("recovery_cooldown_ms", 250)
        self.declare_parameter("max_consecutive_turn_pulses", 3)
        self.declare_parameter("turn_burst_recheck_ms", 150)
        self.declare_parameter("turn_rearm_min_progress_degrees", 8.0)
        self.declare_parameter("max_continuous_turn_degrees", 220.0)
        self.declare_parameter("turn_pose_timeout_seconds", 1.0)
        self.declare_parameter("radians_per_turn_pulse", -0.48)
        self._legal = {"stop"}
        self._legal_received = 0.0
        self._emergency_stop = False
        self._sonar_blocked = False
        self._scan_sequence = 0
        self._scan_baseline = 0
        self._left_range = float("inf")
        self._right_range = float("inf")
        self._recovery_stage: str | None = None
        self._recovery_started = 0.0
        self._recovery_turn = "turn_left"
        self._recovery_cooldown_until = 0.0
        self._decision: PolicyDecision | None = None
        self._decision_received = 0.0
        self._active_action = "stop"
        self._pulse_until = 0.0
        self._settle_until = 0.0
        self._consecutive_turn_pulses = 0
        self._turn_limit_warning_logged = False
        self._pose_heading: float | None = None
        self._pose_received = 0.0
        self._pose_sequence = 0
        self._turn_burst_action: str | None = None
        self._turn_burst_start_heading: float | None = None
        self._turn_verification_pending = False
        self._turn_verification_after = 0.0
        self._turn_verification_pose_sequence = 0
        self._continuous_turn_radians = 0.0
        self._turn_progress_blocked = False
        self._publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LegalActions, "/rover/legal_actions", self._on_legal, 10)
        self.create_subscription(RoverPose, "/rover/pose", self._on_pose, 10)
        self.create_subscription(Range, "/rover/sonar/left", self._on_side_range, 10)
        self.create_subscription(Range, "/rover/sonar/right", self._on_side_range, 10)
        self.create_subscription(UInt32, "/rover/sonar/scan_sequence", self._on_scan_sequence, 10)
        # Even in VLM mode, motor primitives come from the fast classic
        # controller. The VLM only changes its semantic waypoint.
        self.create_subscription(
            PolicyDecision, "/rover/policy/classic_decision", self._on_decision, 10
        )
        self.create_timer(0.05, self._publish_command)

    def _on_legal(self, message: LegalActions) -> None:
        was_blocked = self._sonar_blocked
        self._legal = set(message.actions) if not message.emergency_stop else {"stop"}
        self._legal_received = time.monotonic()
        self._emergency_stop = bool(message.emergency_stop)
        self._sonar_blocked = bool(message.sonar_blocked)
        if self._emergency_stop:
            self._recovery_stage = None
        elif (
            str(self.get_parameter("policy").value) != "vlm"
            and self._sonar_blocked
            and not was_blocked
            and self._recovery_stage is None
        ):
            self._start_recovery(self._legal_received)

    def _on_side_range(self, message: Range) -> None:
        if message.header.frame_id.endswith("left"):
            self._left_range = float(message.range)
        elif message.header.frame_id.endswith("right"):
            self._right_range = float(message.range)

    def _on_scan_sequence(self, message: UInt32) -> None:
        self._scan_sequence = int(message.data)

    def _start_recovery(self, now: float) -> None:
        self._recovery_stage = "wait_scan"
        self._recovery_started = now
        self._scan_baseline = self._scan_sequence
        self.get_logger().warning("Front obstacle: starting scan/reverse/turn recovery")

    def _recovery_request(self, now: float) -> str | None:
        if self._recovery_stage is None or self._emergency_stop:
            return None
        if self._recovery_stage == "wait_scan":
            scan_ready = self._scan_sequence > self._scan_baseline
            scan_timed_out = now - self._recovery_started >= float(
                self.get_parameter("recovery_scan_timeout_seconds").value
            )
            if not scan_ready and not scan_timed_out:
                return "stop"
            self._recovery_turn = (
                "turn_left" if self._left_range >= self._right_range else "turn_right"
            )
            self._recovery_stage = "backward" if "backward" in self._legal else "turning"
        if self._recovery_stage == "backward":
            return "backward" if "backward" in self._legal else "stop"
        if self._recovery_stage == "turning":
            return self._recovery_turn if self._recovery_turn in self._legal else "stop"
        if self._recovery_stage == "cooldown":
            if now < self._recovery_cooldown_until:
                return "stop"
            self._recovery_stage = None
            if self._sonar_blocked:
                self._start_recovery(now)
                return "stop"
            return None
        return "stop"

    def _recovery_pulse_started(self, action: str) -> None:
        if self._recovery_stage == "backward" and action == "backward":
            self._recovery_stage = "turning"
        elif self._recovery_stage == "turning" and action == self._recovery_turn:
            self._recovery_stage = "cooldown"
            self._recovery_cooldown_until = self._settle_until + max(
                0, int(self.get_parameter("recovery_cooldown_ms").value)
            ) / 1000.0

    def _on_decision(self, message: PolicyDecision) -> None:
        self._decision = message
        self._decision_received = time.monotonic()

    def _on_pose(self, message: RoverPose) -> None:
        if not message.has_heading or not math.isfinite(float(message.heading)):
            return
        self._pose_heading = float(message.heading)
        self._pose_received = time.monotonic()
        self._pose_sequence += 1

    @staticmethod
    def _wrapped_angle(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def _reset_turn_guard(self) -> None:
        self._consecutive_turn_pulses = 0
        self._turn_limit_warning_logged = False
        self._turn_burst_action = None
        self._turn_burst_start_heading = None
        self._turn_verification_pending = False
        self._turn_verification_after = 0.0
        self._turn_verification_pose_sequence = self._pose_sequence
        self._continuous_turn_radians = 0.0
        self._turn_progress_blocked = False

    def _begin_turn_burst(self, action: str, now: float) -> None:
        self._turn_burst_action = action
        pose_fresh = (
            self._pose_heading is not None
            and now - self._pose_received
            <= float(self.get_parameter("turn_pose_timeout_seconds").value)
        )
        self._turn_burst_start_heading = self._pose_heading if pose_fresh else None

    def _arm_turn_verification(self, now: float) -> None:
        self._turn_verification_pending = True
        self._turn_verification_after = now + max(
            0, int(self.get_parameter("turn_burst_recheck_ms").value)
        ) / 1000.0
        # Demand a pose sampled after the rover has entered the stationary
        # verification phase; a pose received during the last pulse is not enough.
        self._turn_verification_pose_sequence = self._pose_sequence
        self._turn_limit_warning_logged = True
        self.get_logger().info(
            "Turn burst complete; holding STOP while fresh pose verifies progress."
        )

    def _turn_burst_can_rearm(self, now: float) -> bool:
        if self._turn_progress_blocked:
            return False
        if not self._turn_verification_pending:
            self._arm_turn_verification(now)
            return False
        pose_fresh = (
            self._pose_heading is not None
            and now - self._pose_received
            <= float(self.get_parameter("turn_pose_timeout_seconds").value)
        )
        if (
            now < self._turn_verification_after
            or not pose_fresh
            or self._pose_sequence <= self._turn_verification_pose_sequence
        ):
            return False

        start = self._turn_burst_start_heading
        if start is None or self._turn_burst_action is None:
            self._turn_progress_blocked = True
            self.get_logger().error(
                "Turn burst cannot be verified from a fresh starting pose; forcing STOP until policy changes."
            )
            return False

        observed_delta = self._wrapped_angle(float(self._pose_heading) - start)
        calibrated_left_delta = float(
            self.get_parameter("radians_per_turn_pulse").value
        )
        expected_sign = 1.0 if calibrated_left_delta >= 0.0 else -1.0
        if self._turn_burst_action == "turn_right":
            expected_sign *= -1.0
        directed_progress = observed_delta * expected_sign
        minimum_progress = math.radians(
            max(0.0, float(self.get_parameter("turn_rearm_min_progress_degrees").value))
        )
        maximum_continuous = math.radians(
            max(1.0, float(self.get_parameter("max_continuous_turn_degrees").value))
        )

        if directed_progress < minimum_progress:
            self._turn_progress_blocked = True
            self.get_logger().error(
                "Turn burst made insufficient or wrong-direction progress "
                f"({math.degrees(directed_progress):.1f} deg); forcing STOP until policy changes."
            )
            return False
        if self._continuous_turn_radians + directed_progress > maximum_continuous:
            self._turn_progress_blocked = True
            self.get_logger().error(
                "Continuous verified turn limit reached; forcing STOP until policy changes."
            )
            return False

        self._continuous_turn_radians += directed_progress
        self._consecutive_turn_pulses = 0
        self._turn_burst_start_heading = None
        self._turn_verification_pending = False
        self._turn_limit_warning_logged = False
        self.get_logger().info(
            "Turn burst rearmed after "
            f"{math.degrees(directed_progress):.1f} deg of verified progress."
        )
        return True

    def _twist(self, action: str) -> Twist:
        message = Twist()
        turn_scale = max(0.1, min(1.0, float(self.get_parameter("turn_scale").value)))
        values = {
            "forward": (1.0, 0.0), "backward": (-1.0, 0.0),
            "turn_left": (0.0, turn_scale), "turn_right": (0.0, -turn_scale),
            "arc_left": (0.65, 0.35), "arc_right": (0.65, -0.35),
            "stop": (0.0, 0.0),
        }
        message.linear.x, message.angular.z = values.get(action, (0.0, 0.0))
        return message

    def _publish_command(self) -> None:
        now = time.monotonic()
        legal_fresh = now - self._legal_received <= float(self.get_parameter("legal_actions_timeout_seconds").value)
        decision_fresh = now - self._decision_received <= float(self.get_parameter("decision_timeout_seconds").value)
        recovery_requested = (
            self._recovery_request(now)
            if legal_fresh and str(self.get_parameter("policy").value) != "vlm"
            else None
        )
        requested = recovery_requested or (
            self._decision.action if self._decision is not None else "stop"
        )
        if not legal_fresh or (recovery_requested is None and not decision_fresh) or requested not in self._legal:
            self._active_action = "stop"
            self._pulse_until = self._settle_until = now
            self._reset_turn_guard()
            self._publisher.publish(self._twist("stop"))
            return

        # Any safety change interrupts a pulse. Policy changes are applied only
        # after the current bounded pulse and stationary settling interval.
        if self._active_action not in self._legal:
            self._active_action = "stop"
            self._pulse_until = self._settle_until = now
            self._reset_turn_guard()
            self._publisher.publish(self._twist("stop"))
            return
        # STOP is the only policy change that preempts a bounded primitive. This
        # is essential near a reached waypoint: waiting for the remainder of a
        # translation pulse can move the rover well beyond the hold radius.
        if requested == "stop":
            self._active_action = "stop"
            self._pulse_until = self._settle_until = now
            self._reset_turn_guard()
            self._publisher.publish(self._twist("stop"))
            return
        if now < self._pulse_until:
            self._publisher.publish(self._twist(self._active_action))
            return
        if now < self._settle_until:
            self._publisher.publish(self._twist("stop"))
            return
        is_turn = requested in {"turn_left", "turn_right"}
        if is_turn and self._turn_burst_action not in {None, requested}:
            # An opposite turn is a genuine policy change and starts a new,
            # independently bounded maneuver.
            self._reset_turn_guard()
        maximum_turns = max(1, int(self.get_parameter("max_consecutive_turn_pulses").value))
        if is_turn and self._consecutive_turn_pulses >= maximum_turns:
            self._active_action = "stop"
            self._publisher.publish(self._twist("stop"))
            self._turn_burst_can_rearm(now)
            return
        if not is_turn:
            self._reset_turn_guard()

        if is_turn and self._consecutive_turn_pulses == 0:
            self._begin_turn_burst(requested, now)

        self._active_action = requested
        duration_ms = (
            int(self.get_parameter("turn_ms").value)
            if requested in {"turn_left", "turn_right"}
            else int(self.get_parameter("translation_ms").value)
        )
        self._pulse_until = now + max(0, duration_ms) / 1000.0
        self._settle_until = self._pulse_until + max(
            0, int(self.get_parameter("settle_ms").value)
        ) / 1000.0
        self._recovery_pulse_started(self._active_action)
        if is_turn:
            self._consecutive_turn_pulses += 1
        self._publisher.publish(self._twist(self._active_action))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
