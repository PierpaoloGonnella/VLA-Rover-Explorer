from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, Range

from rover_explorer.coverage import CoverageTracker
from rover_explorer.localize import RoverPose as CorePose
from rover_explorer.motion import Action
from rover_explorer.policy import VlmExplorer
from rover_explorer.temporal import TemporalMemory, TemporalStatus
from rover_explorer_ros2.msg import LegalActions, PolicyDecision, RoverPose, VlmAdvisory

from .common import ImageBridge, action_from_string, declare_transform_parameters, transform_from_node


class PolicyVlmNode(Node):
    """Advisory-only VLM policy. Every exceptional path publishes STOP."""

    def __init__(self) -> None:
        super().__init__("policy_vlm_node")
        declare_transform_parameters(self)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("coverage_cols", 6)
        self.declare_parameter("coverage_rows", 4)
        self.declare_parameter("vlm_grid_cols", 12)
        self.declare_parameter("vlm_grid_rows", 8)
        self.declare_parameter("ollama_url", "http://localhost:11434")
        self.declare_parameter("ollama_model", "qwen2.5vl:3b")
        self.declare_parameter("ollama_timeout_seconds", 20.0)
        self.declare_parameter("ollama_keep_alive", "30m")
        self.declare_parameter("vlm_min_advisory_confidence", 0.35)
        self.declare_parameter("vlm_memory_window_seconds", 20.0)
        self.declare_parameter("vlm_stall_translation_attempts", 2)
        self.declare_parameter("vlm_stall_displacement_px", 12.0)
        self.declare_parameter("vlm_repeated_turn_attempts", 3)
        self._bridge = ImageBridge()
        self._frame = None
        self._frame_stamp = None
        self._pose: CorePose | None = None
        self._legal = [Action.STOP]
        self._sonar_blocked = False
        self._ranges = {"front": float("inf"), "left": float("inf"), "right": float("inf")}
        self._memory = TemporalMemory(
            window_seconds=float(self.get_parameter("vlm_memory_window_seconds").value),
            stall_translation_attempts=int(
                self.get_parameter("vlm_stall_translation_attempts").value
            ),
            stall_displacement_px=float(self.get_parameter("vlm_stall_displacement_px").value),
            repeated_turn_attempts=int(self.get_parameter("vlm_repeated_turn_attempts").value),
        )
        self._prior_frames: deque = deque(maxlen=2)
        self._last_cmd_action = Action.STOP
        self._coverage = CoverageTracker(
            (int(self.get_parameter("camera_height").value), int(self.get_parameter("camera_width").value), 3),
            int(self.get_parameter("vlm_grid_cols").value),
            int(self.get_parameter("vlm_grid_rows").value),
        )
        self._policy = VlmExplorer(
            transform_from_node(self),
            str(self.get_parameter("ollama_url").value),
            str(self.get_parameter("ollama_model").value),
            float(self.get_parameter("ollama_timeout_seconds").value),
            annotation_style="grid",
            keep_alive=str(self.get_parameter("ollama_keep_alive").value),
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-policy")
        self._future: Future | None = None
        self._future_source_pose: CorePose | None = None
        self._future_source_stamp = None
        self._future_source_frame = None
        self._future_temporal_context = ""
        self._future_temporal_status = TemporalStatus(False, False, 0.0, (), 0.0)
        self._publisher = self.create_publisher(VlmAdvisory, "/rover/vlm/advisory", 10)
        self._debug_publisher = self.create_publisher(Image, "/rover/vlm/debug_image", 2)
        self.create_subscription(Image, "/rover/image_raw", self._on_image, 10)
        self.create_subscription(RoverPose, "/rover/pose", self._on_pose, 10)
        self.create_subscription(LegalActions, "/rover/legal_actions", self._on_legal, 10)
        self.create_subscription(Range, "/rover/sonar/front", self._on_range, 10)
        self.create_subscription(Range, "/rover/sonar/left", self._on_range, 10)
        self.create_subscription(Range, "/rover/sonar/right", self._on_range, 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(
            PolicyDecision, "/rover/policy/classic_decision", self._on_decision, 10
        )
        self.create_timer(0.1, self._tick)

    def _on_image(self, message: Image) -> None:
        self._frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        self._frame_stamp = message.header.stamp

    def _on_pose(self, message: RoverPose) -> None:
        self._pose = CorePose(
            (message.centre.x, message.centre.y),
            message.heading if message.has_heading else None,
            message.confidence,
        )
        self._coverage.update(self._pose)
        self._memory.add_pose(self._pose)

    def _on_legal(self, message: LegalActions) -> None:
        self._legal = [action_from_string(value) for value in message.actions] or [Action.STOP]
        self._sonar_blocked = bool(message.sonar_blocked)

    def _on_range(self, message: Range) -> None:
        for direction in self._ranges:
            if message.header.frame_id.endswith(direction):
                self._ranges[direction] = float(message.range)
                break

    @staticmethod
    def _action_from_twist(message: Twist) -> Action:
        linear, angular = float(message.linear.x), float(message.angular.z)
        if abs(linear) < 1e-3 and abs(angular) < 1e-3:
            return Action.STOP
        if linear > 0 and angular > 1e-3:
            return Action.ARC_LEFT
        if linear > 0 and angular < -1e-3:
            return Action.ARC_RIGHT
        if linear > 1e-3:
            return Action.FORWARD
        if linear < -1e-3:
            return Action.BACKWARD
        return Action.TURN_LEFT if angular > 0 else Action.TURN_RIGHT

    def _on_cmd_vel(self, message: Twist) -> None:
        action = self._action_from_twist(message)
        if action != Action.STOP and self._last_cmd_action == Action.STOP:
            self._memory.add_action(action)
        self._last_cmd_action = action

    def _on_decision(self, message: PolicyDecision) -> None:
        self._memory.record_outcome(f"{message.action}: {message.reason}")

    def _sensor_context(self) -> str:
        def render(value: float) -> str:
            return "unknown" if value == float("inf") else f"{value:.2f} m"

        return (
            f"front_blocked={self._sonar_blocked}; "
            f"front={render(self._ranges['front'])}; "
            f"left={render(self._ranges['left'])}; "
            f"right={render(self._ranges['right'])}. "
            "Ultrasonic data has priority over visual interpretation."
        )

    def _invalid(self, reason: str, latency_seconds: float = 0.0) -> None:
        message = VlmAdvisory()
        message.header.stamp = self._future_source_stamp or self.get_clock().now().to_msg()
        message.valid = False
        message.safe_to_advance = False
        message.hazard_type = "unknown"
        message.hazard_cells = []
        message.safe_cells = []
        message.grid_cols = self._coverage.cols
        message.grid_rows = self._coverage.rows
        message.target_col = -1
        message.target_row = -1
        if self._future_source_pose is not None:
            message.source_x, message.source_y = self._future_source_pose.centre
            message.source_has_heading = self._future_source_pose.heading is not None
            message.source_heading = self._future_source_pose.heading or 0.0
        message.reason = reason
        message.scene_description = self._policy.last_scene_description
        message.objective = self._memory.objective
        message.plan_steps = list(self._memory.plan_steps)
        message.maneuver = "hold"
        message.temporal_context = self._future_temporal_context
        message.stall_detected = self._future_temporal_status.stall_detected
        message.repeated_maneuver = self._future_temporal_status.repeated_maneuver
        message.confidence = 0.0
        message.latency_seconds = latency_seconds
        self._publisher.publish(message)

    def _publish_debug(self, decision, valid: bool) -> None:
        annotated = self._policy.last_annotated_frame
        if annotated is None:
            return
        frame = annotated.copy()
        height, width = frame.shape[:2]
        cell_width = width / self._coverage.cols
        cell_height = height / self._coverage.rows

        def mark(cell, colour, thickness=5):
            col, row = cell
            cv2.rectangle(
                frame,
                (round(col * cell_width), round(row * cell_height)),
                (round((col + 1) * cell_width), round((row + 1) * cell_height)),
                colour,
                thickness,
            )

        for cell in decision.safe_cells:
            mark(cell, (0, 180, 0), 3)
        for cell in decision.hazard_cells:
            mark(cell, (0, 0, 255), 6)
        if decision.target_cell is not None:
            mark(decision.target_cell, (0, 255, 255), 5)
        status = (
            f"{'VALID' if valid else 'INVALID'} safe={decision.safe_to_advance} "
            f"hazard={decision.hazard_type} latency={decision.latency_seconds:.1f}s"
        )
        plan_status = f"maneuver={decision.maneuver} objective={decision.objective[:70]}"
        cv2.rectangle(frame, (0, 0), (width, 70), (0, 0, 0), -1)
        cv2.putText(
            frame, status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2
        )
        cv2.putText(
            frame, plan_status, (10, 57), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1
        )
        message = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self._future_source_stamp or self.get_clock().now().to_msg()
        message.header.frame_id = "vlm_debug"
        self._debug_publisher.publish(message)

    def _tick(self) -> None:
        if self._future is not None:
            if not self._future.done():
                return
            try:
                decision = self._future.result()
            except Exception as exc:
                self._invalid(f"VLM worker failure ({type(exc).__name__}); classic fallback active")
            else:
                self._memory.add_scene(decision.scene_description)
                target_label = (
                    "STOP" if decision.target_cell is None
                    else f"{chr(65 + decision.target_cell[0])}{decision.target_cell[1] + 1}"
                )
                self._memory.update_plan(
                    decision.objective,
                    decision.plan_steps,
                    target_label,
                    decision.reason,
                )
                minimum = float(self.get_parameter("vlm_min_advisory_confidence").value)
                if decision.confidence < minimum:
                    self._invalid(decision.reason, decision.latency_seconds)
                    self._publish_debug(decision, False)
                else:
                    message = VlmAdvisory()
                    message.header.stamp = self._future_source_stamp or self.get_clock().now().to_msg()
                    message.valid = True
                    message.safe_to_advance = decision.safe_to_advance
                    message.hazard_type = decision.hazard_type
                    message.hazard_cells = [
                        f"{chr(65 + col)}{row + 1}" for col, row in decision.hazard_cells
                    ]
                    message.safe_cells = [
                        f"{chr(65 + col)}{row + 1}" for col, row in decision.safe_cells
                    ]
                    message.grid_cols = self._coverage.cols
                    message.grid_rows = self._coverage.rows
                    if decision.target_cell is None:
                        message.target_col = -1
                        message.target_row = -1
                    else:
                        message.target_col, message.target_row = decision.target_cell
                    if self._future_source_pose is not None:
                        message.source_x, message.source_y = self._future_source_pose.centre
                        message.source_has_heading = self._future_source_pose.heading is not None
                        message.source_heading = self._future_source_pose.heading or 0.0
                    message.reason = decision.reason
                    message.scene_description = decision.scene_description
                    message.objective = decision.objective
                    message.plan_steps = list(decision.plan_steps)
                    message.maneuver = decision.maneuver
                    message.temporal_context = self._future_temporal_context
                    message.stall_detected = self._future_temporal_status.stall_detected
                    message.repeated_maneuver = self._future_temporal_status.repeated_maneuver
                    message.confidence = decision.confidence
                    message.raw_response = decision.raw_response
                    message.latency_seconds = decision.latency_seconds
                    self._publisher.publish(message)
                    self._publish_debug(decision, True)
                if self._future_source_frame is not None:
                    self._prior_frames.append(self._future_source_frame)
            self._future = None
        if self._frame is None or self._pose is None:
            # The classic controller remains active while semantic perception
            # has no input, so this is diagnostic rather than a motor STOP.
            self._invalid("Missing image or pose; classic fallback active")
            return
        self._future_source_pose = CorePose(
            tuple(self._pose.centre), self._pose.heading, self._pose.confidence
        )
        self._future_source_stamp = self._frame_stamp
        self._future_source_frame = self._frame.copy()
        self._future_temporal_context, self._future_temporal_status = self._memory.context(
            legal_actions=list(self._legal),
            sonar_context=self._sensor_context(),
        )
        self._future = self._executor.submit(
            self._policy.choose,
            self._frame.copy(),
            self._future_source_pose,
            list(self._legal),
            self._coverage,
            self._sensor_context(),
            self._future_temporal_context,
            [frame.copy() for frame in self._prior_frames],
        )

    def destroy_node(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyVlmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
