from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node

from rover_explorer.coverage import CoverageTracker
from rover_explorer.localize import RoverPose as CorePose
from rover_explorer.motion import Action
from rover_explorer.obstacle import ObstacleGrid
from rover_explorer.policy import (
    BottomCenterKeeper,
    CoverageSweep,
    FrontierGreedy,
    WaypointFollower,
    bounded_pose_recovery_action,
    merge_semantic_hazards,
    semantic_path_cells,
    update_severe_hazard_latch,
)
from rover_explorer_ros2.msg import LegalActions, PolicyDecision, RoverPose, VlmAdvisory

from .common import action_from_string, declare_transform_parameters, transform_from_node

try:
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import ComputePathToPose
except ImportError:
    PoseStamped = ComputePathToPose = None


class PolicyClassicNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_classic_node")
        declare_transform_parameters(self)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("coverage_cols", 6)
        self.declare_parameter("coverage_rows", 4)
        self.declare_parameter("margin_frac", 0.12)
        self.declare_parameter("bottom_center_safety_offset_px", 8.0)
        self.declare_parameter("bottom_center_rover_rear_extent_px", 95.0)
        self.declare_parameter("bottom_center_hold_enter_px", 40.0)
        self.declare_parameter("bottom_center_hold_exit_px", 70.0)
        self.declare_parameter("policy", "sweep")
        self.declare_parameter("planner", "astar_legacy")
        self.declare_parameter("vlm_advisory_timeout_seconds", 60.0)
        self.declare_parameter("vlm_use_semantic_waypoint", True)
        self.declare_parameter("vlm_maneuver_timeout_seconds", 3.0)
        self.declare_parameter("vlm_plan_progress_timeout_seconds", 10.0)
        self.declare_parameter("vlm_plan_progress_minimum_px", 10.0)
        self.declare_parameter("vlm_failed_target_cooldown_seconds", 45.0)
        self.declare_parameter("vlm_scene_timeout_seconds", 45.0)
        self.declare_parameter("vlm_hazard_lookahead_pulses", 2.0)
        self.declare_parameter("vlm_hazard_clear_confirmations", 2)
        self.declare_parameter("vlm_severe_hazard_clear_confirmations", 3)
        self.declare_parameter("pose_recovery_retry_seconds", 3.0)
        height = int(self.get_parameter("camera_height").value)
        width = int(self.get_parameter("camera_width").value)
        self._frame = np.zeros((height, width, 3), np.uint8)
        self._coverage = CoverageTracker(
            self._frame.shape,
            int(self.get_parameter("coverage_cols").value),
            int(self.get_parameter("coverage_rows").value),
        )
        transform = transform_from_node(self)
        self._transform = transform
        self._waypoint_follower = WaypointFollower(transform)
        policy = str(self.get_parameter("policy").value)
        if policy == "frontier":
            self._policy = FrontierGreedy(transform)
        elif policy == "bottom_center":
            self._policy = BottomCenterKeeper(
                transform,
                margin_frac=float(self.get_parameter("margin_frac").value),
                safety_offset_px=float(
                    self.get_parameter("bottom_center_safety_offset_px").value
                ),
                rover_rear_extent_px=float(
                    self.get_parameter("bottom_center_rover_rear_extent_px").value
                ),
                hold_enter_radius_px=float(
                    self.get_parameter("bottom_center_hold_enter_px").value
                ),
                hold_exit_radius_px=float(
                    self.get_parameter("bottom_center_hold_exit_px").value
                ),
            )
        else:
            self._policy = CoverageSweep(
                transform, float(self.get_parameter("margin_frac").value)
            )
        self._pose: CorePose | None = None
        self._legal = [Action.STOP]
        self._legal_reason = ""
        self._pose_recovery_next = 0.0
        self._occupancy: OccupancyGrid | None = None
        self._vlm_target: tuple[int, int] | None = None
        self._vlm_target_received = 0.0
        self._vlm_target_confidence = 0.0
        self._vlm_target_reason = ""
        self._vlm_objective = "Awaiting the first VLM navigation objective."
        self._vlm_plan_steps: tuple[str, ...] = ()
        self._vlm_maneuver = "hold"
        self._vlm_waiting_replan = policy == "vlm"
        self._vlm_target_best_distance = float("inf")
        self._vlm_target_last_progress = time.monotonic()
        self._vlm_failed_targets: dict[tuple[int, int], float] = {}
        self._vlm_grid_cols = int(self.get_parameter("coverage_cols").value)
        self._vlm_grid_rows = int(self.get_parameter("coverage_rows").value)
        self._vlm_hazard_type = "unknown"
        self._vlm_hazard_cells: set[tuple[int, int]] = set()
        self._vlm_hazard_misses: dict[tuple[int, int], int] = {}
        self._vlm_safe_cells: set[tuple[int, int]] = set()
        self._vlm_scene_received = 0.0
        self._vlm_scene_source_age_at_receive = float("inf")
        self._vlm_scene_valid = False
        self._vlm_globally_unsafe = True
        self._vlm_severe_hazard_latched = False
        self._vlm_severe_clear_count = 0
        self._vlm_last_error = "No semantic scene received yet."
        self._nav2_action = (
            ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
            if ComputePathToPose is not None else None
        )
        self._nav2_pending = False
        self._nav2_choice = (Action.STOP, "Waiting for a Nav2 path.", 0.0)
        self._publisher = self.create_publisher(PolicyDecision, "/rover/policy/classic_decision", 10)
        self.create_subscription(RoverPose, "/rover/pose", self._on_pose, 10)
        self.create_subscription(LegalActions, "/rover/legal_actions", self._on_legal, 10)
        self.create_subscription(OccupancyGrid, "/rover/occupancy_grid", self._on_occupancy, 10)
        self.create_subscription(OccupancyGrid, "/rover/coverage_map", self._on_coverage, 10)
        self.create_subscription(VlmAdvisory, "/rover/vlm/advisory", self._on_vlm_advisory, 10)
        self.create_timer(0.2, self._choose)

    def _on_pose(self, message: RoverPose) -> None:
        self._pose = CorePose(
            (message.centre.x, message.centre.y),
            message.heading if message.has_heading else None,
            message.confidence,
        )

    def _on_legal(self, message: LegalActions) -> None:
        self._legal = [action_from_string(value) for value in message.actions] or [Action.STOP]
        self._legal_reason = message.reason
        if "pose stale/lost" not in self._legal_reason:
            self._pose_recovery_next = 0.0

    def _on_occupancy(self, message: OccupancyGrid) -> None:
        self._occupancy = message

    def _on_coverage(self, message: OccupancyGrid) -> None:
        for row in range(min(self._coverage.rows, message.info.height)):
            for col in range(min(self._coverage.cols, message.info.width)):
                if message.data[row * message.info.width + col] > 0:
                    self._coverage.visited.add((col, row))

    def _on_vlm_advisory(self, message: VlmAdvisory) -> None:
        if not message.valid:
            # An asynchronous timeout must not erase the last usable static
            # scene map. Its independent TTL will expire it safely.
            self._vlm_last_error = message.reason
            return
        grid_cols = max(1, int(message.grid_cols))
        grid_rows = max(1, int(message.grid_rows))
        if (grid_cols, grid_rows) != (self._vlm_grid_cols, self._vlm_grid_rows):
            self._vlm_hazard_cells.clear()
            self._vlm_hazard_misses.clear()
        self._vlm_grid_cols, self._vlm_grid_rows = grid_cols, grid_rows
        incoming_hazards = self._parse_cell_labels(message.hazard_cells)
        confirmations = max(
            1, int(self.get_parameter("vlm_hazard_clear_confirmations").value)
        )
        self._vlm_hazard_cells, self._vlm_hazard_misses = merge_semantic_hazards(
            self._vlm_hazard_cells,
            self._vlm_hazard_misses,
            incoming_hazards,
            confirmations,
        )
        self._vlm_safe_cells = self._parse_cell_labels(message.safe_cells)
        self._vlm_hazard_type = message.hazard_type or "unknown"
        self._vlm_severe_hazard_latched, self._vlm_severe_clear_count = (
            update_severe_hazard_latch(
                self._vlm_severe_hazard_latched,
                self._vlm_severe_clear_count,
                self._vlm_hazard_type,
                bool(message.safe_to_advance),
                int(self.get_parameter("vlm_severe_hazard_clear_confirmations").value),
            )
        )
        self._vlm_globally_unsafe = bool(
            (self._vlm_severe_hazard_latched and not self._vlm_hazard_cells)
            or (
                not message.safe_to_advance
                and self._vlm_hazard_type != "none"
                and not incoming_hazards
            )
        )
        self._vlm_scene_received = time.monotonic()
        self._vlm_scene_source_age_at_receive = max(0.0, float(message.latency_seconds))
        self._vlm_scene_valid = True
        self._vlm_last_error = ""
        self._vlm_objective = message.objective or self._vlm_objective
        self._vlm_plan_steps = tuple(message.plan_steps)
        self._vlm_maneuver = message.maneuver or "hold"
        now = time.monotonic()
        self._vlm_target_received = now
        self._vlm_target_confidence = float(message.confidence)
        self._vlm_target_reason = message.reason
        if not bool(self.get_parameter("vlm_use_semantic_waypoint").value):
            # The VLM is a semantic safety observer, not a geometric steering
            # controller. Small models choose unstable grid targets and cause
            # repeated alignment turns.
            self._vlm_target = None
            return
        cell = (int(message.target_col), int(message.target_row))
        if not (
            0 <= cell[0] < self._vlm_grid_cols and 0 <= cell[1] < self._vlm_grid_rows
        ) or cell in self._vlm_hazard_cells:
            self._vlm_target = None
            self._vlm_waiting_replan = True
            return
        cooldown = float(self.get_parameter("vlm_failed_target_cooldown_seconds").value)
        self._vlm_failed_targets = {
            target: failed_at for target, failed_at in self._vlm_failed_targets.items()
            if now - failed_at <= cooldown
        }
        if cell in self._vlm_failed_targets:
            self._vlm_target = None
            self._vlm_waiting_replan = True
            self._vlm_last_error = (
                f"VLM repeated recently failed target {chr(65 + cell[0])}{cell[1] + 1}; "
                "waiting for a different plan."
            )
            return
        if cell != self._vlm_target:
            self._vlm_target_best_distance = float("inf")
            self._vlm_target_last_progress = now
        self._vlm_target = cell
        self._vlm_waiting_replan = False

    def _parse_cell_labels(self, labels) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for raw in labels:
            label = str(raw).strip().upper()
            if len(label) < 2 or not label[1:].isdigit():
                continue
            cell = (ord(label[0]) - 65, int(label[1:]) - 1)
            if 0 <= cell[0] < self._vlm_grid_cols and 0 <= cell[1] < self._vlm_grid_rows:
                cells.add(cell)
        return cells

    def _semantic_cell_for(self, point: tuple[float, float]) -> tuple[int, int]:
        height, width = self._frame.shape[:2]
        return (
            min(self._vlm_grid_cols - 1, max(0, int(point[0] * self._vlm_grid_cols / width))),
            min(self._vlm_grid_rows - 1, max(0, int(point[1] * self._vlm_grid_rows / height))),
        )

    def _semantic_motion_clear(self, action: Action) -> tuple[bool, str]:
        if not self._vlm_scene_valid:
            return False, self._vlm_last_error or "No valid semantic scene."
        age = self._vlm_scene_source_age_at_receive + max(
            0.0, time.monotonic() - self._vlm_scene_received
        )
        maximum_age = float(self.get_parameter("vlm_scene_timeout_seconds").value)
        if age > maximum_age:
            return False, f"Semantic scene expired ({age:.1f}s > {maximum_age:.1f}s)."
        if self._vlm_globally_unsafe:
            return False, f"Unlocalized semantic hazard: {self._vlm_hazard_type}."
        path_cells = semantic_path_cells(
            self._pose,
            action,
            self._transform,
            self._frame.shape,
            self._vlm_grid_cols,
            self._vlm_grid_rows,
            lookahead_pulses=float(
                self.get_parameter("vlm_hazard_lookahead_pulses").value
            ),
        )
        if (-1, -1) in path_cells:
            return False, "Projected semantic corridor leaves the camera image."
        blocked = path_cells & self._vlm_hazard_cells
        if blocked:
            labels = ",".join(
                f"{chr(65 + col)}{row + 1}" for col, row in sorted(blocked)
            )
            return False, f"Projected corridor intersects semantic hazard {labels}."
        return True, f"Semantic scene age={age:.1f}s; corridor cells are clear."

    def _vlm_advisory_action(self) -> tuple[Action, str, float] | None:
        if self._pose is None:
            return None
        age = time.monotonic() - self._vlm_target_received
        if age > float(self.get_parameter("vlm_advisory_timeout_seconds").value):
            self._vlm_target = None
            self._vlm_waiting_replan = True
            return Action.STOP, "VLM plan expired; waiting for replanning.", 1.0

        maneuver_actions = {
            "advance": Action.FORWARD,
            "reverse": Action.BACKWARD,
            "turn_left": Action.TURN_LEFT,
            "turn_right": Action.TURN_RIGHT,
            "hold": Action.STOP,
            "replan": Action.STOP,
        }
        maneuver_timeout = float(self.get_parameter("vlm_maneuver_timeout_seconds").value)
        if self._vlm_maneuver in maneuver_actions and age <= maneuver_timeout:
            requested = maneuver_actions[self._vlm_maneuver]
            objective = f"Objective: {self._vlm_objective}"
            if requested in self._legal:
                return (
                    requested,
                    f"VLM bounded maneuver={self._vlm_maneuver}. {objective} "
                    f"Plan: {' -> '.join(self._vlm_plan_steps)}",
                    self._vlm_target_confidence,
                )
            return Action.STOP, f"VLM maneuver {self._vlm_maneuver} is not currently legal.", 1.0

        if self._vlm_target is None:
            if self._vlm_waiting_replan:
                return Action.STOP, self._vlm_last_error or "Waiting for a new VLM objective.", 1.0
            return None
        if self._semantic_cell_for(self._pose.centre) == self._vlm_target:
            self._vlm_target = None
            self._vlm_waiting_replan = True
            return Action.STOP, "VLM target reached; waiting for the next objective.", 1.0

        col, row = self._vlm_target
        target = (
            (col + .5) * self._frame.shape[1] / self._vlm_grid_cols,
            (row + .5) * self._frame.shape[0] / self._vlm_grid_rows,
        )
        distance = math.dist(self._pose.centre, target)
        minimum_progress = float(self.get_parameter("vlm_plan_progress_minimum_px").value)
        now = time.monotonic()
        if distance + minimum_progress < self._vlm_target_best_distance:
            self._vlm_target_best_distance = distance
            self._vlm_target_last_progress = now
        progress_timeout = float(self.get_parameter("vlm_plan_progress_timeout_seconds").value)
        if now - self._vlm_target_last_progress > progress_timeout:
            failed = self._vlm_target
            self._vlm_failed_targets[failed] = now
            self._vlm_target = None
            self._vlm_waiting_replan = True
            label = f"{chr(65 + failed[0])}{failed[1] + 1}"
            self._vlm_last_error = (
                f"No measurable progress toward {label} for {progress_timeout:.1f}s; "
                "target marked failed and temporal replanning required."
            )
            return Action.STOP, self._vlm_last_error, 1.0
        waypoint = target
        if self._occupancy is not None and self._occupancy.info.width and self._occupancy.info.height:
            width, height = self._occupancy.info.width, self._occupancy.info.height
            grid = ObstacleGrid(self._frame.shape, width, height)
            blocked = {
                (index % width, index // width)
                for index, value in enumerate(self._occupancy.data)
                if value >= 50
            }
            for hazard_cell in self._vlm_hazard_cells:
                centre = (
                    (hazard_cell[0] + .5) * self._frame.shape[1] / self._vlm_grid_cols,
                    (hazard_cell[1] + .5) * self._frame.shape[0] / self._vlm_grid_rows,
                )
                blocked.add(grid.cell_for(centre))
            start, goal = grid.cell_for(self._pose.centre), grid.cell_for(target)
            path = None if goal in blocked else grid.astar(start, goal, blocked)
            if not path:
                self.get_logger().warning(
                    f"Rejecting blocked VLM waypoint {chr(65 + col)}{row + 1}"
                )
                self._vlm_failed_targets[self._vlm_target] = time.monotonic()
                self._vlm_target = None
                self._vlm_waiting_replan = True
                return Action.STOP, "VLM waypoint has no path around mapped hazards; replanning.", 1.0
            if len(path) > 1:
                waypoint = grid.centre(path[1])

        label = f"{chr(65 + col)}{row + 1}"
        decision = self._waypoint_follower.choose(self._pose, waypoint, self._legal, label)
        reason = f"{decision.reason} VLM: {self._vlm_target_reason}"
        return decision.action, reason, self._vlm_target_confidence

    def _legacy_astar_action(self) -> tuple[Action, str] | None:
        if self._pose is None or self._occupancy is None or self._pose.heading is None:
            return None
        width, height = self._occupancy.info.width, self._occupancy.info.height
        if not width or not height:
            return None
        grid = ObstacleGrid(self._frame.shape, width, height)
        blocked = {
            (index % width, index // width)
            for index, value in enumerate(self._occupancy.data)
            if value >= 50
        }
        start = grid.cell_for(self._pose.centre)
        free = [(c, r) for r in range(height) for c in range(width) if (c, r) not in blocked]
        if not free:
            return Action.STOP, "No free occupancy cells."
        goal = max(free, key=lambda cell: math.dist(cell, start))
        path = grid.astar(start, goal, blocked)
        if not path or len(path) < 2:
            return Action.STOP, "No safe A* route."
        tx, ty = grid.centre(path[1])
        error = (math.atan2(ty - self._pose.centre[1], tx - self._pose.centre[0]) - self._pose.heading + math.pi) % (2 * math.pi) - math.pi
        if abs(error) > math.radians(12):
            return (Action.TURN_LEFT if error > 0 else Action.TURN_RIGHT), "A* path alignment."
        return Action.FORWARD, "Following legacy A* path."

    def _request_nav2_path(self) -> None:
        if self._nav2_pending or self._pose is None or self._occupancy is None:
            return
        if self._nav2_action is None:
            self._nav2_choice = (Action.STOP, "nav2_msgs is not installed.", 0.0)
            return
        if not self._nav2_action.server_is_ready():
            self._nav2_choice = (Action.STOP, "Nav2 planner server unavailable.", 0.0)
            return
        grid = self._occupancy
        width, height = int(grid.info.width), int(grid.info.height)
        if not width or not height:
            return
        start_col = min(width - 1, max(0, int(self._pose.centre[0] * width / self._frame.shape[1])))
        start_row = min(height - 1, max(0, int(self._pose.centre[1] * height / self._frame.shape[0])))
        free = [
            (index % width, index // width)
            for index, value in enumerate(grid.data)
            if value >= 0 and value < 50
        ]
        if not free:
            self._nav2_choice = (Action.STOP, "Nav2 map contains no free goal.", 0.0)
            return
        goal_col, goal_row = max(free, key=lambda cell: math.dist(cell, (start_col, start_row)))
        request = ComputePathToPose.Goal()
        request.use_start = True
        request.start = self._grid_pose(start_col, start_row, grid)
        request.goal = self._grid_pose(goal_col, goal_row, grid)
        self._nav2_pending = True
        future = self._nav2_action.send_goal_async(request)
        future.add_done_callback(self._nav2_goal_response)

    def _grid_pose(self, col: int, row: int, grid: OccupancyGrid) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = grid.header.frame_id or "map"
        pose.pose.position.x = grid.info.origin.position.x + (col + 0.5) * grid.info.resolution
        pose.pose.position.y = grid.info.origin.position.y + (row + 0.5) * grid.info.resolution
        pose.pose.orientation.w = 1.0
        return pose

    def _nav2_goal_response(self, future) -> None:
        try:
            handle = future.result()
            if not handle.accepted:
                raise RuntimeError("path goal rejected")
            result_future = handle.get_result_async()
            result_future.add_done_callback(self._nav2_result)
        except Exception as exc:
            self._nav2_pending = False
            self._nav2_choice = (Action.STOP, f"Nav2 request failed: {exc}", 0.0)

    def _nav2_result(self, future) -> None:
        self._nav2_pending = False
        try:
            poses = future.result().result.path.poses
            if len(poses) < 2 or self._pose is None or self._pose.heading is None:
                raise RuntimeError("empty path or heading unavailable")
            first, second = poses[0].pose.position, poses[1].pose.position
            # Image y grows downward whereas ROS map y normally grows upward.
            desired = math.atan2(-(second.y - first.y), second.x - first.x)
            error = (desired - self._pose.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(error) > math.radians(12):
                action = Action.TURN_LEFT if error > 0 else Action.TURN_RIGHT
            else:
                action = Action.FORWARD
            self._nav2_choice = (action, "Following Nav2-computed path.", 1.0)
        except Exception as exc:
            self._nav2_choice = (Action.STOP, f"Invalid Nav2 path: {exc}", 0.0)

    def _choose(self) -> None:
        policy = str(self.get_parameter("policy").value)
        planner = str(self.get_parameter("planner").value)
        recovery_action, self._pose_recovery_next = bounded_pose_recovery_action(
            self._legal_reason,
            self._legal,
            time.monotonic(),
            self._pose_recovery_next,
            float(self.get_parameter("pose_recovery_retry_seconds").value),
        )
        if recovery_action is not None:
            action = recovery_action
            reason = "Pose stale/lost: blind recovery motion disabled."
            confidence = 1.0
        elif policy == "vlm":
            advisory = self._vlm_advisory_action()
            if advisory is None:
                decision = self._policy.choose(self._frame, self._pose, self._legal, self._coverage)
                action = decision.action
                reason = f"Hybrid classic fallback: {decision.reason}"
                confidence = decision.confidence
            else:
                action, reason, confidence = advisory
        elif planner == "nav2":
            self._request_nav2_path()
            action, reason, confidence = self._nav2_choice
        elif policy == "obstacle_sweep":
            result = self._legacy_astar_action()
            action, reason = result if result is not None else (Action.STOP, "Pose/map unavailable for A*.")
            confidence = 1.0
        else:
            decision = self._policy.choose(self._frame, self._pose, self._legal, self._coverage)
            action, reason, confidence = decision.action, decision.reason, decision.confidence
        if action not in self._legal:
            action, reason, confidence = Action.STOP, "Classic policy action vetoed by latest guard.", 0.0
        if policy == "vlm" and action in {
            Action.FORWARD, Action.BACKWARD, Action.ARC_LEFT, Action.ARC_RIGHT
        }:
            clear, semantic_reason = self._semantic_motion_clear(action)
            if not clear:
                action = Action.STOP
                reason = f"VLM scene gate: {semantic_reason}"
                confidence = 0.0
            else:
                reason = f"{reason} {semantic_reason}"
        message = PolicyDecision()
        message.header.stamp = self.get_clock().now().to_msg()
        message.action = action.value
        message.reason = reason
        message.confidence = confidence
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyClassicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
