import json

import numpy as np
import pytest
import requests

from rover_explorer.calibrate import BodyToImage
from rover_explorer.coverage import CoverageTracker
from rover_explorer.localize import RoverPose
from rover_explorer.motion import Action
from rover_explorer.policy import (
    BottomCenterKeeper,
    CoverageSweep,
    VlmExplorer,
    WaypointFollower,
    bounded_pose_recovery_action,
    merge_semantic_hazards,
    scene_report_supports_clear_ground,
    scene_report_supports_hazard,
    semantic_clearance_is_valid,
    semantic_path_cells,
    update_severe_hazard_latch,
)


class Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self.content}}


@pytest.mark.parametrize("failure", ["timeout", "json", "range"])
def test_vlm_failures_fall_back_to_stop(failure):
    def requester(*args, **kwargs):
        if failure == "timeout":
            raise requests.Timeout("injected")
        if failure == "json":
            return Response("not json")
        return Response(json.dumps({"choice": 999, "reason": "bad", "confidence": 1}))

    transform = BodyToImage(30, .4, (1, 0))
    policy = VlmExplorer(transform, requester=requester)
    frame = np.zeros((480, 640, 3), np.uint8)
    pose = RoverPose((320, 240), 0, 1)
    coverage = CoverageTracker(frame.shape)
    decision = policy.choose(frame, pose, [Action.FORWARD, Action.TURN_LEFT, Action.STOP], coverage)
    assert decision.action == Action.STOP
    assert policy.fallback_count == 1


def test_vlm_arrow_schema_is_limited_to_visible_legal_choices():
    captured = {}

    def requester(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response(json.dumps({"choice": 2, "reason": "open space", "confidence": .8}))

    transform = BodyToImage(30, .4, (1, 0))
    policy = VlmExplorer(transform, requester=requester)
    frame = np.zeros((480, 640, 3), np.uint8)
    pose = RoverPose((320, 240), 0, 1)
    coverage = CoverageTracker(frame.shape)
    allowed = [Action.FORWARD, Action.TURN_LEFT, Action.STOP]

    decision = policy.choose(frame, pose, allowed, coverage)

    assert decision.action == Action.TURN_LEFT
    assert captured["format"]["properties"]["choice"]["enum"] == [1, 2, 3]
    assert "1=forward, 2=turn_left, 3=stop" in captured["messages"][1]["content"]


def test_vlm_receives_ultrasonic_scene_context():
    captured = {}

    def requester(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response(json.dumps({"choice": 1, "reason": "clear", "confidence": .8}))

    frame = np.zeros((480, 640, 3), np.uint8)
    pose = RoverPose((320, 240), 0, 1)
    coverage = CoverageTracker(frame.shape)
    policy = VlmExplorer(BodyToImage(30, -.48, (1, 0)), requester=requester)

    policy.choose(
        frame,
        pose,
        [Action.TURN_LEFT, Action.STOP],
        coverage,
        "front_blocked=True; front=0.18 m; left=0.80 m; right=0.25 m",
    )

    prompt = captured["messages"][1]["content"]
    assert "front_blocked=True" in prompt
    assert "left=0.80 m" in prompt


def test_vlm_grid_mode_returns_semantic_target_and_keep_alive():
    captured = {}

    def requester(*args, **kwargs):
        if "format" not in kwargs["json"]:
            return Response("A room with a flat tiled floor and no stairs or drop-off.")
        captured.update(kwargs["json"])
        return Response(json.dumps({
            "cell": "B2", "safe_to_advance": True, "hazard_type": "none",
            "hazard_cells": [], "safe_cells": ["B2"],
            "objective": "Explore B2 safely", "plan_steps": ["Advance to B2"],
            "maneuver": "advance",
            "reason": "open floor", "confidence": .9,
        }))

    frame = np.zeros((480, 640, 3), np.uint8)
    pose = RoverPose((320, 240), 0, 1)
    coverage = CoverageTracker(frame.shape)
    policy = VlmExplorer(
        BodyToImage(30, -.48, (1, 0)),
        annotation_style="grid",
        requester=requester,
        keep_alive="30m",
    )

    decision = policy.choose(frame, pose, list(Action), coverage)

    assert decision.target_cell == (1, 1)
    assert decision.safe_to_advance
    assert captured["keep_alive"] == "30m"
    assert policy.last_annotated_frame.shape == frame.shape
    assert len(captured["messages"][1]["images"]) == 1
    assert "flat tiled floor" in captured["messages"][1]["content"]
    assert decision.scene_description.startswith("A room with a flat tiled floor")


def test_vlm_grid_mode_stops_for_stairs_and_marks_hazard_cells():
    def requester(*args, **kwargs):
        if "format" not in kwargs["json"]:
            return Response("A staircase with multiple visible wooden steps is ahead.")
        return Response(json.dumps({
            "cell": "STOP", "safe_to_advance": False, "hazard_type": "stairs",
            "hazard_cells": ["A1", "B1"], "safe_cells": ["C3"],
            "objective": "Avoid stairs", "plan_steps": ["Hold", "Replan around A1-B1"],
            "maneuver": "hold",
            "reason": "stairs in the immediate corridor", "confidence": .95,
        }))

    frame = np.zeros((480, 640, 3), np.uint8)
    pose = RoverPose((320, 240), 0, 1)
    policy = VlmExplorer(
        BodyToImage(30, -.48, (1, 0)), annotation_style="grid", requester=requester
    )
    decision = policy.choose(frame, pose, list(Action), CoverageTracker(frame.shape))

    assert decision.action == Action.STOP
    assert decision.target_cell is None
    assert not decision.safe_to_advance
    assert decision.hazard_type == "stairs"
    assert decision.hazard_cells == ((0, 0), (1, 0))


def test_vlm_grid_mode_stops_when_raw_scene_observer_fails():
    def requester(*args, **kwargs):
        raise requests.Timeout("observer timeout")

    frame = np.zeros((480, 640, 3), np.uint8)
    policy = VlmExplorer(
        BodyToImage(30, -.48, (1, 0)), annotation_style="grid", requester=requester
    )
    decision = policy.choose(
        frame,
        RoverPose((320, 240), 0, 1),
        list(Action),
        CoverageTracker(frame.shape),
    )

    assert decision.action == Action.STOP
    assert "scene observer failure" in decision.reason


def test_raw_scene_observer_vetoes_hallucinated_planner_drop_off():
    def requester(*args, **kwargs):
        if "format" not in kwargs["json"]:
            return Response(
                "The tiled floor is flat and continuous. There are no stairs or a drop-off."
            )
        return Response(json.dumps({
            "cell": "STOP", "safe_to_advance": False, "hazard_type": "drop_off",
            "hazard_cells": ["F4"], "safe_cells": ["F3"],
            "objective": "Explore safely", "plan_steps": ["Verify F4", "Choose safe cell"],
            "maneuver": "replan",
            "reason": "invented red warning line", "confidence": .95,
        }))

    frame = np.zeros((480, 640, 3), np.uint8)
    policy = VlmExplorer(
        BodyToImage(30, -.48, (1, 0)), annotation_style="grid", requester=requester
    )
    decision = policy.choose(
        frame,
        RoverPose((320, 240), 0, 1),
        list(Action),
        CoverageTracker(frame.shape),
    )

    assert decision.hazard_type == "none"
    assert decision.hazard_cells == ()
    assert decision.safe_to_advance
    assert "Rejected unsupported drop_off" in decision.reason
    assert not scene_report_supports_hazard(decision.scene_description, "drop_off")


def test_raw_scene_without_visible_floor_is_fail_closed():
    def requester(*args, **kwargs):
        if "format" not in kwargs["json"]:
            return Response(
                "The floor structure is not clearly visible, but it seems to be flat."
            )
        return Response(json.dumps({
            "cell": "B2", "safe_to_advance": True, "hazard_type": "none",
            "hazard_cells": [], "safe_cells": ["B2"],
            "objective": "Find visible floor", "plan_steps": ["Hold for visibility"],
            "maneuver": "hold",
            "reason": "probably clear", "confidence": .9,
        }))

    frame = np.zeros((480, 640, 3), np.uint8)
    policy = VlmExplorer(
        BodyToImage(30, -.48, (1, 0)), annotation_style="grid", requester=requester
    )
    decision = policy.choose(
        frame, RoverPose((320, 240), 0, 1), list(Action), CoverageTracker(frame.shape)
    )

    assert decision.action == Action.STOP
    assert decision.hazard_type == "unknown"
    assert not decision.safe_to_advance
    assert "did not positively confirm visible clear ground" in decision.reason
    assert not scene_report_supports_clear_ground(decision.scene_description)


def test_pose_loss_recovery_never_moves_blindly():
    allowed = [Action.BACKWARD, Action.STOP]
    action, next_allowed = bounded_pose_recovery_action(
        "pose stale/lost; conservative recovery", allowed, 10.0, 0.0, 3.0
    )
    assert action == Action.STOP
    assert next_allowed == 13.0

    action, unchanged = bounded_pose_recovery_action(
        "pose stale/lost; conservative recovery", allowed, 10.2, next_allowed, 3.0
    )
    assert action == Action.STOP
    assert unchanged == 13.2

    action, reset = bounded_pose_recovery_action(
        "fresh guard calculation", allowed, 10.4, next_allowed, 3.0
    )
    assert action is None
    assert reset == 0.0


def test_severe_terrain_hazard_latches_until_three_clean_reports():
    latched, clean = update_severe_hazard_latch(False, 0, "drop_off", False, 3)
    assert latched and clean == 0

    latched, clean = update_severe_hazard_latch(latched, clean, "none", True, 3)
    assert latched and clean == 1
    latched, clean = update_severe_hazard_latch(latched, clean, "unknown", False, 3)
    assert latched and clean == 0
    latched, clean = update_severe_hazard_latch(latched, clean, "none", True, 3)
    latched, clean = update_severe_hazard_latch(latched, clean, "none", True, 3)
    latched, clean = update_severe_hazard_latch(latched, clean, "none", True, 3)
    assert not latched and clean == 0


def test_semantic_clearance_expires_after_time_motion_or_rotation():
    source = RoverPose((100, 100), 0.0, 1.0)
    common = dict(
        safe_to_advance=True,
        maximum_age_seconds=4.0,
        maximum_motion_px=55.0,
        maximum_heading_change_radians=np.deg2rad(35),
    )

    assert semantic_clearance_is_valid(source, source, source_age_seconds=1.0, **common)
    assert not semantic_clearance_is_valid(source, source, source_age_seconds=5.0, **common)
    assert not semantic_clearance_is_valid(
        RoverPose((160, 100), 0.0, 1.0), source, source_age_seconds=1.0, **common
    )
    assert not semantic_clearance_is_valid(
        RoverPose((100, 100), np.deg2rad(40), 1.0),
        source,
        source_age_seconds=1.0,
        **common,
    )
    assert not semantic_clearance_is_valid(
        source, source, source_age_seconds=1.0, **{**common, "safe_to_advance": False}
    )


def test_semantic_path_projects_current_motion_into_camera_fixed_grid():
    transform = BodyToImage(35, -.48, (1, 0))
    pose = RoverPose((300, 240), 0.0, 1.0)

    cells = semantic_path_cells(
        pose, Action.FORWARD, transform, (480, 640, 3), 8, 6, lookahead_pulses=2
    )

    assert cells == {(3, 3), (4, 3)}
    assert semantic_path_cells(
        RoverPose((620, 240), 0.0, 1.0),
        Action.FORWARD,
        transform,
        (480, 640, 3),
        8,
        6,
        lookahead_pulses=2,
    ) == {(-1, -1)}
    assert semantic_path_cells(
        pose, Action.TURN_LEFT, transform, (480, 640, 3), 8, 6
    ) == set()


def test_bottom_center_keeper_advances_toward_recomputed_safe_waypoint():
    frame = np.zeros((480, 640, 3), np.uint8)
    transform = BodyToImage(35, -.48, (1, 0))
    policy = BottomCenterKeeper(transform)
    coverage = CoverageTracker(frame.shape, 6, 4)

    decision = policy.choose(
        frame,
        RoverPose((320, 300), np.pi / 2, 1.0),
        [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STOP],
        coverage,
    )

    assert decision.action == Action.FORWARD
    assert decision.objective == "stay_bottom_center"
    assert decision.target_cell == (3, 2)
    assert decision.plan_steps == (
        "Advance one calibrated pulse.", "Recompute from the next pose."
    )
    assert "(320, 342)" in decision.reason


def test_bottom_center_keeper_rejects_stale_pose_and_transform():
    frame = np.zeros((480, 640, 3), np.uint8)
    transform = BodyToImage(35, -.48, (1, 0))
    policy = BottomCenterKeeper(transform, maximum_data_age_seconds=1.0)
    coverage = CoverageTracker(frame.shape)
    pose = RoverPose((320, 300), np.pi / 2, 1.0)

    stale_pose = policy.choose(
        frame, pose, list(Action), coverage, localization_age_seconds=1.1
    )
    stale_transform = policy.choose(
        frame, pose, list(Action), coverage, transform_age_seconds=1.1
    )

    assert stale_pose.action == Action.STOP
    assert stale_transform.action == Action.STOP
    assert stale_pose.hazard_type == "unknown"
    assert stale_transform.objective == "stay_bottom_center"


def test_bottom_center_keeper_turn_hysteresis_avoids_wrap_oscillation():
    frame = np.zeros((480, 640, 3), np.uint8)
    transform = BodyToImage(35, -.48, (1, 0))
    policy = BottomCenterKeeper(transform)
    coverage = CoverageTracker(frame.shape)
    desired = np.pi / 2
    allowed = [Action.TURN_LEFT, Action.TURN_RIGHT, Action.STOP]

    before_wrap = policy.choose(
        frame,
        RoverPose((320, 300), desired - np.deg2rad(179), 1.0),
        allowed,
        coverage,
    )
    after_wrap = policy.choose(
        frame,
        RoverPose((320, 300), desired + np.deg2rad(179), 1.0),
        allowed,
        coverage,
    )

    assert before_wrap.action in {Action.TURN_LEFT, Action.TURN_RIGHT}
    assert after_wrap.action == before_wrap.action


def test_bottom_center_keeper_reports_guarded_forward_as_obstacle():
    frame = np.zeros((480, 640, 3), np.uint8)
    policy = BottomCenterKeeper(BodyToImage(35, -.48, (1, 0)))

    decision = policy.choose(
        frame,
        RoverPose((320, 300), np.pi / 2, 1.0),
        [Action.TURN_LEFT, Action.STOP],
        CoverageTracker(frame.shape),
    )

    assert decision.action == Action.TURN_LEFT
    assert decision.hazard_type == "obstacle"
    assert not decision.safe_to_advance


def test_bottom_center_hold_precedes_heading_alignment_and_has_hysteresis():
    frame = np.zeros((480, 640, 3), np.uint8)
    transform = BodyToImage(35, -.48, (1, 0))
    policy = BottomCenterKeeper(transform, hold_enter_radius_px=30, hold_exit_radius_px=60)
    coverage = CoverageTracker(frame.shape)
    allowed = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STOP]

    entered = policy.choose(
        frame, RoverPose((320, 342), -2.4, 1.0), allowed, coverage
    )
    noisy_but_held = policy.choose(
        frame, RoverPose((320, 297), 0.1, 1.0), allowed, coverage
    )
    outside = policy.choose(
        frame, RoverPose((320, 281), np.pi / 2, 1.0), allowed, coverage
    )

    assert entered.action == Action.STOP
    assert "Entered bottom-centre hold radius" in entered.reason
    assert noisy_but_held.action == Action.STOP
    assert "inside exit radius" in noisy_but_held.reason
    assert outside.action == Action.FORWARD


def test_bottom_center_recomputes_target_after_frame_and_pose_change():
    transform = BodyToImage(35, -.48, (1, 0))
    policy = BottomCenterKeeper(transform)
    first = np.zeros((720, 1280, 3), np.uint8)
    second = np.zeros((800, 1280, 3), np.uint8)
    allowed = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STOP]

    held = policy.choose(
        first, RoverPose((640, 582), 0.0, 1.0), allowed, CoverageTracker(first.shape)
    )
    updated = policy.choose(
        second,
        RoverPose((640, 582), np.pi / 2, 1.0),
        allowed,
        CoverageTracker(second.shape),
    )

    assert held.action == Action.STOP
    assert updated.action == Action.FORWARD
    assert "(640, 662)" in updated.reason


def test_semantic_hazards_need_two_clean_maps_before_removal():
    hazards, misses = merge_semantic_hazards(set(), {}, {(5, 3)}, 2)
    assert hazards == {(5, 3)}
    assert misses == {(5, 3): 0}

    hazards, misses = merge_semantic_hazards(hazards, misses, set(), 2)
    assert hazards == {(5, 3)}
    assert misses == {(5, 3): 1}

    hazards, misses = merge_semantic_hazards(hazards, misses, set(), 2)
    assert hazards == set()
    assert misses == {}


def test_waypoint_follower_turns_then_drives_without_waiting_for_vlm():
    follower = WaypointFollower(BodyToImage(30, -.48, (1, 0)))
    allowed = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STOP]

    turn = follower.choose(RoverPose((100, 100), 0, 1), (100, 200), allowed, "B2")
    forward = follower.choose(
        RoverPose((100, 100), np.pi / 2, 1), (100, 200), allowed, "B2"
    )

    assert turn.action == Action.TURN_RIGHT
    assert forward.action == Action.FORWARD


def test_coverage_sweep_builds_alternating_rows_and_stops_when_complete():
    frame = np.zeros((480, 640, 3), np.uint8)
    coverage = CoverageTracker(frame.shape, cols=3, rows=2)
    policy = CoverageSweep(BodyToImage(30, -.2, (1, 0)), margin_frac=.12)

    cells = [waypoint[2] for waypoint in policy._waypoints(frame, coverage)]
    assert cells == [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]

    coverage.visited.update(cells)
    pose = RoverPose((320, 240), 0, 1)
    decision = policy.choose(frame, pose, list(Action), coverage)
    assert decision.action == Action.STOP


def test_coverage_sweep_turns_toward_first_waypoint_and_stops_if_pose_is_lost():
    frame = np.zeros((480, 640, 3), np.uint8)
    coverage = CoverageTracker(frame.shape, cols=3, rows=2)
    policy = CoverageSweep(BodyToImage(30, -.2, (1, 0)), margin_frac=.12)
    pose = RoverPose((320, 240), 0, 1)

    decision = policy.choose(frame, pose, list(Action), coverage)
    assert decision.action == Action.TURN_LEFT
    assert "A1" in decision.reason
    assert policy.choose(frame, None, [Action.BACKWARD, Action.STOP], coverage).action == Action.STOP


def test_coverage_sweep_keeps_turn_direction_across_angle_wrap():
    frame = np.zeros((480, 640, 3), np.uint8)
    coverage = CoverageTracker(frame.shape, cols=3, rows=2)
    policy = CoverageSweep(BodyToImage(30, .2, (1, 0)), margin_frac=.12)
    target_x, target_y, _ = policy._waypoints(frame, coverage)[0]
    desired = np.arctan2(target_y - 240, target_x - 320)

    before_wrap = RoverPose((320, 240), desired - np.deg2rad(179), 1)
    after_wrap = RoverPose((320, 240), desired + np.deg2rad(179), 1)
    first = policy.choose(frame, before_wrap, list(Action), coverage)
    second = policy.choose(frame, after_wrap, list(Action), coverage)

    assert first.action in (Action.TURN_LEFT, Action.TURN_RIGHT)
    assert second.action == first.action


def test_coverage_sweep_reverses_turn_after_overshooting_alignment():
    frame = np.zeros((480, 640, 3), np.uint8)
    coverage = CoverageTracker(frame.shape, cols=3, rows=2)
    policy = CoverageSweep(BodyToImage(30, -.48, (1, 0)), margin_frac=.12)
    target_x, target_y, _ = policy._waypoints(frame, coverage)[0]
    desired = np.arctan2(target_y - 240, target_x - 320)

    before_target = RoverPose((320, 240), desired - np.deg2rad(30), 1)
    after_target = RoverPose((320, 240), desired + np.deg2rad(30), 1)
    first = policy.choose(frame, before_target, list(Action), coverage)
    second = policy.choose(frame, after_target, list(Action), coverage)

    assert first.action == Action.TURN_RIGHT
    assert second.action == Action.TURN_LEFT
