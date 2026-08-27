import numpy as np

from rover_explorer.calibrate import BodyToImage
from rover_explorer.coverage import CoverageTracker
from rover_explorer.localize import RoverPose
from rover_explorer.motion import Action
from rover_explorer.policy import VlmExplorer
from rover_explorer.temporal import TemporalMemory


def test_temporal_memory_detects_motor_stall_and_repeated_turns():
    memory = TemporalMemory(window_seconds=20, stall_displacement_px=12)
    memory.add_pose(RoverPose((100, 100), 0.0, 1.0), now=0.0)
    memory.add_action(Action.FORWARD, now=1.0)
    memory.add_action(Action.FORWARD, now=3.0)
    memory.add_pose(RoverPose((104, 101), 0.0, 1.0), now=5.0)

    status = memory.status(now=5.0)
    assert status.stall_detected
    assert not status.repeated_maneuver

    memory.add_action(Action.TURN_LEFT, now=6.0)
    memory.add_action(Action.TURN_LEFT, now=7.0)
    memory.add_action(Action.TURN_LEFT, now=8.0)
    assert memory.status(now=8.0).repeated_maneuver


def test_temporal_context_carries_objective_plan_scenes_and_outcome():
    memory = TemporalMemory()
    memory.add_scene("Flat floor with a chair on the left.")
    memory.update_plan(
        "Reach open cell C3 without crossing the chair.",
        ("Turn right", "Advance to C3"),
        "C3",
        "Previous attempt was blocked by front sonar.",
    )

    context, status = memory.context(
        legal_actions=[Action.BACKWARD, Action.TURN_RIGHT, Action.STOP],
        sonar_context="front_blocked=True; front=0.18 m",
        now=1.0,
    )

    assert "Reach open cell C3" in context
    assert "Turn right -> Advance to C3" in context
    assert "Flat floor with a chair" in context
    assert "blocked by front sonar" in context
    assert "backward, turn_right, stop" in context
    assert not status.stall_detected


class _Response:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


def test_vlm_observer_receives_past_frames_and_planner_receives_temporal_context():
    calls = []

    def requester(*args, **kwargs):
        payload = kwargs["json"]
        calls.append(payload)
        if "format" not in payload:
            return _Response("The current frame shows visible flat tiled floor with no stairs.")
        return _Response(
            '{"cell":"B2","safe_to_advance":true,"hazard_type":"none",'
            '"hazard_cells":[],"safe_cells":["B2"],'
            '"objective":"Escape the stall toward B2","plan_steps":["Change target","Advance"],'
            '"maneuver":"advance","reason":"past forward attempt stalled","confidence":0.9}'
        )

    frame = np.zeros((240, 320, 3), np.uint8)
    prior = [np.full_like(frame, 10), np.full_like(frame, 20)]
    policy = VlmExplorer(
        BodyToImage(30, -.48, (1, 0)), annotation_style="grid", requester=requester
    )
    decision = policy.choose(
        frame,
        RoverPose((160, 120), 0.0, 1.0),
        list(Action),
        CoverageTracker(frame.shape),
        "front_blocked=False",
        "stall_detected=True; repeated_maneuver=False",
        prior,
    )

    assert len(calls[0]["messages"][0]["images"]) == 3
    assert "ordered oldest to newest" in calls[0]["messages"][0]["content"]
    assert "stall_detected=True" in calls[1]["messages"][1]["content"]
    assert decision.objective == "Escape the stall toward B2"
    assert decision.plan_steps == ("Change target", "Advance")
    assert decision.maneuver == "advance"
