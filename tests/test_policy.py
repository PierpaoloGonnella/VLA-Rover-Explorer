import json

import numpy as np
import pytest
import requests

from rover_explorer.calibrate import BodyToImage
from rover_explorer.coverage import CoverageTracker
from rover_explorer.localize import RoverPose
from rover_explorer.motion import Action
from rover_explorer.policy import CoverageSweep, VlmExplorer


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
