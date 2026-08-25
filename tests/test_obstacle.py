import math

import numpy as np

from rover_explorer.calibrate import BodyToImage
from rover_explorer.coverage import CoverageTracker
from rover_explorer.localize import RoverPose
from rover_explorer.motion import Action
from rover_explorer.obstacle import ObstacleGrid
from rover_explorer.policy import ObstacleSweep


def test_astar_routes_around_inflated_obstacle():
    grid = ObstacleGrid((500, 500, 3), cols=5, rows=5)
    path = grid.astar((0, 2), (4, 2), {(2, 2)})

    assert path is not None
    assert path[0] == (0, 2)
    assert path[-1] == (4, 2)
    assert (2, 2) not in path
    assert len(path) == 7


def test_observed_ray_marks_hit_and_inflates_rover_clearance():
    grid = ObstacleGrid((400, 600, 3), cols=6, rows=4)
    hit = grid.observe_ray((100, 200), 0, 250, hit=True, cycle=1)

    assert hit == (3, 2)
    assert hit in grid.occupied(0)
    assert (2, 2) in grid.occupied(1)
    assert (4, 2) in grid.occupied(1)


def test_obstacle_sweep_waits_for_scan_then_builds_map():
    frame = np.zeros((480, 640, 3), np.uint8)
    coverage = CoverageTracker(frame.shape, cols=3, rows=2)
    transform = BodyToImage(30, -.2, (1, 0))
    policy = ObstacleSweep(
        transform,
        map_cols=8,
        map_rows=6,
        cm_per_translation_pulse=10,
        rover_radius_cm=8,
        maximum_mapping_distance_cm=100,
    )
    pose = RoverPose((320, 240), 0, 1)
    escape_actions = [Action.BACKWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STOP]

    policy.update_obstacles(
        pose, frame.shape, coverage,
        centre_cm=20, left_cm=100, right_cm=100,
        blocked=True, scan_sequence=0,
    )
    waiting = policy.choose(frame, pose, escape_actions, coverage)
    assert waiting.action == Action.STOP
    assert "waiting" in waiting.reason

    policy.update_obstacles(
        pose, frame.shape, coverage,
        centre_cm=20, left_cm=45, right_cm=80,
        blocked=True, scan_sequence=1,
    )
    decision = policy.choose(frame, pose, escape_actions, coverage)
    assert "waiting" not in decision.reason
    assert policy.grid is not None
    assert policy.grid.hits
    assert coverage.report()["blocked_cells"] >= 1
