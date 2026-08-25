import random

import pytest

from rover_explorer.ble import MockBle
from rover_explorer.calibrate import BodyToImage
from rover_explorer.camera import SimulatedSource
from rover_explorer.coverage import CoverageTracker
from rover_explorer.guard import allowed_actions, safe_rectangle
from rover_explorer.localize import ArucoLocalizer
from rover_explorer.motion import Action, pulse
from rover_explorer.policy import RandomWalk
from rover_explorer.simulator import RoverSimulator


@pytest.mark.asyncio
async def test_full_mock_random_walk_never_leaves_safe_rectangle():
    simulator = RoverSimulator(
        wheel_slip=0, ble_latency_ms=0,
        # Shortened, but still realistic, pulse lengths avoid sub-timer-resolution
        # behavior on Windows while keeping the integration test quick.
        px_per_second_at_full_speed=450,
        radians_per_second_at_full_speed=12,
    )
    ble, camera = MockBle(simulator), SimulatedSource(simulator)
    localizer = ArucoLocalizer()
    transform = BodyToImage(450 * 150 / 255 * .04, 12 * 150 / 255 * .025, (1, 0))
    policy = RandomWalk(seed=19)
    coverage = CoverageTracker((480, 640, 3))
    await ble.connect()
    for _ in range(100):
        frame = camera.read()
        pose = localizer.locate(frame)
        assert pose is not None
        left, top, right, bottom = safe_rectangle(frame.shape, .12)
        assert left <= pose.centre[0] <= right
        assert top <= pose.centre[1] <= bottom
        coverage.update(pose)
        allowed = allowed_actions(pose, transform, frame.shape, .12)
        decision = policy.choose(frame, pose, allowed, coverage)
        duration = 25 if decision.action in (Action.TURN_LEFT, Action.TURN_RIGHT) else 40
        await pulse(ble, decision.action, 150, duration, settle_ms=0)
    await ble.disconnect()
