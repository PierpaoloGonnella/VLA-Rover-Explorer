import math

import numpy as np
import pytest

from rover_explorer.ble import MockBle
from rover_explorer.calibrate import _recover_after_lost_motion, calibrate
from rover_explorer.camera import SimulatedSource
from rover_explorer.localize import ArucoLocalizer, RoverPose
from rover_explorer.motion import Action
from rover_explorer.simulator import RoverSimulator


@pytest.mark.asyncio
async def test_calibration_recovers_simulator_forward_axis():
    simulator = RoverSimulator(
        wheel_slip=0, ble_latency_ms=0,
        px_per_second_at_full_speed=1000,
        radians_per_second_at_full_speed=6,
    )
    ble = MockBle(simulator)
    await ble.connect()
    heartbeats = []
    start = (simulator.x, simulator.y)
    transform = await calibrate(
        ble, SimulatedSource(simulator), ArucoLocalizer(),
        speed=150, translation_ms=30, turn_ms=30, settle_ms=0,
        repetitions=3, noise_threshold_px=1,
        return_to_start_each_sample=True,
        on_pulse_completed=lambda: heartbeats.append(True),
    )
    angle = math.atan2(transform.forward_axis_in_image[1], transform.forward_axis_in_image[0])
    assert abs(angle) < math.radians(8)
    assert transform.px_per_forward_pulse > 10
    assert transform.radians_per_turn_pulse > 0
    # Three forward/backward and three left/right pairs each heartbeat, preventing
    # the independent watchdog from interrupting later calibration samples.
    assert len(heartbeats) == 12
    assert math.dist((simulator.x, simulator.y), start) < transform.px_per_forward_pulse


class _RecoveryBle:
    def __init__(self):
        self.commands = []

    async def send(self, command):
        self.commands.append(command)


class _RecoveryCamera:
    def read(self):
        return np.zeros((100, 100, 3), np.uint8)


class _RecoveryLocalizer:
    def __init__(self, ble, expected_command):
        self.ble = ble
        self.expected_command = expected_command

    def locate(self, frame):
        if self.expected_command in self.ble.commands:
            return RoverPose((50, 50), 0, 1)
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_action", "inverse_command"),
    [
        (Action.FORWARD, "A#-150#-150#"),
        (Action.TURN_LEFT, "A#150#-150#"),
    ],
)
async def test_lost_calibration_motion_is_reversed_before_recapture(failed_action, inverse_command):
    ble = _RecoveryBle()
    frame, pose = await _recover_after_lost_motion(
        ble, _RecoveryCamera(), _RecoveryLocalizer(ble, inverse_command),
        failed_action=failed_action, speed=150, duration_ms=0, settle_ms=0,
        recovery_pulses=1, localization_timeout=0, min_confidence=.25,
        on_pulse_completed=None,
    )
    assert inverse_command in ble.commands
    assert pose is not None
