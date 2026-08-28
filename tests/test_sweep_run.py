import pytest

from rover_explorer.config import AppConfig
from rover_explorer.runner import run


@pytest.mark.parametrize("policy_name", ["sweep", "obstacle_sweep"])
@pytest.mark.asyncio
async def test_sweep_policy_covers_complete_mock_grid(tmp_path, policy_name):
    config = AppConfig()
    config.camera.width = 640
    config.camera.height = 480
    config.motion.speed = 150
    config.motion.translation_ms = 40
    config.motion.turn_ms = 25
    config.motion.settle_ms = 0
    config.calibration.repetitions = 2
    config.calibration.minimum_valid_samples = 2
    config.calibration.sample_retries = 1
    config.guard.margin_frac = .12
    config.coverage.cols = 3
    config.coverage.rows = 2
    config.coverage.target = 1.0
    config.runner.target_cycle_seconds = 0
    config.simulator.wheel_slip = 0
    config.simulator.ble_latency_ms = 0
    config.simulator.px_per_second_at_full_speed = 450
    config.simulator.radians_per_second_at_full_speed = 12

    report = await run(
        config,
        transport="mock",
        localizer_name="aruco",
        policy_name=policy_name,
        annotation_style="grid",
        cycles=300,
        session_dir=tmp_path,
        # Coverage-policy behavior must not depend on platform-specific ArUco
        # calibration estimates. The simulator's nominal motion model is fixed
        # for this test; calibration is verified by its dedicated test suite.
        do_calibrate=False,
    )

    assert report["fraction_visited"] == 1.0
    assert report["lost_frames"] == 0
