from pathlib import Path

from rover_explorer.calibrate import BodyToImage
from rover_explorer.guard import allowed_actions, apply_ultrasonic_guard
from rover_explorer.localize import RoverPose
from rover_explorer.motion import Action
from rover_explorer.policy import VlmExplorer


def test_lost_pose_is_conservative():
    transform = BodyToImage(30, 0.3, (1, 0))
    assert allowed_actions(None, transform, (480, 640, 3)) == [Action.BACKWARD, Action.STOP]


def test_sonar_veto_preserves_escape_actions():
    guarded = apply_ultrasonic_guard(list(Action), True)
    assert Action.FORWARD not in guarded
    assert Action.BACKWARD in guarded
    assert Action.STOP in guarded


def test_vlm_fallback_is_stop():
    decision = VlmExplorer._fallback([Action.FORWARD, Action.STOP], "failure", "", 0.1)
    assert decision.action == Action.STOP


def test_only_ble_bridge_imports_real_ble_transport():
    package = Path(__file__).parents[1] / "rover_explorer_ros2"
    importers = []
    for path in package.glob("*_node.py"):
        if "RoverBle" in path.read_text(encoding="utf-8"):
            importers.append(path.name)
    assert importers == ["ble_bridge_node.py"]
