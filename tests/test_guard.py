import math

from rover_explorer.calibrate import BodyToImage
from rover_explorer.guard import TRANSLATING_ACTIONS, allowed_actions, apply_ultrasonic_guard, safe_rectangle
from rover_explorer.localize import RoverPose
from rover_explorer.motion import Action


def test_lost_action_set_is_strictly_conservative():
    transform = BodyToImage(30, 0.4, (1, 0))
    assert allowed_actions(None, transform, (480, 640, 3), .12) == [Action.BACKWARD, Action.STOP]


def test_safe_rectangle_scales_each_axis_independently():
    left, top, right, bottom = safe_rectangle((480, 640, 3), .12)
    assert math.isclose(left, 76.8)
    assert math.isclose(top, 57.6)
    assert math.isclose(right, 563.2)
    assert math.isclose(bottom, 422.4)


def test_ultrasonic_guard_vetoes_forward_motion_but_preserves_escape_actions():
    actions = list(Action)
    guarded = apply_ultrasonic_guard(actions, blocked=True)
    assert Action.FORWARD not in guarded
    assert Action.ARC_LEFT not in guarded
    assert Action.ARC_RIGHT not in guarded
    assert Action.BACKWARD in guarded
    assert Action.TURN_LEFT in guarded
    assert Action.TURN_RIGHT in guarded
    assert Action.STOP in guarded


def test_guard_property_no_allowed_translation_predicts_outside_inset():
    shape = (480, 640, 3)
    transform = BodyToImage(42, 0.45, (1, 0))
    rectangle = safe_rectangle(shape, .12)
    for x in range(40, 641, 25):
        for y in range(40, 481, 25):
            for heading_index in range(16):
                pose = RoverPose((x, y), heading_index * 2 * math.pi / 16, 1.0)
                allowed = allowed_actions(pose, transform, shape, .12)
                for action in set(allowed).intersection(TRANSLATING_ACTIONS):
                    px, py = transform.predict(pose, action)
                    left, top, right, bottom = rectangle
                    assert left <= px <= right
                    assert top <= py <= bottom
                assert Action.TURN_LEFT in allowed
                assert Action.TURN_RIGHT in allowed
