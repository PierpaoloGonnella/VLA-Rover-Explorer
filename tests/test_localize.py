import json
import math

from rover_explorer.localize import ArucoLocalizer, VlmLocalizer
from rover_explorer.simulator import RoverSimulator


class StubResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": json.dumps(self.content)}}


def test_aruco_is_near_ground_truth_on_simulated_frame():
    simulator = RoverSimulator(wheel_slip=0)
    pose = ArucoLocalizer().locate(simulator.render())
    assert pose is not None
    assert math.dist(pose.centre, (simulator.x, simulator.y)) < 2.0


def test_aruco_heading_offset_rotates_reported_forward_axis():
    simulator = RoverSimulator(wheel_slip=0)
    frame = simulator.render()
    base = ArucoLocalizer().locate(frame)
    offset = ArucoLocalizer(heading_offset_radians=-math.pi / 2).locate(frame)
    assert base is not None and offset is not None
    difference = (offset.heading - base.heading + math.pi) % (2 * math.pi) - math.pi
    assert math.isclose(difference, -math.pi / 2, abs_tol=1e-6)


def test_vlm_grid_label_converts_to_cell_centre():
    simulator = RoverSimulator(width=600, height=400)
    calls = []

    def requester(*args, **kwargs):
        calls.append((args, kwargs))
        return StubResponse({"cell": "C2", "confidence": .4})

    pose = VlmLocalizer(cols=6, rows=4, requester=requester).locate(simulator.render())
    assert calls
    assert pose is not None
    assert pose.centre == (250.0, 150.0)
    assert pose.heading is None
