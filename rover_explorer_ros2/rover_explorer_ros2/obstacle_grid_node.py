from __future__ import annotations

import math

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Range

from rover_explorer.localize import RoverPose as CorePose
from rover_explorer.obstacle import ObstacleGrid
from rover_explorer_ros2.msg import RoverPose

from .common import declare_transform_parameters, transform_from_node


class ObstacleGridNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_grid_node")
        declare_transform_parameters(self)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("map_cols", 12)
        self.declare_parameter("map_rows", 8)
        self.declare_parameter("obstacle_ttl_cycles", 40)
        self.declare_parameter("cm_per_translation_pulse", 10.0)
        self.declare_parameter("maximum_mapping_distance_cm", 150)
        self.declare_parameter("map_frame_id", "map")
        self._grid = ObstacleGrid(
            (int(self.get_parameter("camera_height").value), int(self.get_parameter("camera_width").value), 3),
            int(self.get_parameter("map_cols").value),
            int(self.get_parameter("map_rows").value),
            int(self.get_parameter("obstacle_ttl_cycles").value),
        )
        self._pose: CorePose | None = None
        self._cycle = 0
        self._publisher = self.create_publisher(OccupancyGrid, "/rover/occupancy_grid", 10)
        self.create_subscription(RoverPose, "/rover/pose", self._on_pose, 10)
        self.create_subscription(Range, "/rover/sonar", self._on_sonar, 10)

    def _on_pose(self, message: RoverPose) -> None:
        self._pose = CorePose(
            (message.centre.x, message.centre.y),
            message.heading if message.has_heading else None,
            message.confidence,
        )

    def _on_range(self, message: Range, offset: float) -> None:
        if self._pose is None or not math.isfinite(message.range):
            return
        self._cycle += 1
        transform = transform_from_node(self)
        heading = self._pose.heading
        if heading is None:
            dx, dy = transform.direction(self._pose)
            heading = math.atan2(dy, dx)
        distance_cm = message.range * 100.0
        maximum = float(self.get_parameter("maximum_mapping_distance_cm").value)
        distance_px = min(distance_cm, maximum) * transform.px_per_forward_pulse / float(
            self.get_parameter("cm_per_translation_pulse").value
        )
        self._grid.observe_ray(
            self._pose.centre, heading + offset, distance_px,
            hit=distance_cm < maximum, cycle=self._cycle,
        )
        self._publish()

    def _on_sonar(self, message: Range) -> None:
        offsets = {
            "sonar_front": 0.0,
            "sonar_left": math.radians(50),
            "sonar_right": -math.radians(50),
        }
        if message.header.frame_id in offsets:
            self._on_range(message, offsets[message.header.frame_id])

    def _publish(self) -> None:
        message = OccupancyGrid()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter("map_frame_id").value)
        message.info.width = self._grid.cols
        message.info.height = self._grid.rows
        message.info.resolution = 1.0
        message.info.origin.orientation.w = 1.0
        occupied = self._grid.occupied()
        message.data = [
            100 if (col, row) in occupied else 0
            for row in range(self._grid.rows)
            for col in range(self._grid.cols)
        ]
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObstacleGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
