from __future__ import annotations

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

from rover_explorer.coverage import CoverageTracker
from rover_explorer.localize import RoverPose as CorePose
from rover_explorer_ros2.msg import RoverPose


class CoverageNode(Node):
    def __init__(self) -> None:
        super().__init__("coverage_node")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("coverage_cols", 6)
        self.declare_parameter("coverage_rows", 4)
        self.declare_parameter("map_frame_id", "map")
        self._tracker = CoverageTracker(
            (int(self.get_parameter("camera_height").value), int(self.get_parameter("camera_width").value), 3),
            int(self.get_parameter("coverage_cols").value),
            int(self.get_parameter("coverage_rows").value),
        )
        self._publisher = self.create_publisher(OccupancyGrid, "/rover/coverage_map", 10)
        self.create_subscription(RoverPose, "/rover/pose", self._on_pose, 10)

    def _on_pose(self, message: RoverPose) -> None:
        pose = CorePose(
            (message.centre.x, message.centre.y),
            message.heading if message.has_heading else None,
            message.confidence,
        )
        self._tracker.update(pose)
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = str(self.get_parameter("map_frame_id").value)
        grid.info.width = self._tracker.cols
        grid.info.height = self._tracker.rows
        grid.info.resolution = 1.0
        grid.info.origin.orientation.w = 1.0
        grid.data = [
            100 if (col, row) in self._tracker.visited else 0
            for row in range(self._tracker.rows)
            for col in range(self._tracker.cols)
        ]
        self._publisher.publish(grid)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoverageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
