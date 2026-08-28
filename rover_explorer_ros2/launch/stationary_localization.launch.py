import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _executable(name):
    return f"{name}.exe" if os.name == "nt" else name


def generate_launch_description():
    params = LaunchConfiguration("params_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("rover_explorer_ros2"),
                "config",
                "rover_params.yaml",
            ]),
        ),
        Node(
            package="rover_explorer_ros2",
            executable=_executable("camera_node"),
            parameters=[params],
            output="screen",
        ),
        Node(
            package="rover_explorer_ros2",
            executable=_executable("localizer_node"),
            parameters=[params, {
                "capture_failure_images": True,
                "failure_image_min_interval_seconds": 2.0,
                "failure_image_max_per_session": 50,
            }],
            output="screen",
        ),
        Node(
            package="rover_explorer_ros2",
            executable=_executable("logger_node"),
            parameters=[params],
            output="screen",
        ),
    ])
