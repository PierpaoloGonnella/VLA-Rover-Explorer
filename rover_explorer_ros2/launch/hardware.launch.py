import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _executable(name):
    return f"{name}.exe" if os.name == "nt" else name


def generate_launch_description():
    params = LaunchConfiguration("params_file")
    policy = LaunchConfiguration("policy")
    is_vlm = PythonExpression(["'", policy, "' == 'vlm'"])
    common = {"parameters": [params, {"policy": policy, "planner": LaunchConfiguration("planner")}], "output": "screen"}
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=PathJoinSubstitution([
            FindPackageShare("rover_explorer_ros2"), "config", "rover_params.yaml"
        ])),
        DeclareLaunchArgument("policy", default_value="sweep", choices=["sweep", "frontier", "obstacle_sweep", "bottom_center", "vlm"]),
        DeclareLaunchArgument("planner", default_value="astar_legacy", choices=["astar_legacy", "nav2"]),
        DeclareLaunchArgument("record_rosbag", default_value="false", choices=["true", "false"]),
        Node(package="rover_explorer_ros2", executable=_executable("camera_node"), parameters=[params], output="screen"),
        Node(package="rover_explorer_ros2", executable=_executable("localizer_node"), parameters=[params], output="screen"),
        Node(package="rover_explorer_ros2", executable=_executable("ble_bridge_node"), parameters=[params], output="screen"),
        Node(package="rover_explorer_ros2", executable=_executable("guard_node"), parameters=[params], output="screen"),
        Node(package="rover_explorer_ros2", executable=_executable("coverage_node"), parameters=[params], output="screen"),
        Node(package="rover_explorer_ros2", executable=_executable("obstacle_grid_node"), parameters=[params], output="screen"),
        # The classic controller always runs. In VLM mode it follows the latest
        # semantic waypoint while continuing its sweep between VLM updates.
        Node(package="rover_explorer_ros2", executable=_executable("policy_classic_node"), **common),
        Node(package="rover_explorer_ros2", executable=_executable("policy_vlm_node"), condition=IfCondition(is_vlm), **common),
        Node(package="rover_explorer_ros2", executable=_executable("motion_node"), **common),
        Node(package="rover_explorer_ros2", executable=_executable("logger_node"), parameters=[params], output="screen"),
        ExecuteProcess(
            cmd=["ros2", "bag", "record", "-a"],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_rosbag")),
        ),
    ])
