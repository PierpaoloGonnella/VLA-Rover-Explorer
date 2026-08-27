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
    package = "rover_explorer_ros2"
    params = PathJoinSubstitution([FindPackageShare(package), "config", "rover_params.yaml"])
    policy = LaunchConfiguration("policy")
    is_vlm = PythonExpression(["'", policy, "' == 'vlm'"])
    # The image-plane sign measured on physical hardware is negative, while
    # the simulator models positive angular.z as increasing image heading.
    sim_transform = {"radians_per_turn_pulse": 0.48}
    common = {
        "parameters": [params, sim_transform, {"policy": policy, "planner": LaunchConfiguration("planner")}],
        "output": "screen",
    }
    nodes = [
        Node(package=package, executable=_executable("simulator_node"), name="simulator_node", parameters=[params], output="screen"),
        Node(package=package, executable=_executable("localizer_node"), name="localizer_node", parameters=[params, {"aruco_heading_offset_degrees": 0.0}], output="screen"),
        Node(package=package, executable=_executable("guard_node"), name="guard_node", parameters=[params, sim_transform], output="screen"),
        Node(package=package, executable=_executable("coverage_node"), name="coverage_node", parameters=[params], output="screen"),
        Node(package=package, executable=_executable("obstacle_grid_node"), name="obstacle_grid_node", parameters=[params], output="screen"),
        Node(package=package, executable=_executable("policy_classic_node"), name="policy_classic_node", **common),
        Node(package=package, executable=_executable("policy_vlm_node"), name="policy_vlm_node", condition=IfCondition(is_vlm), **common),
        Node(package=package, executable=_executable("motion_node"), name="motion_node", **common),
        Node(package=package, executable=_executable("logger_node"), name="logger_node", parameters=[params], output="screen"),
    ]
    return LaunchDescription([
        DeclareLaunchArgument("policy", default_value="sweep", choices=["sweep", "frontier", "obstacle_sweep", "bottom_center", "vlm"]),
        DeclareLaunchArgument("planner", default_value="astar_legacy", choices=["astar_legacy", "nav2"]),
        DeclareLaunchArgument("record_rosbag", default_value="false", choices=["true", "false"]),
        *nodes,
        ExecuteProcess(
            cmd=["ros2", "bag", "record", "-a"],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_rosbag")),
        ),
    ])
