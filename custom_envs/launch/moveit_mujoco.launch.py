"""MoveIt planning-only launch for the MuJoCo Piper integration.

No ros2_control controller is started here.  move_group computes IK / collision
checks / trajectories; navigate_mujoco_moveit.py executes those joint paths using
the existing MuJoCo PD controller.
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def generate_launch_description():
    here = os.path.dirname(os.path.abspath(__file__))
    custom_envs = os.path.abspath(os.path.join(here, ".."))

    default_urdf = os.path.join(
        custom_envs, "assets", "m20_piper_single", "M20_Piper_moveit.urdf")
    config_dir = os.path.join(custom_envs, "config", "moveit")
    default_srdf = os.path.join(config_dir, "M20_Piper_moveit.srdf")

    urdf_arg = LaunchConfiguration("moveit_urdf")
    srdf_arg = LaunchConfiguration("moveit_srdf")
    log_level = LaunchConfiguration("moveit_log_level")

    # Launch substitutions cannot be opened directly here, so this launch is
    # intentionally based on the default files next to the project.  The args
    # remain visible for future extension, but the normal project invocation
    # does not need to override them.
    with open(default_urdf, "r") as f:
        robot_description_text = f.read()
    with open(default_srdf, "r") as f:
        robot_description_semantic_text = f.read()

    kinematics = _load_yaml(os.path.join(config_dir, "kinematics.yaml"))
    joint_limits = _load_yaml(os.path.join(config_dir, "joint_limits.yaml"))
    ompl = _load_yaml(os.path.join(config_dir, "ompl_planning.yaml"))

    robot_description = {"robot_description": robot_description_text}
    robot_description_semantic = {
        "robot_description_semantic": robot_description_semantic_text}
    robot_description_kinematics = {"robot_description_kinematics": kinematics}
    robot_description_planning = {"robot_description_planning": joint_limits}

    planning_pipeline = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl,
    }

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            planning_pipeline,
            {
                "use_sim_time": False,
                "allow_trajectory_execution": False,
                "publish_robot_description": True,
                "publish_robot_description_semantic": True,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
                "monitor_dynamics": False,
            },
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="moveit_robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": False}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("moveit_urdf", default_value=default_urdf),
        DeclareLaunchArgument("moveit_srdf", default_value=default_srdf),
        DeclareLaunchArgument("moveit_log_level", default_value="info"),
        robot_state_publisher,
        move_group,
    ])
