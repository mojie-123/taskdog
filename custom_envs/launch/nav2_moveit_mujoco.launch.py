"""Start the existing Nav2 stack and the new planning-only MoveIt stack."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    here = os.path.dirname(os.path.abspath(__file__))
    nav_launch = os.path.join(here, "nav2_isaac.launch.py")
    moveit_launch = os.path.join(here, "moveit_mujoco.launch.py")

    params_file = LaunchConfiguration("params_file")
    map_file = LaunchConfiguration("map")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    log_level = LaunchConfiguration("log_level")
    moveit_log_level = LaunchConfiguration("moveit_log_level")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", description="Nav2 params yaml"),
        DeclareLaunchArgument("map", description="Nav2 map yaml"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("moveit_log_level", default_value="info"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_launch),
            launch_arguments={
                "params_file": params_file,
                "map": map_file,
                "use_sim_time": use_sim_time,
                "autostart": autostart,
                "log_level": log_level,
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(moveit_launch),
            launch_arguments={
                "moveit_log_level": moveit_log_level,
            }.items(),
        ),
    ])
