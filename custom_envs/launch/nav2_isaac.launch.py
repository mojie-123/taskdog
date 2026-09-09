# Custom Nav2 launch file for Isaac Sim ground-truth localisation.
# Differences from nav2_bringup/navigation_launch.py:
#   1. NO /tf -> tf remapping: bridge publishes to absolute /tf and /tf_static
#   2. Adds map_server (no AMCL): bridge provides ground-truth TF directly
#   3. controller_server publishes cmd_vel_nav; velocity_smoother reads cmd_vel_nav
#      and publishes cmd_vel (remapped from cmd_vel_smoothed); bridge subscribes /cmd_vel

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():

    params_file   = LaunchConfiguration('params_file')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time  = LaunchConfiguration('use_sim_time')
    autostart     = LaunchConfiguration('autostart')
    log_level     = LaunchConfiguration('log_level')

    # Inject map path and use_sim_time into the yaml at launch time
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'yaml_filename': map_yaml_file,
        'autostart': autostart,
    }
    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key='',
            param_rewrites=param_substitutions,
            convert_types=True),
        allow_substs=True)

    lifecycle_nodes = [
        'map_server',
        'controller_server',
        'smoother_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
        'velocity_smoother',
    ]

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),

        DeclareLaunchArgument('params_file',
            description='Full path to nav2 params yaml'),
        DeclareLaunchArgument('map',
            description='Full path to map yaml file'),
        DeclareLaunchArgument('use_sim_time',  default_value='false'),
        DeclareLaunchArgument('autostart',     default_value='true'),
        DeclareLaunchArgument('log_level',     default_value='info'),

        GroupAction(actions=[

            # --- Map server (no AMCL; Isaac bridge provides ground-truth TF) ---
            Node(
                package='nav2_map_server',
                executable='map_server',
                name='map_server',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level]),

            # --- Navigation stack ---
            Node(
                package='nav2_controller',
                executable='controller_server',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=[('cmd_vel', 'cmd_vel_nav')]),
            Node(
                package='nav2_smoother',
                executable='smoother_server',
                name='smoother_server',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_waypoint_follower',
                executable='waypoint_follower',
                name='waypoint_follower',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level]),
            Node(
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=[
                    ('cmd_vel_smoothed', 'cmd_vel'),
                ]),


            # --- Lifecycle manager ---
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'autostart':    autostart,
                    'node_names':   lifecycle_nodes,
                }],
                arguments=['--ros-args', '--log-level', log_level]),
        ]),
    ])
