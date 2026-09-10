# Custom Nav2 launch file for Isaac Sim ground-truth localisation.
# Differences from nav2_bringup/navigation_launch.py:
#   1. NO /tf -> tf remapping: bridge publishes to absolute /tf and /tf_static
#      标准 Nav2 中，AMCL 发布 map → odom 变换，而机器人驱动发布 odom → base_link 变换。而这里Isaac Sim 的 ROS 2 Bridge 直接发布完整的坐标变换链：map → odom → base_link
#   2. Adds map_server (no AMCL): bridge provides ground-truth TF directly
#   3. controller_server publishes cmd_vel_nav; velocity_smoother reads cmd_vel_nav
#      and publishes cmd_vel (remapped from cmd_vel_smoothed); bridge subscribes /cmd_vel

# 重映射remapping：将节点默认发布/订阅的话题名称改为另一个名称。

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
        RewrittenYaml(   # 将命令行传入的 map 路径注入到 map_server 的 yaml_filename 参数中
            source_file=params_file,   # 读取命令行传入的 nav2_params.yaml 文件
            root_key='',   # 从 YAML 根节点开始替换
            param_rewrites=param_substitutions,   # 用字典中的值替换 YAML 中的对应参数
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
    ]   # 声明所有需要生命周期管理器统一控制的节点

    # 启动参数声明，DeclareLaunchArgument的这几个参数可以通过命令行传入
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
            Node(   # 启动一个ROS2节点
                package='nav2_map_server',   # ROS2包名
                executable='map_server',   # 可执行文件名
                name='map_server',   # 节点名称
                output='screen',    # 日志输出方式 screen终端/log日志文件
                parameters=[configured_params],   # 加载yaml文件
                arguments=['--ros-args', '--log-level', log_level]),   # 命令行参数

            # --- Navigation stack ---
            Node(
                package='nav2_controller',
                executable='controller_server',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=[('cmd_vel', 'cmd_vel_nav')]),   # 重映射话题列表
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
