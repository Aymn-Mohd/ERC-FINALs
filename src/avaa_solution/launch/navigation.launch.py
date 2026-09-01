"""Bring up Nav2 for the ERC arena.

Map-less: no map_server and no AMCL. Everything runs in the ``odom`` frame with rolling
costmaps built from the two LiDARs. See ``config/nav2_params.yaml`` for why.

Started separately from solution.launch.py so navigation can be brought up and debugged on
its own:

    ros2 launch avaa_solution navigation.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Order matters: the lifecycle manager brings these up in sequence, and the controller
# needs its costmap before it can configure.
LIFECYCLE_NODES = [
    "controller_server",
    "planner_server",
    "behavior_server",
    "bt_navigator",
    "velocity_smoother",
]


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    default_params = os.path.join(
        get_package_share_directory("avaa_solution"), "config", "nav2_params.yaml"
    )

    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Nav2 parameters YAML.",
    )
    declare_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description=(
            "Use /clock from Gazebo. Must stay true in simulation: the sim runs at a "
            "real-time factor well below 1, so wall-clock timing would make every "
            "transform look stale and Nav2 would reject it."
        ),
    )

    common = {"use_sim_time": use_sim_time}

    controller = Node(
        package="nav2_controller", executable="controller_server", output="screen",
        parameters=[params_file, common],
        # Nav2 publishes to /cmd_vel_nav ahead of the smoother, which then emits /cmd_vel.
        remappings=[("cmd_vel", "cmd_vel_nav")],
    )
    planner = Node(
        package="nav2_planner", executable="planner_server", output="screen",
        parameters=[params_file, common],
    )
    behaviors = Node(
        package="nav2_behaviors", executable="behavior_server", output="screen",
        parameters=[params_file, common],
    )
    bt_navigator = Node(
        package="nav2_bt_navigator", executable="bt_navigator", output="screen",
        parameters=[params_file, common],
    )
    smoother = Node(
        package="nav2_velocity_smoother", executable="velocity_smoother", output="screen",
        parameters=[params_file, common],
        remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
    )
    lifecycle = Node(
        package="nav2_lifecycle_manager", executable="lifecycle_manager",
        name="lifecycle_manager_navigation", output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": LIFECYCLE_NODES,
        }],
    )

    return LaunchDescription([
        declare_params,
        declare_sim_time,
        controller,
        planner,
        behaviors,
        bt_navigator,
        smoother,
        lifecycle,
    ])
