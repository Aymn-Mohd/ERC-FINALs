"""Bring up move_group for the left arm.

There is no SRDF in the competition image, so there is no MoveIt config package either
and nothing to hand to moveit_configs_utils. The configuration is assembled here from the
files in avaa_solution/moveit/ instead, which is more code than a generated package but
has the advantage that every value in it is one we chose and can explain.

    ros2 launch avaa_solution moveit.launch.py
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "avaa_solution"
URDF = "/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf"


def read(path):
    with open(path, "r") as handle:
        return handle.read()


def load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def generate_launch_description():
    config = os.path.join(get_package_share_directory(PACKAGE), "moveit")

    kinematics = load_yaml(os.path.join(config, "kinematics.yaml"))
    joint_limits = load_yaml(os.path.join(config, "joint_limits.yaml"))
    ompl = load_yaml(os.path.join(config, "ompl_planning.yaml"))
    controllers = load_yaml(os.path.join(config, "moveit_controllers.yaml"))

    parameters = [
        {"robot_description": read(URDF)},
        {"robot_description_semantic": read(os.path.join(config, "tiago_pro.srdf"))},
        {"robot_description_kinematics": kinematics},
        {"robot_description_planning": joint_limits},
        {"planning_pipelines": ["ompl"]},
        {"default_planning_pipeline": "ompl"},
        {"ompl": ompl},
        controllers,
        {
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            # Other nodes read the semantic model from here rather than parsing the SRDF
            # again and risking a different view of the robot than the planner has.
            "publish_robot_description_semantic": True,
            # The planning scene is published on change so the grasp controller can see
            # what the planner believes about the shelf, which is the whole point of
            # moving to MoveIt: the failures that cost the most were invisible.
            "publish_planning_scene": True,
            "publish_geometry_updates": True,
            "publish_state_updates": True,
            "publish_transforms_updates": True,
            # No depth camera into the octomap. The shelf is added as an explicit box by
            # avaa_solution/shelf_scene.py, measured from the laser, because an octomap
            # built from a head camera that is busy looking at one book has holes exactly
            # where the arm is about to reach.
            "octomap_frame": "",
            "trajectory_execution.allowed_execution_duration_scaling": 3.0,
            "trajectory_execution.allowed_goal_duration_margin": 10.0,
            # This arm lags its commands badly; the default 0.01 rad tolerance fails
            # every trajectory it is given.
            "trajectory_execution.allowed_start_tolerance": 0.10,
        },
    ]

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=parameters,
        ),
    ])
