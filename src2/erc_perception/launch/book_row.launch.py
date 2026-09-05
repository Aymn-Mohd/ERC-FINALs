from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('book_colour', default_value='red'),
        Node(
            package='erc_perception',
            executable='book_row_node',
            output='screen',
            parameters=[{
                'book_colour': LaunchConfiguration('book_colour'),
                'use_sim_time': True,
            }],
        ),
    ])
