"""Team AVAA — ERC 2026 solution entry point.

The organisers invoke this file, and only this file:

    ros2 launch avaa_solution solution.launch.py shelf_column_number:=2 book_colour:=red

Both argument names are fixed by the competition specification. A submission that does
not accept them exactly is not evaluated, so they are validated here and the launch is
aborted with a clear message rather than starting a run that cannot score.

Note that both arguments are *semantic*, not spatial. The column marker digits are
randomised on every simulation load, and the colour->row assignment is randomised
vertically, so `shelf_column_number:=2` refers to whichever physical column happens to
carry the marker "2" on this run. Nothing about position may be hardcoded.

Nodes are kept separate rather than combined into one process: the Phase 1 rubric marks
modularity explicitly, and it means perception can be run and debugged on its own.

The order they run in is not encoded here. Every controller starts when the mission node
publishes its phase, so the sequence lives in one readable state machine rather than
being implied by five nodes each guessing when its turn has come.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

VALID_COLUMNS = ["1", "2", "3", "4", "5"]
VALID_COLOURS = ["red", "blue", "green", "yellow"]


def generate_launch_description() -> LaunchDescription:
    shelf_column_number = LaunchConfiguration("shelf_column_number")
    book_colour = LaunchConfiguration("book_colour")
    save_images = LaunchConfiguration("save_images")

    declare_column = DeclareLaunchArgument(
        "shelf_column_number",
        description=(
            "Overhead marker digit of the target shelf column, 1-5. "
            "Marker placement is randomised each run."
        ),
        choices=VALID_COLUMNS,
    )

    declare_colour = DeclareLaunchArgument(
        "book_colour",
        description=(
            "Colour of the target book. The row it sits on is randomised each run "
            "and must be determined from the camera."
        ),
        choices=VALID_COLOURS,
    )

    declare_save_images = DeclareLaunchArgument(
        "save_images",
        default_value="true",
        description=(
            "Write timestamped annotated frames to erc_images/ during the trial. "
            "Leave on for scored runs; the images are worth +2 per identification."
        ),
    )

    # Launch substitutions are strings. Both nodes declare shelf_column_number as an
    # integer, so the type has to be stated explicitly or the declaration is rejected at
    # startup -- which would fail the run before it began.
    column_as_int = ParameterValue(shelf_column_number, value_type=int)
    save_as_bool = ParameterValue(save_images, value_type=bool)

    # Every node must run on simulation time. Gazebo stamps TF and sensor messages with
    # /clock, which is far behind wall time and advances at the real-time factor. A node
    # on wall time sees every transform as ancient -- tf2 floods with
    # "TF_OLD_DATA ignoring data from the past" and lookups fail, so anything using TF
    # silently gets nothing.
    sim_time = {"use_sim_time": True}

    perception = Node(
        package="avaa_solution",
        executable="perception",
        name="avaa_perception",
        output="screen",
        emulate_tty=True,
        parameters=[sim_time, {
            "shelf_column_number": column_as_int,
            "book_colour": book_colour,
            "save_images": save_as_bool,
        }],
    )

    approach = Node(
        package="avaa_solution",
        executable="approach",
        name="avaa_approach",
        output="screen",
        emulate_tty=True,
        parameters=[sim_time],
    )

    mission = Node(
        package="avaa_solution",
        executable="mission",
        name="avaa_mission",
        output="screen",
        emulate_tty=True,
        parameters=[sim_time, {
            "shelf_column_number": column_as_int,
            "book_colour": book_colour,
        }],
    )

    # move_group, from this package's own configuration. There is no SRDF in the
    # competition image, so there is no MoveIt config package to depend on; the launch
    # that assembles one is included here rather than left to be started by hand,
    # because the grasp cannot plan without it and a scored run gets one attempt.
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("avaa_solution"), "launch",
            "moveit.launch.py")))

    # The arm controllers wait to be told it is their turn. Perception publishes a row
    # and a book point as soon as the book is in frame, which is several metres out
    # while the base is still driving, and both of these start on exactly that.
    grasp = Node(
        package="avaa_solution",
        executable="grasp",
        name="avaa_grasp",
        output="screen",
        emulate_tty=True,
        parameters=[sim_time, {"start_phase": "grasp"}],
    )

    deliver = Node(
        package="avaa_solution",
        executable="deliver",
        name="avaa_deliver",
        output="screen",
        emulate_tty=True,
        parameters=[sim_time, {"start_phase": "deliver"}],
    )

    return LaunchDescription([
        declare_column,
        declare_colour,
        declare_save_images,
        LogInfo(msg=[
            "[AVAA] target column marker=", shelf_column_number,
            "  book colour=", book_colour,
        ]),
        moveit,
        perception,
        approach,
        grasp,
        deliver,
        mission,
    ])
