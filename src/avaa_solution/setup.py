import os
from glob import glob

from setuptools import find_packages, setup

package_name = "avaa_solution"

setup(
    name=package_name,
    version="0.1.0",
    # find_packages picks up avaa_solution and avaa_solution.vision.
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        # Without this the launch file is not installed and
        # `ros2 launch avaa_solution solution.launch.py` fails with "file not found".
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        # The MoveIt configuration: an SRDF written by hand because the image ships
        # none, plus the kinematics, limits, planner and controller settings that go
        # with it. move_group reads these from the share directory at launch.
        (os.path.join("share", package_name, "moveit"),
         glob("moveit/*.srdf") + glob("moveit/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team AVAA",
    maintainer_email="psych.rest@gmail.com",
    description="Team AVAA solution for the Emirates Robotics Competition 2026 (Phase 1).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mission = avaa_solution.mission_node:main",
            "approach = avaa_solution.approach_node:main",
            "grasp = avaa_solution.grasp_node:main",
            "perception = avaa_solution.perception_node:main",
        ],
    },
)
