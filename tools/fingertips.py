#!/usr/bin/env python3
"""Where are the fingertips when the grasping frame is on the book?

Everything upstream is now correct and the grasp still fails. Given a target taken
straight from Gazebo the arm puts gripper_left_grasping_link at the commanded point to
the millimetre, and the fingers still close to 0.0000, which is full closure -- nothing
between them. So the question is whether that frame is where the fingers actually meet.

The URDF puts the finger roots 0.0756 m along the gripper base z, and the grasping frame
0.157 m along it. That is an 81 mm difference, and the whole book is only 160 mm deep, so
it matters a great deal whether the grasping frame is between the fingertips or well past
them.

This walks the real chains to both fingertips and compares them against the grasping
frame, at the actual joint values the last grasp used.
"""
import sys

import numpy as np

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402

URDF = "/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf"

# The configuration the last ideal grasp actually held, from /joint_states.
TORSO = 0.335
ARM = [3.61, -1.66, -0.91, -0.47, -2.05, 0.79, -0.83]
FINGER_OPEN = 0.040


def describe(tip, values, label):
    chain = ArmChain.from_urdf(URDF, "base_link", tip)
    names = chain.joint_names
    if len(values) != len(names):
        print("  %-28s expected %d joints %s" % (label, len(names), names))
        return None
    pose = chain.fk(values)
    print("  %-28s %s" % (label, np.round(pose[:3, 3], 4).tolist()))
    return pose[:3, 3]


def main():
    print("chain to the grasping frame:")
    grasping = ArmChain.from_urdf(URDF, "base_link", "gripper_left_grasping_link")
    print("  joints: %s" % grasping.joint_names)
    origin = grasping.fk([TORSO] + ARM)[:3, 3]
    print("  grasping frame at %s" % np.round(origin, 4).tolist())
    print()

    for tip in ("gripper_left_fingertip_left_link", "gripper_left_fingertip_right_link",
                "gripper_left_base_finger_left_link", "gripper_left_base_link"):
        chain = ArmChain.from_urdf(URDF, "base_link", tip)
        names = chain.joint_names
        # Fill the finger joints: the prismatic opening, and the passive linkage at zero.
        values = [TORSO] + ARM
        for name in names[len(values):]:
            values.append(FINGER_OPEN if "finger_joint" in name else 0.0)
        pose = chain.fk(values)
        point = pose[:3, 3]
        print("%-42s %s   %+.3f m along approach from grasping frame"
              % (tip.replace("gripper_left_", ""), np.round(point, 4).tolist(),
                 float(np.dot(point - origin, grasping.fk([TORSO] + ARM)[:3, 0]))))
        print("     extra joints: %s" % names[8:])

    print()
    print("the book face was at x=0.819 and the book is 0.160 m deep, so it occupies")
    print("0.819 to 0.979 along base x. The grasping frame was commanded to 0.869.")


if __name__ == "__main__":
    main()
