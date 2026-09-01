#!/usr/bin/env python3
"""Where are the jaws, relative to the frame the IK aims?

The grasp aims gripper_left_grasping_link at the book and reports arriving within a
millimetre, and the jaws still close beside it: measured through TF, the fingertip
midpoint sat at y=0.206 while the grasping frame was at 0.161.

An earlier version of this calculation left the finger linkage at zero, which is wrong.
The fingers are driven through mimic joints -- inner and outer at -8.28 times the
prismatic command, the fingertip at +8.28 -- so the tips swing as the gripper opens and
the offset between the grasping frame and the jaws is not a constant.
"""
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402

URDF = "/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf"
ARM = [2.86, -2.38, -0.88, -2.22, 1.57, -0.32, -1.71]   # a real grasp posture
TORSO = 0.25


def mimics():
    """Read the mimic multipliers so the linkage is posed correctly."""
    table = {}
    for joint in ET.parse(URDF).getroot().findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            table[joint.get("name")] = (mimic.get("joint"),
                                        float(mimic.get("multiplier", 1.0)),
                                        float(mimic.get("offset", 0.0)))
    return table


def main():
    table = mimics()
    grasping = ArmChain.from_urdf(URDF, "base_link", "gripper_left_grasping_link")
    origin_pose = grasping.fk([TORSO] + ARM)
    origin = origin_pose[:3, 3]
    approach_axis = origin_pose[:3, 0]
    closing_axis = origin_pose[:3, 1]

    print("grasping frame at %s" % np.round(origin, 4).tolist())
    print("approach axis    %s" % np.round(approach_axis, 3).tolist())
    print("closing axis     %s" % np.round(closing_axis, 3).tolist())
    print()
    print("%-9s %-26s %-26s %8s %10s" %
          ("finger", "left tip", "right tip", "span mm", "midpoint offset"))

    for finger in (0.040, 0.020, 0.0012, -0.001):
        tips = []
        for side in ("left", "right"):
            tip = "gripper_left_fingertip_%s_link" % side
            chain = ArmChain.from_urdf(URDF, "base_link", tip)
            values = [TORSO] + ARM
            for name in chain.joint_names[len(values):]:
                if name in table:
                    _, multiplier, offset = table[name]
                    values.append(finger * multiplier + offset)
                elif "finger_joint" in name:
                    values.append(finger)
                else:
                    values.append(0.0)
            tips.append(chain.fk(values)[:3, 3])
        left, right = tips
        mid = (left + right) / 2.0
        delta = mid - origin
        along = float(np.dot(delta, approach_axis))
        across = float(np.dot(delta, closing_axis))
        print("%-9.4f (%.3f,%+.3f,%.3f)    (%.3f,%+.3f,%.3f)    %8.1f  "
              "%+.0f mm along, %+.0f mm across"
              % (finger, left[0], left[1], left[2], right[0], right[1], right[2],
                 float(np.linalg.norm(left - right)) * 1000,
                 along * 1000, across * 1000))

    print()
    print("'along' is into the shelf, 'across' is the direction the fingers close.")
    print("The IK aims the grasping frame, so a non-zero offset is exactly how far the")
    print("jaws end up from the book when the frame is placed on it.")


if __name__ == "__main__":
    main()
