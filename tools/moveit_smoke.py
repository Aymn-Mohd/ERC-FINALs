#!/usr/bin/env python3
"""Does MoveIt plan the motions this solution needs, given goals it can actually use?

The division of labour that this settles on:

    the analytic IK in avaa_solution/kinematics/arm_chain.py says WHERE the arm should be
    MoveIt says HOW to get there without hitting anything

Asking MoveIt for a pose goal instead makes it solve IK with KDL, which is a numerical
solver on an eight-joint redundant chain with a 50 ms budget, and it fails often enough to
be useless here: every pose goal came back "Unable to sample any valid states for goal
tree" for points the analytic solver reaches to a tenth of a millimetre. The analytic
solver also pins the wrist, which a pose goal would have to express as a tolerance anyway.

    python3 moveit_smoke.py <face_x> <row> <book_y>
"""
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Quaternion

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402
from avaa_solution.moveit_client import MoveItClient, error_name  # noqa: E402

ARM_JOINTS = ["arm_left_%d_joint" % i for i in range(1, 8)]
CHAIN_JOINTS = ["torso_lift_joint"] + ARM_JOINTS
TUCK = [0.15, 2.1521, 0.3824, 1.2785, -2.1517, 0.8325, 0.1926, 1.3944]
ROW_HEIGHTS = [1.391, 1.061, 0.731, 0.401]
BOARD_DROP = 0.145
APPROACH = [1.0, 0.0, 0.0]
CLOSING = [0.0, 1.0, 0.0]


def facing_shelf() -> Quaternion:
    """Identity: the grasping frame reaches along base x and closes across base y."""
    return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


def main():
    face_x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.64
    row = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    book_y = float(sys.argv[3]) if len(sys.argv) > 3 else 0.159
    height = ROW_HEIGHTS[row - 1]

    chain = ArmChain.from_urdf()
    rclpy.init()
    client = MoveItClient()
    print("waiting for move_group...", flush=True)
    if not client.wait_until_ready(40.0):
        print("move_group never came up")
        return 1
    print("connected")

    print()
    print("shelf into the scene as boards, so the openings stay open")
    depth = 0.30
    centre_x = face_x + depth / 2.0 - 0.05
    for index, board_height in enumerate(ROW_HEIGHTS):
        client.add_box("shelf_board_%d" % index, "base_link",
                       (centre_x, 0.0, board_height - BOARD_DROP), (depth, 4.8, 0.04))
    client.add_box("shelf_back", "base_link",
                   (face_x + depth, 0.0, 0.9), (0.04, 4.8, 1.8))
    print("   5 boxes placed from a measured face at x=%.3f" % face_x)

    print()
    print("1. plan to the tuck posture")
    started = time.time()
    code = client.move_to_joints(CHAIN_JOINTS, TUCK, timeout=200.0)
    print("   %s (%.0f s)" % (error_name(code), time.time() - started))

    print()
    print("2. a goal inside a shelf board, which must be refused")
    inside = [face_x + 0.15, book_y, height - BOARD_DROP]
    solution = chain.ik(inside, approach=APPROACH, closing=CLOSING)
    if solution is None:
        print("   the analytic IK will not even solve it, so MoveIt is not asked")
    else:
        started = time.time()
        code = client.move_to_joints(CHAIN_JOINTS, solution, timeout=90.0)
        verdict = "correctly refused" if code != 1 else "ACCEPTED, which is wrong"
        print("   %s: %s (%.0f s)" % (error_name(code), verdict, time.time() - started))

    print()
    print("3. reach the pre-grasp in front of row %d" % row)
    pre = [face_x - 0.15, book_y, height]
    solution = chain.ik(pre, approach=APPROACH, closing=CLOSING)
    if solution is None:
        print("   no analytic IK for %s" % np.round(pre, 3).tolist())
        return 1
    started = time.time()
    code = client.move_to_joints(CHAIN_JOINTS, solution, timeout=200.0)
    print("   %s (%.0f s)" % (error_name(code), time.time() - started))
    if code != 1:
        return 1

    print()
    print("4. straight line in to the book")
    target = Pose()
    target.position.x = face_x + 0.11
    target.position.y = book_y
    target.position.z = height
    target.orientation = facing_shelf()
    started = time.time()
    code, fraction = client.straight_line([target], timeout=200.0)
    print("   %s, %.0f%% of the way (%.0f s)"
          % (error_name(code), fraction * 100.0, time.time() - started))

    client.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
