#!/usr/bin/env python3
"""Where can this robot actually stand to reach each shelf row?

Two constraints squeeze from opposite sides. Stand too far and the arm has to hold a long
reach it does not have the torque for -- three joints saturate and it stalls short. Stand
too close and the pre-grasp folds the arm into the robot's own body and no posture is
collision free at all. Somewhere between is a band, and finding it one five-minute grasp
run at a time is not the way.

This asks the question directly: for each standoff and each row, place the shelf in the
planning scene where it would be, solve the pre-grasp, and check whether a posture exists
that is collision free and can travel the whole way in.

    python3 envelope.py [standoff ...]
"""
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Quaternion

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402
from avaa_solution.moveit_client import MoveItClient  # noqa: E402

CHAIN = ["torso_lift_joint"] + ["arm_left_%d_joint" % i for i in range(1, 8)]
ROWS = [1.391, 1.061, 0.731, 0.401]
SHOULDER_BASE_Z = 0.677
BOARD_DROP = 0.145
SHELF_DEPTH = 0.30
BOOK_HALF_DEPTH = 0.08
SHOULDER_Y = 0.159
STANDOFF_MIN_X = 0.34
GRASP_DEPTH = 0.11
PRE_STANDOFF = 0.15


def main():
    standoffs = [float(v) for v in sys.argv[1:]] or [0.52, 0.56, 0.60, 0.64, 0.68, 0.72]

    rclpy.init()
    client = MoveItClient("envelope")
    if not client.wait_until_ready(40.0):
        print("move_group not up")
        return 1
    chain = ArmChain.from_urdf()

    print("%-9s %-7s %-7s %-7s %-7s" % ("standoff", "row 1", "row 2", "row 3", "row 4"))
    results = {}
    for standoff in standoffs:
        face_x = standoff - BOOK_HALF_DEPTH
        # The shelf where it would be for this standoff.
        centre_x = face_x + SHELF_DEPTH / 2.0 - 0.05
        for index, height in enumerate(ROWS):
            client.add_box("shelf_board_%d" % index, "base_link",
                           (centre_x, 0.0, height - BOARD_DROP), (SHELF_DEPTH, 4.8, 0.04))
        client.add_box("shelf_back", "base_link",
                       (face_x + SHELF_DEPTH, 0.0, 0.9), (0.04, 4.8, 1.8))
        time.sleep(0.4)

        row_marks = []
        for row, height in enumerate(ROWS, start=1):
            pre_x = max(face_x - PRE_STANDOFF, STANDOFF_MIN_X)
            pre = [pre_x, SHOULDER_Y, height]
            grasp = [face_x + GRASP_DEPTH, SHOULDER_Y, height]
            ideal = float(np.clip(height - SHOULDER_BASE_Z + 0.25, 0.0, 0.35))
            pin = {"torso_lift_joint": (ideal, 0.10)}

            mark = "none"
            for _ in range(8):
                solution = chain.ik(pre, approach=[1.0, 0.0, 0.0],
                                    closing=[0.0, 1.0, 0.0], pin=pin)
                if solution is None:
                    continue
                if client.state_valid(CHAIN, solution) is False:
                    mark = "blocked" if mark == "none" else mark
                    continue
                # Can it then travel in?
                seed = list(solution)
                steps = 6
                reached = 0
                for step in range(1, steps + 1):
                    point = [a + (step / float(steps)) * (b - a)
                             for a, b in zip(pre, grasp)]
                    nxt = chain.ik(point, seed=seed, approach=[1.0, 0.0, 0.0],
                                   closing=[0.0, 1.0, 0.0], pin=pin)
                    if nxt is None or client.state_valid(CHAIN, nxt) is False:
                        break
                    seed = nxt
                    reached = step
                if reached == steps:
                    mark = "OK"
                    break
                mark = "%d%%" % int(100 * reached / steps)
            row_marks.append(mark)
            results[(standoff, row)] = mark
        print("%-9.2f %-7s %-7s %-7s %-7s" % (standoff, *row_marks), flush=True)

    print()
    good = [k for k, v in results.items() if v == "OK"]
    if good:
        print("standoffs that work, by row:")
        for row in (1, 2, 3, 4):
            ok = sorted(s for (s, r) in good if r == row)
            print("   row %d: %s" % (row, ", ".join("%.2f" % s for s in ok) or "none"))
    else:
        print("no combination reaches all the way in")

    client.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
