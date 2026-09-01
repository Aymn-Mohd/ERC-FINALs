#!/usr/bin/env python3
"""Send one plan request and report exactly what came back."""
import sys
import time

import rclpy
from geometry_msgs.msg import Quaternion

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.moveit_client import MoveItClient, error_name  # noqa: E402

ARM_JOINTS = ["arm_left_%d_joint" % i for i in range(1, 8)]
CHAIN_JOINTS = ["torso_lift_joint"] + ARM_JOINTS
TUCK = [0.15, 2.1521, 0.3824, 1.2785, -2.1517, 0.8325, 0.1926, 1.3944]


def report(label, code, client, started):
    detail = (" -- " + client.last_failure) if client.last_failure else ""
    print("%-34s %s%s  (%.0f s)"
          % (label, error_name(code), detail, time.time() - started))


def main():
    rclpy.init()
    client = MoveItClient()
    if not client.wait_until_ready(40.0):
        print("move_group never came up")
        return 1
    print("connected; scene is empty on purpose for this test")
    print()

    started = time.time()
    code = client.move_to_joints(CHAIN_JOINTS, TUCK, timeout=180.0)
    report("plan to tuck (already there)", code, client, started)

    nudged = list(TUCK)
    nudged[1] -= 0.25
    started = time.time()
    code = client.move_to_joints(CHAIN_JOINTS, nudged, timeout=180.0)
    report("plan to tuck with one joint moved", code, client, started)

    started = time.time()
    code = client.move_to_pose((0.45, 0.20, 0.90),
                               Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                               timeout=180.0)
    report("plan to a pose in open space", code, client, started)

    client.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
