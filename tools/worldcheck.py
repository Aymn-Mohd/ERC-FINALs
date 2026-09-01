#!/usr/bin/env python3
"""Where is the gripper, and where is the book, in the world?

Every measurement so far has been in base_link, and base_link moves. The grasp reports
arriving within 2 mm of its target while the book is left completely undisturbed, which
can only mean the target is not where the book is. Frames are the obvious suspect: the
target is taken from ground truth, converted into base_link, anchored in odom, and
converted back, and any of those steps can be carrying an error.

World coordinates settle it. Both positions come from the simulator, one via the robot
pose and TF, the other straight from the book model.
"""
import math
import subprocess
import sys
import time

import numpy as np
import rclpy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


def gz_pose(model):
    out = subprocess.run(["gz", "model", "-m", model, "-p"],
                         capture_output=True, text=True, timeout=25).stdout
    lines = [l.strip() for l in out.splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
            return ([float(v) for v in line.strip("[]").split()],
                    [float(v) for v in lines[i + 1].strip("[]").split()])
    return None, None


def main():
    book_name = sys.argv[1] if len(sys.argv) > 1 else "book_col_3_row_3_blue"

    rclpy.init()
    node = rclpy.create_node("world_check")
    node.set_parameters([rclpy.parameter.Parameter(
        "use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    buf = Buffer()
    TransformListener(buf, node)
    state = {"grasp": "?"}
    node.create_subscription(String, "/avaa/grasp/state",
                             lambda m: state.__setitem__("grasp", m.data), 10)

    end = time.time() + 8
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    print("%-11s %-26s %-26s %s"
          % ("state", "gripper in world", "book in world", "miss (mm)"))
    seen = set()
    stop = time.time() + 600
    while rclpy.ok() and time.time() < stop:
        rclpy.spin_once(node, timeout_sec=0.2)
        now = state["grasp"]
        try:
            t = buf.lookup_transform(
                "base_link", "gripper_left_grasping_link",
                rclpy.time.Time()).transform.translation
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
            continue
        robot, rpy = gz_pose("tiago_pro")
        book, _ = gz_pose(book_name)
        if robot is None or book is None:
            continue
        yaw = rpy[2]
        # The gripper, from the base pose and the arm's own transform.
        gx = robot[0] + t.x * math.cos(yaw) - t.y * math.sin(yaw)
        gy = robot[1] + t.x * math.sin(yaw) + t.y * math.cos(yaw)
        gz_ = robot[2] + t.z
        miss = np.array([gx - book[0], gy - book[1], gz_ - book[2]])

        key = (now, round(miss[1], 3))
        if key not in seen:
            seen.add(key)
            print("%-11s (%.3f,%+.3f,%.3f)    (%.3f,%+.3f,%.3f)    "
                  "%+.0f %+.0f %+.0f"
                  % (now, gx, gy, gz_, book[0], book[1], book[2],
                     miss[0] * 1000, miss[1] * 1000, miss[2] * 1000))
        if now in ("done", "failed"):
            break
        time.sleep(0.6)

    print()
    print("The book centre is the target for the jaws; the grasping frame should sit")
    print("about 27 mm short of it along the approach, and level with it otherwise.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
