#!/usr/bin/env python3
"""How accurate is the published book point, in the direction that matters?

    python3 point_accuracy.py [seconds]

The jaws open to 60.5 mm around a 30 mm book, so there is 15 mm of clearance either side.
A lateral error larger than that means a jaw meets the front of the book during the
advance and shoves it instead of straddling it, which is exactly what the last run did:
the right book, reached correctly, knocked flat.

Depth accuracy was measured before at 17 mm. Lateral accuracy never was. This subscribes
to what perception actually publishes and compares every point against Gazebo, in
base_link, so the answer separates a bias (correctable with a constant) from noise
(correctable by averaging) from neither.

Run it alongside a trial.
"""
import math
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped

BASE_Z = 0.186
BOOK_HALF_DEPTH = 0.08


def gz(*args):
    return subprocess.run(["gz", *args], capture_output=True, text=True,
                          timeout=25).stdout


def pose(model):
    lines = [l.strip() for l in gz("model", "-m", model, "-p").splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
            try:
                return ([float(v) for v in line.strip("[]").split()],
                        [float(v) for v in lines[i + 1].strip("[]").split()])
            except ValueError:
                return None, None
    return None, None


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0

    books = [l.strip(" -") for l in gz("model", "--list").splitlines() if "book_col" in l]
    truth = {}
    for name in books:
        p, _ = pose(name)
        if p:
            truth[name] = p
    print("tracking %d books" % len(truth))

    rclpy.init()
    node = rclpy.create_node("point_accuracy")
    samples = []

    def on_point(msg):
        robot, rpy = pose("tiago_pro")
        if robot is None:
            return
        yaw = rpy[2]
        # What perception says, lifted into world coordinates.
        bx, by, bz = msg.point.x, msg.point.y, msg.point.z
        wx = robot[0] + bx * math.cos(yaw) - by * math.sin(yaw)
        wy = robot[1] + bx * math.sin(yaw) + by * math.cos(yaw)
        wz = bz + BASE_Z

        # The book it is closest to. Columns are ~0.95 m apart, so an error of a few
        # centimetres cannot pick the wrong one and the comparison stays honest.
        best = min(truth.items(),
                   key=lambda kv: math.dist((wx, wy, wz), kv[1]))
        name, actual = best
        # Perception reports the face; ground truth is the centre.
        face = [actual[0] - BOOK_HALF_DEPTH, actual[1], actual[2]]
        # Errors expressed back in base_link, which is the frame the grasp solves in.
        ex_w, ey_w = wx - face[0], wy - face[1]
        ex = ex_w * math.cos(-yaw) - ey_w * math.sin(-yaw)
        ey = ex_w * math.sin(-yaw) + ey_w * math.cos(-yaw)
        samples.append((name, bx, ex, ey, wz - face[2]))

    node.create_subscription(PointStamped, "/avaa/perception/target_book_point",
                             on_point, 10)

    print("listening... run a trial now")
    end = time.time() + duration
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()

    if not samples:
        print("no book points were published")
        return

    print()
    print("%d samples" % len(samples))
    print()
    print("%-9s %5s %9s %9s %9s" % ("range", "n", "x err mm", "y err mm", "z err mm"))
    bands = [(0.0, 0.7), (0.7, 0.9), (0.9, 1.2), (1.2, 1.6), (1.6, 9.9)]
    for lo, hi in bands:
        rows = [s for s in samples if lo <= s[1] < hi]
        if not rows:
            continue
        ex = np.array([r[2] for r in rows]) * 1000
        ey = np.array([r[3] for r in rows]) * 1000
        ez = np.array([r[4] for r in rows]) * 1000
        print("%4.1f-%4.1fm %5d %+5.0f+-%-3.0f %+5.0f+-%-3.0f %+5.0f+-%-3.0f"
              % (lo, hi, len(rows), ex.mean(), ex.std(),
                 ey.mean(), ey.std(), ez.mean(), ez.std()))

    close = [s for s in samples if s[1] < 0.95]
    if close:
        ey = np.array([r[3] for r in close]) * 1000
        print()
        print("at grasping range (under 0.95 m): y bias %+.0f mm, spread %.0f mm, "
              "worst %+.0f mm" % (ey.mean(), ey.std(), ey[np.argmax(np.abs(ey))]))
        print("the jaws allow +/-15 mm before a finger meets the front of the book")
        names = {s[0] for s in close}
        print("books seen at that range: %s" % ", ".join(sorted(names)))


if __name__ == "__main__":
    main()
