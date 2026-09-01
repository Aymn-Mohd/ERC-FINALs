#!/usr/bin/env python3
"""Watch the real fingertips close, from TF, against the real book, from Gazebo.

Modelling the finger linkage by hand is guesswork -- the opening is driven through a
passive four-bar, and the prismatic joint that commands it is not even on the chain to
the fingertip. So this asks the simulator instead: where are the two fingertip links, how
far apart are they, and where is the book, at every stage of the grasp.

Run it alongside ideal_grasp.py. The question it answers is whether the fingers ever
straddle the book, or whether they close somewhere the book is not and shove it on the way.
"""
import math
import subprocess
import sys
import time

import rclpy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

LEFT = "gripper_left_fingertip_left_link"
RIGHT = "gripper_left_fingertip_right_link"
GRASP = "gripper_left_grasping_link"
BASE = "base_link"
BASE_Z = 0.186


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
    colour = sys.argv[1] if len(sys.argv) > 1 else "red"
    names = [l.strip(" -") for l in gz("model", "--list").splitlines()
             if "book_col" in l and colour in l]
    robot, rpy = pose("tiago_pro")
    best = None
    for name in names:
        p, _ = pose(name)
        if p:
            d = math.hypot(p[0] - robot[0], p[1] - robot[1])
            if best is None or d < best[0]:
                best = (d, name, p)
    _, book_name, book = best
    yaw = rpy[2]
    dx, dy = book[0] - robot[0], book[1] - robot[1]
    bx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    by = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    bz = book[2] - BASE_Z
    print("%s in base_link: centre x=%.3f y=%+.3f z=%.3f" % (book_name, bx, by, bz))
    print("  it occupies x %.3f..%.3f and y %+.3f..%+.3f (30 mm thick, 160 mm deep)"
          % (bx - 0.08, bx + 0.08, by - 0.015, by + 0.015))
    print()

    rclpy.init()
    node = rclpy.create_node("finger_watch")
    buf = Buffer()
    TransformListener(buf, node)
    state = {"grasp": "?"}
    node.create_subscription(String, "/avaa/grasp/state",
                             lambda m: state.__setitem__("grasp", m.data), 10)

    def where(frame):
        try:
            t = buf.lookup_transform(BASE, frame, rclpy.time.Time()).transform.translation
            return (t.x, t.y, t.z)
        except Exception:  # noqa: BLE001
            return None

    def book_in_base():
        # Re-read every sample. Computing it once and reusing it was wrong: the base
        # moves while the arm works -- 2.5 degrees of yaw right after being placed
        # square, and more as the arm swings out -- so a book position fixed at
        # startup drifts by tens of millimetres and the straddling verdict below
        # becomes a statement about where the book used to be.
        now, nowrpy = pose("tiago_pro")
        here, _ = pose(book_name)
        if now is None or here is None:
            return None
        yaw = nowrpy[2]
        dx, dy = here[0] - now[0], here[1] - now[1]
        return (dx * math.cos(-yaw) - dy * math.sin(-yaw),
                dx * math.sin(-yaw) + dy * math.cos(-yaw),
                here[2] - BASE_Z)

    print("%-11s %-22s %-22s %7s  %s" %
          ("state", "left tip", "right tip", "span", "straddling the book?"))
    seen = set()
    end = time.time() + 200
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
        left, right = where(LEFT), where(RIGHT)
        now = state["grasp"]
        if left is None or right is None:
            continue
        span = math.dist(left, right)
        # Does the book sit between the jaws, in the axis they close along?
        live = book_in_base()
        if live is not None:
            bx, by = live[0], live[1]
        lo, hi = sorted((left[1], right[1]))
        inside = lo < by - 0.015 and hi > by + 0.015
        deep = min(left[0], right[0]) > bx - 0.08
        verdict = "yes" if (inside and deep) else (
            "no: jaws at y %.3f..%.3f, book %.3f..%.3f" % (lo, hi, by - 0.015, by + 0.015)
            if not inside else "no: tips only reach x %.3f, face at %.3f"
            % (min(left[0], right[0]), bx - 0.08))
        key = (now, round(span, 3))
        if key not in seen:
            seen.add(key)
            print("%-11s (%.3f,%+.3f,%.3f) (%.3f,%+.3f,%.3f) %6.1fmm  %s" %
                  (now, left[0], left[1], left[2], right[0], right[1], right[2],
                   span * 1000, verdict), flush=True)
        if now in ("done", "failed"):
            break
        time.sleep(0.3)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
