#!/usr/bin/env python3
"""Watch the arm during a grasp, not just the state machine.

    python3 grasp_probe.py <book_colour>

The targeting is now right: measured against Gazebo the robot ends square, the book is a
few degrees off the nose, the row is correct and the grasp point lands inside the book.
The book still does not move, so the question is no longer where the arm was told to go
but whether it went there and whether the fingers closed on anything.

Run this with the robot already in front of the shelf. It starts perception and the grasp
controller and then samples, twice a second:

    the grasp state
    every arm joint, commanded against actual
    where forward kinematics puts the gripper, against where the grasp asked for it
    the finger opening
    the target book pose from Gazebo

Anything that is commanded and not reached shows up as a standing gap in one column.
"""
import math
import re
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import String

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402

ARM_JOINTS = ["arm_left_%d_joint" % i for i in range(1, 8)]
CHAIN_JOINTS = ["torso_lift_joint"] + ARM_JOINTS
FINGER = "gripper_left_finger_joint"

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


def gz_pose(model):
    out = subprocess.run(["gz", "model", "-m", model, "-p"],
                         capture_output=True, text=True, timeout=25).stdout
    lines = [l.strip() for l in out.splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
            try:
                return [float(v) for v in line.strip("[]").split()]
            except ValueError:
                return None
    return None


def target_book(colour):
    """The book of ``colour`` nearest the robot, by Gazebo truth."""
    listing = subprocess.run(["gz", "model", "--list"], capture_output=True,
                             text=True, timeout=25).stdout
    names = [l.strip(" -") for l in listing.splitlines()
             if "book_col" in l and colour in l]
    robot = gz_pose("tiago_pro")
    best = None
    for name in names:
        p = gz_pose(name)
        if p is None:
            continue
        d = math.hypot(p[0] - robot[0], p[1] - robot[1])
        if best is None or d < best[0]:
            best = (d, name, p)
    return best


def run_node(executable, extra=(), log="/tmp/probe_node.log"):
    cmd = ("source /opt/erc_ws/install/setup.bash && "
           "exec python3 -u /opt/erc_ws/install/avaa_solution/lib/avaa_solution/"
           "%s --ros-args -p use_sim_time:=true %s > %s 2>&1"
           % (executable, " ".join(extra), log))
    return subprocess.Popen(["/entrypoint.sh", "bash", "-c", cmd])


def main():
    colour = sys.argv[1] if len(sys.argv) > 1 else "red"
    chain = ArmChain.from_urdf()

    found = target_book(colour)
    if found is None:
        print("no %s book found" % colour)
        return
    distance, name, truth = found
    print("target by ground truth: %s at %s, %.2f m away"
          % (name, [round(v, 3) for v in truth], distance))

    rclpy.init()
    node = rclpy.create_node("grasp_probe")
    state = {"joints": None, "grasp": "?"}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: state.__setitem__("joints", m), SENSOR_QOS)
    node.create_subscription(String, "/avaa/grasp/state",
                             lambda m: state.__setitem__("grasp", m.data), 10)

    perception = run_node("perception",
                          ["-p shelf_column_number:=3", "-p book_colour:=%s" % colour,
                           "-p save_images:=false"], "/tmp/probe_perception.log")
    time.sleep(6)
    grasp = run_node("grasp", log="/tmp/probe_grasp.log")

    print()
    print("%-11s %7s %7s %7s   %8s %8s %8s   %6s" %
          ("state", "torso", "arm1..7", "gap", "fk x", "fk y", "fk z", "finger"))

    wanted = None
    start = time.time()
    last = ""
    try:
        while time.time() - start < 150:
            rclpy.spin_once(node, timeout_sec=0.2)
            js = state["joints"]
            if js is None:
                continue

            index = {n: i for i, n in enumerate(js.name)}
            if not all(n in index for n in CHAIN_JOINTS):
                continue
            actual = [js.position[index[n]] for n in CHAIN_JOINTS]
            finger = js.position[index[FINGER]] if FINGER in index else float("nan")
            fk = chain.fk(actual)[:3, 3]

            if wanted is None:
                try:
                    text = open("/tmp/probe_grasp.log", errors="ignore").read()
                except OSError:
                    text = ""
                m = re.search(r"book at x=([-\d.]+) y=([-\d.]+)", text)
                r = re.search(r"row \d+ at z=([-\d.]+)", text)
                if m and r:
                    wanted = np.array([float(m.group(1)) + 0.05,
                                       float(m.group(2)), float(r.group(1))])
                    print("grasp point asked for: %s" % np.round(wanted, 3).tolist())

            gap = "" if wanted is None else "%.3f" % float(np.linalg.norm(fk - wanted))
            line = ("%-11s %7.3f %s   %8.3f %8.3f %8.3f   %6.4f" %
                    (state["grasp"], actual[0],
                     " ".join("%+.2f" % v for v in actual[1:]),
                     fk[0], fk[1], fk[2], finger))
            line = line + "  gap=" + gap
            if line[:11] != last[:11] or (time.time() * 2) % 4 < 0.5:
                print(line, flush=True)
            last = line
            if state["grasp"] in ("done", "failed"):
                break
            time.sleep(0.4)
    finally:
        for proc in (perception, grasp):
            proc.terminate()
        subprocess.run(["pkill", "-f", "avaa_solution/lib"], capture_output=True)
        node.destroy_node()
        rclpy.shutdown()

    after = gz_pose(name)
    moved = math.dist(truth[:3], after[:3]) if after else float("nan")
    print()
    print("%s moved %.3f m" % (name, moved))


if __name__ == "__main__":
    main()
