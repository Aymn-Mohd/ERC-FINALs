#!/usr/bin/env python3
"""Measure where the gripper can actually be placed, as the torso extends.

    python3 reach_envelope.py [torso_positions...]

The four stocked shelf rows sit at Z = 1.577, 1.247, 0.917 and 0.587 m in the world --
0.99 m from top to bottom. The torso has only 0.35 m of travel, so whether every row is
reachable, and at what torso extension, is a question that has to be answered before any
grasp planning. This drives the torso to each position with the arm extended and reports
the gripper pose in base_footprint.

Base_footprint sits on the floor, so gripper z here is directly comparable to the world
row heights.
"""
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

BASE = "base_footprint"
# The URDF joint is gripper_left_grasping_frame_joint but its child link -- and so the
# TF frame -- is gripper_left_grasping_link.
GRIPPER = "gripper_left_grasping_link"
ARM_TOPIC = "/arm_left_controller/joint_trajectory"
TORSO_TOPIC = "/torso_controller/joint_trajectory"

# Arm extended forward: the spawn pose, which reaches furthest.
ARM_EXTENDED = [0.0] * 7

ROW_HEIGHTS = {1: 1.577, 2: 1.247, 3: 0.917, 4: 0.587}


def main():
    argv = sys.argv[1:]
    if "--ros-args" in argv:
        argv = argv[:argv.index("--ros-args")]
    positions = [float(v) for v in argv] or [0.0, 0.15, 0.35]

    rclpy.init()
    node = rclpy.create_node("reach_envelope")
    buf = Buffer()
    TransformListener(buf, node)
    arm = node.create_publisher(JointTrajectory, ARM_TOPIC, 10)
    torso = node.create_publisher(JointTrajectory, TORSO_TOPIC, 10)

    def pump(seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def send(pub, names, values, seconds):
        traj = JointTrajectory()
        traj.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in values]
        point.time_from_start = Duration(sec=int(seconds), nanosec=0)
        traj.points = [point]
        for _ in range(3):
            pub.publish(traj)
            pump(0.15)

    pump(3.0)
    arm_names = [f"arm_left_{i}_joint" for i in range(1, 8)]
    send(arm, arm_names, ARM_EXTENDED, 4)
    pump(6.0)

    print(f"{'torso':>7}  {'gripper x':>10}  {'gripper y':>10}  {'gripper z':>10}")
    results = []
    for pos in positions:
        # The torso moves at 0.035 m/s, so allow generous time for a full stroke.
        send(torso, ["torso_lift_joint"], [pos], 12)
        pump(16.0)
        try:
            tf = buf.lookup_transform(BASE, GRIPPER, rclpy.time.Time())
        except Exception as exc:  # noqa: BLE001
            print(f"{pos:>7.3f}  TF unavailable: {exc}", flush=True)
            continue
        t = tf.transform.translation
        results.append((pos, t.x, t.y, t.z))
        print(f"{pos:>7.3f}  {t.x:>10.3f}  {t.y:>10.3f}  {t.z:>10.3f}", flush=True)

    if len(results) >= 2:
        lo, hi = min(r[3] for r in results), max(r[3] for r in results)
        print(f"\ngripper height range with arm extended: {lo:.3f} .. {hi:.3f} m")
        print("shelf rows (world z, base_footprint is on the floor so directly comparable):")
        for row, z in ROW_HEIGHTS.items():
            if lo <= z <= hi:
                verdict = "reachable at this arm pose"
            elif z > hi:
                verdict = f"ABOVE range by {z - hi:.3f} m - needs a different arm pose"
            else:
                verdict = f"BELOW range by {lo - z:.3f} m - needs a different arm pose"
            print(f"  row {row}: z={z:.3f}  {verdict}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
