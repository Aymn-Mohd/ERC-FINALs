#!/usr/bin/env python3
"""Measure the physical gap between the gripper fingertips as the gripper is driven.

The linkage joints are not published in /joint_states, so commanding the actuated joint
and reading joint states says nothing about whether the fingers actually move. TF carries
the real link poses, so the distance between the two fingertip links is the honest
measure of whether the gripper opens and closes.

A book is 3 cm thick, so the closed span must go below 0.03 m for a grasp to be possible.
"""
import math
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ACTUATED = "gripper_left_finger_joint"
TOPIC = "/gripper_left_controller_raw/joint_trajectory"
LEFT_TIP = "gripper_left_fingertip_left_link"
RIGHT_TIP = "gripper_left_fingertip_right_link"


def main():
    # Stop at --ros-args; everything after it belongs to rclpy, not to us.
    argv = sys.argv[1:]
    if "--ros-args" in argv:
        argv = argv[:argv.index("--ros-args")]
    targets = [float(v) for v in argv] or [0.0, 0.02, 0.04]

    rclpy.init()
    node = rclpy.create_node("gripper_span")
    buf = Buffer()
    TransformListener(buf, node)
    pub = node.create_publisher(JointTrajectory, TOPIC, 10)

    def pump(seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def span():
        try:
            tf = buf.lookup_transform(LEFT_TIP, RIGHT_TIP, rclpy.time.Time())
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
        t = tf.transform.translation
        return math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z), None

    pump(4.0)
    for target in targets:
        traj = JointTrajectory()
        traj.joint_names = [ACTUATED]
        point = JointTrajectoryPoint()
        point.positions = [target]
        point.time_from_start = Duration(sec=3, nanosec=0)
        traj.points = [point]
        for _ in range(3):
            pub.publish(traj)
            pump(0.15)
        pump(7.0)
        value, err = span()
        if value is None:
            print(f"{ACTUATED}={target:+.3f}  ->  TF unavailable ({err})", flush=True)
        else:
            note = "  (book is 0.030 m thick)" if value < 0.05 else ""
            print(f"{ACTUATED}={target:+.3f}  ->  fingertip span {value:.4f} m{note}",
                  flush=True)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
