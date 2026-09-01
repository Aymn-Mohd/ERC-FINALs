#!/usr/bin/env python3
"""Find which way head_2_joint points the camera.

The joint's axis is (0, 0, -1) and its range is asymmetric (-1.047 to +0.349), so the
sign that means "look down" is not obvious from the URDF alone. This drives the joint and
reports the camera's optical axis in base_link, where a negative z component means the
camera is looking downward.
"""
import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

CAMERA = "head_front_camera_depth_optical_frame"


def main():
    rclpy.init()
    node = rclpy.create_node("head_tilt_probe")
    buf = Buffer()
    TransformListener(buf, node)
    pub = node.create_publisher(JointTrajectory, "/head_controller/joint_trajectory", 10)

    def pump(seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def forward():
        tf = buf.lookup_transform("base_link", CAMERA, rclpy.time.Time())
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        # The optical frame's +Z is the view direction; express it in base_link.
        return np.array([2 * (x * z + w * y),
                         2 * (y * z - w * x),
                         1 - 2 * (x * x + y * y)])

    def tilt(value):
        traj = JointTrajectory()
        traj.joint_names = ["head_1_joint", "head_2_joint"]
        point = JointTrajectoryPoint()
        point.positions = [0.0, float(value)]
        point.time_from_start = Duration(sec=3)
        traj.points = [point]
        pub.publish(traj)
        pump(8.0)

    pump(3.0)
    print("head_2   optical forward (base_link)        pitch")
    for value in (0.0, -0.4, -0.8, 0.3):
        tilt(value)
        try:
            f = forward()
        except Exception as exc:  # noqa: BLE001
            print(f"{value:+.2f}   TF unavailable: {exc}")
            continue
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -f[2]))))
        arrow = "DOWN" if f[2] < -0.02 else ("UP" if f[2] > 0.02 else "level")
        print(f"{value:+.2f}   [{f[0]:+.3f} {f[1]:+.3f} {f[2]:+.3f}]   "
              f"{pitch:+6.1f} deg  {arrow}", flush=True)

    tilt(0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
