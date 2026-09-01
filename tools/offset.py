#!/usr/bin/env python3
"""Exactly where do the fingers close, relative to the frame the IK aims?

The grasp solves IK to gripper_left_grasping_link. Whether that frame is where the jaws
actually meet decides everything: the last run pushed a book 50 mm straight back into the
shelf, which is precisely the grasp depth, so something solid is arriving at the face
instead of passing either side of it.

Rather than model the finger linkage, this asks TF for all three frames at the same
instant, at whatever pose the arm is in, and reports the offsets along the gripper axes.
It sweeps the gripper open and closed so the dependence on opening is visible too.
"""
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

BASE = "base_link"
GRASP = "gripper_left_grasping_link"
LEFT = "gripper_left_fingertip_left_link"
RIGHT = "gripper_left_fingertip_right_link"
PALM = "gripper_left_base_link"
GRIPPER_TOPIC = "/gripper_left_controller_raw/joint_trajectory"
FINGER = "gripper_left_finger_joint"

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


def quat_matrix(q):
    xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
    xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
    wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def main():
    rclpy.init()
    node = rclpy.create_node("offset")
    buf = Buffer()
    TransformListener(buf, node)
    pub = node.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
    state = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: state.__setitem__("js", m), SENSOR_QOS)

    for _ in range(60):
        rclpy.spin_once(node, timeout_sec=0.1)

    def look(frame):
        try:
            return buf.lookup_transform(BASE, frame, rclpy.time.Time()).transform
        except Exception as exc:  # noqa: BLE001
            print("  no transform for %s: %s" % (frame, exc))
            return None

    for opening in (0.040, 0.020, 0.000):
        traj = JointTrajectory()
        traj.joint_names = [FINGER]
        point = JointTrajectoryPoint()
        point.positions = [float(opening)]
        point.time_from_start = Duration(sec=2, nanosec=0)
        traj.points = [point]
        pub.publish(traj)
        end = time.time() + 5
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)

        grasp = look(GRASP)
        left, right, palm = look(LEFT), look(RIGHT), look(PALM)
        if not all((grasp, left, right, palm)):
            continue

        origin = np.array([grasp.translation.x, grasp.translation.y, grasp.translation.z])
        rotation = quat_matrix(grasp.rotation)
        approach, closing = rotation[:, 0], rotation[:, 1]

        def offsets(t, label):
            p = np.array([t.translation.x, t.translation.y, t.translation.z])
            d = p - origin
            print("    %-14s along approach %+7.1f mm   along closing %+7.1f mm"
                  % (label, np.dot(d, approach) * 1000, np.dot(d, closing) * 1000))
            return p

        js = state.get("js")
        actual = (js.position[js.name.index(FINGER)]
                  if js and FINGER in js.name else float("nan"))
        print("gripper commanded %.3f, joint reads %.4f" % (opening, actual))
        lp = offsets(left, "left tip")
        rp = offsets(right, "right tip")
        offsets(palm, "palm")
        print("    tip separation %.1f mm" % (np.linalg.norm(lp - rp) * 1000))
        midpoint = (lp + rp) / 2.0
        print("    JAW MIDPOINT is %+.1f mm along approach from the grasping frame"
              % (np.dot(midpoint - origin, approach) * 1000))
        print()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
