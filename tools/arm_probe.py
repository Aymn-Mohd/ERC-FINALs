#!/usr/bin/env python3
"""Does the arm reach what it is commanded, and does it hold there?

During a grasp the distance from the gripper to its planned point was measured growing:
389, 398, 408, 423, 441 mm. Growing, not shrinking. So either the trajectory is never
executed, or it is executed and then not held.

This commands the arm to a few reachable poses and reports, per joint, the command, what
the joint reached after the move, and what it is doing several seconds later. A joint that
reaches and then drifts is being let go; one that never reaches is being refused.
"""
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ["arm_left_%d_joint" % i for i in range(1, 8)]
TOPIC = "/arm_left_controller/joint_trajectory"
SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)

POSES = [
    ("tuck", [-0.5, -2.4, 0.0, -2.4, 0.0, 0.0, 0.0]),
    ("half out", [0.5, -1.4, 0.0, -1.6, 0.0, 0.4, 0.0]),
    ("reaching", [1.2, -0.8, -0.3, -1.4, 0.2, 0.8, -0.5]),
]


def main():
    rclpy.init()
    node = rclpy.create_node("arm_probe")
    pub = node.create_publisher(JointTrajectory, TOPIC, 10)
    state = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: state.__setitem__("js", m), SENSOR_QOS)

    def read():
        for _ in range(40):
            rclpy.spin_once(node, timeout_sec=0.1)
            js = state.get("js")
            if js and all(n in js.name for n in ARM_JOINTS):
                return [js.position[js.name.index(n)] for n in ARM_JOINTS]
        return None

    for label, target in POSES:
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in target]
        point.time_from_start = Duration(sec=4, nanosec=0)
        traj.points = [point]
        pub.publish(traj)

        end = time.time() + 8
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        reached = read()

        end = time.time() + 8
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        later = read()

        print("=== %s" % label)
        print("%-18s %8s %8s %8s %9s" %
              ("joint", "command", "at 8s", "at 16s", "drift mm/deg"))
        for i, name in enumerate(ARM_JOINTS):
            error = reached[i] - target[i] if reached else float("nan")
            drift = later[i] - reached[i] if reached and later else float("nan")
            flag = ""
            if abs(error) > 0.05:
                flag = "  NOT REACHED"
            if abs(drift) > 0.02:
                flag += "  DRIFTING"
            print("%-18s %+8.3f %+8.3f %+8.3f %+9.3f%s"
                  % (name, target[i], reached[i] if reached else float("nan"),
                     later[i] if later else float("nan"), drift, flag))
        print()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
