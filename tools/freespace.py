#!/usr/bin/env python3
"""Can the arm hold a grasp posture at all, with nothing in front of it?

During a grasp, arm_left_1_joint was measured 2.69 rad from its commanded value and
drifting further away, while every other joint converged. Joint limits are not the reason
-- the whole solution sits well inside them -- so something is physically resisting.

This commands the same posture twice: once where the robot stands, and once with the base
moved out into open floor. If it holds in the open and not at the shelf, the shelf is in
the way. If it holds in neither, the arm is fighting itself or its own body.
"""
import math
import subprocess
import time

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ["arm_left_%d_joint" % i for i in range(1, 8)]
POSTURE = [2.304, 0.093, -0.852, -1.487, 0.534, 1.149, -1.472]
TORSO = 0.284
SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


def teleport(x, y):
    subprocess.run(
        ["gz", "service", "-s", "/world/erc_world/set_pose",
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "--req",
         'name: "tiago_pro", position: {x: %f, y: %f, z: 0.0}, '
         'orientation: {x: 0, y: 0, z: 0, w: 1}' % (x, y)],
        capture_output=True, timeout=20)


def main():
    rclpy.init()
    node = rclpy.create_node("freespace")
    arm = node.create_publisher(JointTrajectory, "/arm_left_controller/joint_trajectory", 10)
    torso = node.create_publisher(JointTrajectory, "/torso_controller/joint_trajectory", 10)
    stop = node.create_publisher(Twist, "/cmd_vel", 10)
    state = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: state.__setitem__("js", m), SENSOR_QOS)
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)

    def send(pub, names, values, seconds):
        traj = JointTrajectory()
        traj.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in values]
        point.time_from_start = Duration(sec=int(seconds), nanosec=0)
        traj.points = [point]
        pub.publish(traj)

    def read():
        for _ in range(40):
            rclpy.spin_once(node, timeout_sec=0.1)
            js = state.get("js")
            if js and all(n in js.name for n in ARM_JOINTS):
                return [js.position[js.name.index(n)] for n in ARM_JOINTS]
        return None

    for label, place in (("at the shelf (x=2.0)", (2.0, -0.2)),
                         ("in open floor (x=0.5)", (0.5, 0.0))):
        for _ in range(20):
            stop.publish(Twist())
            rclpy.spin_once(node, timeout_sec=0.02)
        teleport(*place)
        time.sleep(2)

        # Start from tuck so both attempts travel the same distance.
        send(arm, ARM_JOINTS, [-0.5, -2.4, 0.0, -2.4, 0.0, 0.0, 0.0], 20)
        end = time.time() + 25
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)

        send(torso, ["torso_lift_joint"], [TORSO], 12)
        send(arm, ARM_JOINTS, POSTURE, 25)
        end = time.time() + 40
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)

        actual = read()
        print("=== %s" % label)
        if actual is None:
            print("   no joint states")
            continue
        worst = max(range(7), key=lambda i: abs(actual[i] - POSTURE[i]))
        for i, name in enumerate(ARM_JOINTS):
            gap = actual[i] - POSTURE[i]
            print("   %-18s want %+7.3f  got %+7.3f  off %+7.3f%s"
                  % (name, POSTURE[i], actual[i], gap,
                     "   <-- worst" if i == worst else ""))
        print("   sum |gap| = %.3f" % sum(abs(a - b) for a, b in zip(actual, POSTURE)))
        print()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
