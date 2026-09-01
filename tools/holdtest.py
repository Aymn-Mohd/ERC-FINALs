#!/usr/bin/env python3
"""Can the arm reach and hold a given posture, given as long as it likes?

The reach stops short with no collision anywhere in Gazebo and no trajectory abort. That
leaves two possibilities that need separating: the trajectory is timed faster than the
arm can follow, or the posture is one the arm cannot hold at all. Commanding it directly
over forty seconds and watching where it settles tells them apart.

    python3 holdtest.py t a1 a2 a3 a4 a5 a6 a7
"""
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM = ["arm_left_%d_joint" % i for i in range(1, 8)]
CHAIN = ["torso_lift_joint"] + ARM
SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


def main():
    target = [float(v) for v in sys.argv[1:9]]
    if len(target) != 8:
        print(__doc__)
        return 1

    rclpy.init()
    node = rclpy.create_node("hold_test")
    node.set_parameters([rclpy.parameter.Parameter(
        "use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    arm = node.create_publisher(
        JointTrajectory, "/arm_left_controller/joint_trajectory", 10)
    torso = node.create_publisher(
        JointTrajectory, "/torso_controller/joint_trajectory", 10)
    state = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: state.__setitem__("js", m), SENSOR_QOS)

    deadline = time.time() + 10
    while time.time() < deadline and arm.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)

    def send(pub, names, values, seconds):
        traj = JointTrajectory()
        traj.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in values]
        point.time_from_start = Duration(sec=int(seconds), nanosec=0)
        traj.points = [point]
        pub.publish(traj)

    send(torso, ["torso_lift_joint"], target[:1], 30)
    send(arm, ARM, target[1:], 30)
    print("commanded over 30 s; watching for 55")
    print("%6s  %s" % ("t", "worst joint gap"))

    started = time.time()
    while time.time() - started < 55:
        rclpy.spin_once(node, timeout_sec=0.2)
        js = state.get("js")
        if js is None:
            continue
        index = {n: i for i, n in enumerate(js.name)}
        if not all(n in index for n in CHAIN):
            continue
        actual = [js.position[index[n]] for n in CHAIN]
        gaps = [a - b for a, b in zip(actual, target)]
        worst = max(range(len(gaps)), key=lambda i: abs(gaps[i]))
        elapsed = time.time() - started
        if abs(elapsed % 5.0) < 0.25:
            print("%6.0f  %-22s %+.3f   total %.3f"
                  % (elapsed, CHAIN[worst], gaps[worst],
                     sum(abs(g) for g in gaps)))
            time.sleep(0.3)

    print()
    print("if the total is still large after 55 s, the arm cannot hold this posture")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
