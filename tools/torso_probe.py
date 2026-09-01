#!/usr/bin/env python3
"""Does the torso still undershoot?

grasp_node adds TORSO_BIAS = 0.028 to every torso command, to compensate an undershoot
measured earlier. In the last grasp the IK asked for 0.136, the node commanded 0.164, and
the joint settled at 0.164 -- so the compensation is now a pure 28 mm error, and the
gripper arrived 25 mm above the book.

This commands a series of heights with no compensation at all and reads back where the
joint actually settles.
"""
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

TOPIC = "/torso_controller/joint_trajectory"
JOINT = "torso_lift_joint"
SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


def main():
    rclpy.init()
    node = rclpy.create_node("torso_probe")
    pub = node.create_publisher(JointTrajectory, TOPIC, 10)
    state = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: state.__setitem__("js", m), SENSOR_QOS)

    def read():
        for _ in range(40):
            rclpy.spin_once(node, timeout_sec=0.1)
            js = state.get("js")
            if js and JOINT in js.name:
                return js.position[js.name.index(JOINT)]
        return float("nan")

    print("%10s %10s %10s" % ("commanded", "settled", "error"))
    errors = []
    for target in (0.10, 0.15, 0.20, 0.25, 0.30, 0.15):
        traj = JointTrajectory()
        traj.joint_names = [JOINT]
        point = JointTrajectoryPoint()
        point.positions = [float(target)]
        point.time_from_start = Duration(sec=4, nanosec=0)
        traj.points = [point]
        pub.publish(traj)

        deadline = time.time() + 9
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        settled = read()
        error = settled - target
        errors.append(error)
        print("%10.3f %10.3f %+10.4f" % (target, settled, error))

    print()
    mean = sum(errors) / len(errors)
    print("mean error %+.4f m" % mean)
    print("TORSO_BIAS is 0.028; a mean error near zero means it should be removed")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
