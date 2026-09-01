#!/usr/bin/env python3
"""Fold both arms in, and wait until they are actually folded.

The robot spawns with its arms straight out: all joints at zero puts the left gripper at
(0.984, 0.493, 0.229) in base_link. Teleporting the base to a grasping distance from that
pose plants the outstretched arm inside the shelf, and every subsequent command jams
against base_link_shelf_collision before it can start.

The real pipeline never sees this because the approach tucks before it drives. Test
fixtures have to do the same, and they have to do it *before* the base is placed.
"""
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

TUCK_POSE = [2.1521, 0.3824, 1.2785, -2.1517, 0.8325, 0.1926, 1.3944]
TUCK_TORSO = 0.15
# The right arm is never used, but it still has to be out of the way, and mirroring
# the left tuck by flipping two joints does not do that: it leaves arm_right_4_link
# at x=0.491, which is inside the shelf once the base is close enough to grasp from.
# MoveIt then reports the robot in collision and refuses every left-arm goal, so the
# grasp fails for a reason that has nothing to do with the arm doing the work.
#
# Found the same way as the left tuck, by sampling against /check_state_validity with
# the shelf in the scene at grasping distance. See tools/find_right_tuck.py.
#
# Scored on where the links actually end up, not on how small the joint angles are.
# Scoring on the angles picks postures close to all-zeros, and all-zeros is the arm
# stretched straight out: the first answer put the right gripper 0.88 m forward,
# inside the shelf, against shelf_back. This one reaches 0.204 m forward and is
# checked at both torso heights, because raising the torso for the top rows takes
# the whole upper body with it.
RIGHT_TUCK = [-0.7194, -2.2867, -0.5064, 0.5221, 2.3399, 1.0503, 1.9772]
SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


def main():
    settle = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    rclpy.init()
    node = rclpy.create_node("tuck_arm")
    state = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: state.__setitem__("js", m), SENSOR_QOS)

    publishers = {
        side: node.create_publisher(
            JointTrajectory, "/arm_%s_controller/joint_trajectory" % side, 10)
        for side in ("left", "right")
    }

    # Wait for the controllers before publishing: a trajectory sent before the
    # subscription is matched is dropped without a word.
    deadline = time.time() + 15
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if all(p.get_subscription_count() > 0 for p in publishers.values()):
            break
    for side, pub in publishers.items():
        if pub.get_subscription_count() == 0:
            print("nothing listening on the %s arm controller" % side)

    # The torso is part of the tuck. Without it the folded arm sits inside base_link
    # and MoveIt will not plan from that state at all.
    torso_pub = node.create_publisher(
        JointTrajectory, "/torso_controller/joint_trajectory", 10)
    torso = JointTrajectory()
    torso.joint_names = ["torso_lift_joint"]
    lift = JointTrajectoryPoint()
    lift.positions = [float(TUCK_TORSO)]
    lift.time_from_start = Duration(sec=20, nanosec=0)
    torso.points = [lift]
    torso_pub.publish(torso)

    for side, pub in publishers.items():
        traj = JointTrajectory()
        traj.joint_names = ["arm_%s_%d_joint" % (side, i) for i in range(1, 8)]
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in
                           (RIGHT_TUCK if side == "right" else TUCK_POSE)]
        point.time_from_start = Duration(sec=20, nanosec=0)
        traj.points = [point]
        pub.publish(traj)

    end = time.time() + settle
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    js = state.get("js")
    if js:
        names = ["arm_left_%d_joint" % i for i in range(1, 8)]
        if all(n in js.name for n in names):
            actual = [js.position[js.name.index(n)] for n in names]
            gap = sum(abs(a - b) for a, b in zip(actual, TUCK_POSE))
            print("left arm tucked to within %.3f rad total" % gap)
            if gap > 0.5:
                print("  WARNING: not tucked; anything placed near the shelf will jam")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
