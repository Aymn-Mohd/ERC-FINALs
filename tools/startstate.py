#!/usr/bin/env python3
"""Is reach_fraction broken, or are the postures it is judging genuinely bad?

It returns 0.0 for every candidate while the same reach, requested without a start state,
plans a third of the way. Those cannot both be true, so this asks for the same path three
ways: with no start state, with the current state supplied explicitly, and with a
candidate posture. If the second disagrees with the first, the start state is the problem.
"""
import sys
import time

import rclpy
from geometry_msgs.msg import Pose, Quaternion
from sensor_msgs.msg import JointState

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402
from avaa_solution.moveit_client import MoveItClient  # noqa: E402

CHAIN_JOINTS = ["torso_lift_joint"] + ["arm_left_%d_joint" % i for i in range(1, 8)]


def main():
    face_x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.64
    book_y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.159
    height = float(sys.argv[3]) if len(sys.argv) > 3 else 0.731

    rclpy.init()
    node = rclpy.create_node("startstate_probe")
    node.set_parameters([rclpy.parameter.Parameter(
        "use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    latest = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: latest.__setitem__("js", m), 10)
    end = time.time() + 15
    while rclpy.ok() and time.time() < end and "js" not in latest:
        rclpy.spin_once(node, timeout_sec=0.2)
    js = latest.get("js")
    if js is None:
        print("no joint states")
        return 1
    print("joint state has %d joints" % len(js.name))

    client = MoveItClient("startstate_client")
    if not client.wait_until_ready(40.0):
        print("move_group not up")
        return 1

    target = Pose()
    target.position.x = face_x + 0.11
    target.position.y = book_y
    target.position.z = height
    target.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    print()
    print("A. no start state (uses whatever the arm is doing now)")
    code, fraction = client.straight_line([target], timeout=5.0)
    print("   fraction %.2f  (execution result ignored)" % fraction)

    print()
    print("B. the CURRENT state supplied explicitly -- should match A")
    names = list(js.name)
    values = list(js.position)
    print("   fraction %.2f" % client.reach_fraction([target], names, values))

    print()
    print("C. only the eight arm joints, current values")
    index = {n: i for i, n in enumerate(js.name)}
    subset = [js.position[index[n]] for n in CHAIN_JOINTS if n in index]
    print("   fraction %.2f" % client.reach_fraction([target], CHAIN_JOINTS, subset))

    print()
    print("D. a fresh IK posture for the pre-grasp, full state")
    chain = ArmChain.from_urdf()
    solution = chain.ik([face_x - 0.15, book_y, height],
                        approach=[1.0, 0.0, 0.0], closing=[0.0, 1.0, 0.0])
    if solution is None:
        print("   no IK")
    else:
        merged = list(values)
        for name, value in zip(CHAIN_JOINTS, solution):
            if name in index:
                merged[index[name]] = float(value)
        print("   fraction %.2f" % client.reach_fraction([target], names, merged))

    client.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
