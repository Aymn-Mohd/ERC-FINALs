#!/usr/bin/env python3
"""Why is every pre-grasp posture rejected?

The grasp reports "none of 12 postures for the pre-grasp is even collision free" while
the robot as it stands is clear and the pre-grasp point sits in front of the shelf boards.
So something about the posture, not the point, is in contact -- and the rejection path
only counts, it does not say what.

    python3 whyrejected.py <pre_x> <y> <z>
"""
import sys
import time

import numpy as np
import rclpy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402

CHAIN = ["torso_lift_joint"] + ["arm_left_%d_joint" % i for i in range(1, 8)]
SHOULDER_BASE_Z = 0.677


def main():
    pre_x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.431
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.175
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 1.061

    rclpy.init()
    node = rclpy.create_node("why_rejected")
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

    client = node.create_client(GetStateValidity, "/check_state_validity")
    if not client.wait_for_service(timeout_sec=15.0):
        print("no /check_state_validity")
        return 1
    index = {n: i for i, n in enumerate(js.name)}

    def check(values):
        state = JointState()
        state.name = list(js.name)
        state.position = list(js.position)
        for name, value in zip(CHAIN, values):
            if name in index:
                state.position[index[name]] = float(value)
        request = GetStateValidity.Request()
        request.robot_state = RobotState()
        request.robot_state.joint_state = state
        request.robot_state.is_diff = False
        request.group_name = "arm_left_torso"
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            return None, []
        pairs = []
        for contact in result.contacts:
            pair = (contact.contact_body_1, contact.contact_body_2)
            if pair not in pairs:
                pairs.append(pair)
        return bool(result.valid), pairs

    chain = ArmChain.from_urdf()
    ideal = float(np.clip(z - SHOULDER_BASE_Z + 0.25, 0.0, 0.35))
    print("pre-grasp [%.3f, %+.3f, %.3f], torso pinned near %.3f" % (pre_x, y, z, ideal))
    print()

    tally = {}
    for attempt in range(8):
        solution = chain.ik([pre_x, y, z], approach=[1.0, 0.0, 0.0],
                            closing=[0.0, 1.0, 0.0],
                            pin={"torso_lift_joint": (ideal, 0.10)})
        if solution is None:
            print("%2d: no IK" % attempt)
            continue
        valid, pairs = check(solution)
        summary = "clear" if valid else "; ".join("%s <-> %s" % p for p in pairs[:2])
        print("%2d: torso %.3f  %s" % (attempt, solution[0], summary))
        for pair in pairs:
            tally[pair] = tally.get(pair, 0) + 1

    if tally:
        print()
        print("most common contacts:")
        for pair, count in sorted(tally.items(), key=lambda kv: -kv[1])[:5]:
            print("   %2dx  %s <-> %s" % (count, pair[0], pair[1]))

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
