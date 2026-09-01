#!/usr/bin/env python3
"""Walk the reach into the shelf and name what stops it.

compute_cartesian_path reports a fraction and nothing else, so a reach that gets 2 per
cent of the way says only that something is wrong. /check_state_validity names the pair of
links in contact, so stepping along the line and asking at each point turns "blocked" into
"blocked at x=0.58 by arm_left_4_link against shelf_board_2".
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

GROUP = "arm_left_torso"
CHAIN_JOINTS = ["torso_lift_joint"] + ["arm_left_%d_joint" % i for i in range(1, 8)]


def main():
    face_x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.64
    book_y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.159
    height = float(sys.argv[3]) if len(sys.argv) > 3 else 0.731

    rclpy.init()
    node = rclpy.create_node("walk_reach")
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

    chain = ArmChain.from_urdf()
    index = {n: i for i, n in enumerate(js.name)}

    def check(values):
        state = JointState()
        state.name = list(js.name)
        state.position = list(js.position)
        for name, value in zip(CHAIN_JOINTS, values):
            if name in index:
                state.position[index[name]] = float(value)
        request = GetStateValidity.Request()
        request.robot_state = RobotState()
        request.robot_state.joint_state = state
        request.robot_state.is_diff = False
        request.group_name = GROUP
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

    print("walking from the pre-grasp to the book, at z=%.3f y=%+.3f" % (height, book_y))
    print("%8s  %-7s  %s" % ("x", "valid", "what is touching"))
    seed = None
    for x in np.arange(face_x - 0.15, face_x + 0.12, 0.02):
        solution = chain.ik([float(x), book_y, height], seed=seed,
                            approach=[1.0, 0.0, 0.0], closing=[0.0, 1.0, 0.0])
        if solution is None:
            print("%8.3f  %-7s  no IK" % (x, "-"))
            continue
        seed = solution
        valid, pairs = check(solution)
        summary = "" if valid else "; ".join("%s <-> %s" % p for p in pairs[:2])
        print("%8.3f  %-7s  %s" % (x, valid, summary))

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
