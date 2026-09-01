#!/usr/bin/env python3
"""Find a folded arm posture that is genuinely collision-free.

The tuck this solution has used from the beginning puts arm_left_2 through arm_left_5
against torso_base_link and torso_lift_link. Gazebo never complained -- self-collision is
not checked there -- so it went unnoticed until MoveIt refused to plan from it at all:
"Motion planning start tree could not be initialized".

The fix is a different posture, not a disabled check. Switching those pairs off in the
SRDF would let the planner route the arm through the robot's own torso, which is a worse
problem than the one being solved.

Searches for a posture that is valid, compact (gripper close to the body so the base can
drive and the laser can see), and low (out of the camera's view of the shelf).
"""
import math
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
ARM_JOINTS = ["arm_left_%d_joint" % i for i in range(1, 8)]
CHAIN_JOINTS = ["torso_lift_joint"] + ARM_JOINTS
CURRENT_TUCK = [0.15, -0.5, -2.4, 0.0, -2.4, 0.0, 0.0, 0.0]


def main():
    rclpy.init()
    node = rclpy.create_node("find_tuck")
    node.set_parameters([rclpy.parameter.Parameter(
        "use_sim_time", rclpy.Parameter.Type.BOOL, True)])

    latest = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: latest.__setitem__("js", m), 10)
    deadline = time.time() + 15
    while rclpy.ok() and time.time() < deadline and "js" not in latest:
        rclpy.spin_once(node, timeout_sec=0.2)

    client = node.create_client(GetStateValidity, "/check_state_validity")
    if not client.wait_for_service(timeout_sec=15.0):
        print("no /check_state_validity")
        return 1

    template = latest.get("js")
    if template is None:
        print("no joint states")
        return 1

    chain = ArmChain.from_urdf()
    limits = chain.limits

    def valid(values):
        state = JointState()
        state.name = list(template.name)
        state.position = list(template.position)
        index = {n: i for i, n in enumerate(state.name)}
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
        return None if result is None else bool(result.valid)

    print("current tuck valid: %s" % valid(CURRENT_TUCK))
    print()
    print("searching for a folded posture that is collision free...")

    rng = np.random.default_rng(0)
    best = None
    checked = 0
    started = time.time()
    while time.time() - started < 300 and checked < 900:
        # Bias towards folded: joints near the middle of their range, torso low.
        values = [0.15]
        for lo, hi in limits[1:]:
            centre = 0.5 * (lo + hi)
            span = 0.35 * (hi - lo)
            values.append(float(np.clip(rng.normal(centre, span), lo, hi)))
        checked += 1
        if valid(values) is not True:
            continue
        margin = min(min(v - lo, hi - v) for v, (lo, hi) in zip(values, limits))
        if margin < 0.15:
            # A joint on its stop has nowhere to correct to, and this arm sags onto its
            # stops under load: one grasp held the gripper 95 mm short with arm_left_4
            # pinned at its lower limit while every other joint sat exactly on target.
            continue
        tip = chain.position(values)
        # Compact and low: close to the base in x and y, and out of the camera view.
        reach = math.hypot(float(tip[0]), float(tip[1]))
        score = (reach + 0.5 * max(0.0, float(tip[2]) - 0.6)
                 + 0.3 * max(0.0, 0.4 - margin))
        if best is None or score < best[0]:
            best = (score, list(values), tip)
            print("   candidate: reach %.3f m, margin %.2f rad, tip %s"
                  % (reach, margin, np.round(tip, 3).tolist()), flush=True)

    print()
    if best is None:
        print("found nothing valid in %d samples" % checked)
        return 1
    score, values, tip = best
    print("best of %d samples:" % checked)
    print("   torso %.3f  arm %s" % (values[0], [round(v, 3) for v in values[1:]]))
    print("   gripper at %s" % np.round(tip, 3).tolist())
    print()
    print("as an SRDF group_state:")
    for name, value in zip(CHAIN_JOINTS, values):
        print('    <joint name="%s" value="%.4f"/>' % (name, value))
    print()
    print("as a Python list:")
    print("TUCK_POSE = [%s]" % ", ".join("%.4f" % v for v in values[1:]))

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
