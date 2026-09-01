#!/usr/bin/env python3
"""Does forward kinematics agree with TF about where the gripper is?

Every arrival check in the grasp controller is FK over /joint_states. If that disagrees
with the simulator, the controller is chasing a number rather than the arm, and a reach
reported 100 mm off might be arriving perfectly.
"""
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402

CHAIN_JOINTS = ["torso_lift_joint"] + ["arm_left_%d_joint" % i for i in range(1, 8)]


def main():
    rclpy.init()
    node = rclpy.create_node("fk_check")
    node.set_parameters([rclpy.parameter.Parameter(
        "use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    buf = Buffer()
    TransformListener(buf, node)
    latest = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: latest.__setitem__("js", m), 10)

    end = time.time() + 20
    while rclpy.ok() and time.time() < end and "js" not in latest:
        rclpy.spin_once(node, timeout_sec=0.2)
    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.1)

    js = latest.get("js")
    if js is None:
        print("no joint states")
        return 1
    index = {n: i for i, n in enumerate(js.name)}
    missing = [n for n in CHAIN_JOINTS if n not in index]
    if missing:
        print("joint state is missing %s" % missing)
        return 1
    values = [js.position[index[n]] for n in CHAIN_JOINTS]

    chain = ArmChain.from_urdf()
    fk = chain.position(values)

    try:
        tf = buf.lookup_transform("base_link", "gripper_left_grasping_link",
                                  rclpy.time.Time())
    except Exception as exc:  # noqa: BLE001
        print("no TF: %s" % exc)
        return 1
    t = tf.transform.translation
    truth = np.array([t.x, t.y, t.z])

    print("joints  %s" % [round(v, 3) for v in values])
    print("FK      %s" % np.round(fk, 4).tolist())
    print("TF      %s" % np.round(truth, 4).tolist())
    print("differ  %s  (%.1f mm)"
          % (np.round(fk - truth, 4).tolist(), float(np.linalg.norm(fk - truth)) * 1000))

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
