#!/usr/bin/env python3
"""Check the URDF forward kinematics against what the simulator actually does.

Commands a series of arm postures, waits for them to settle, and compares the predicted
gripper position with the one TF reports. Two different things show up in the residual and
it is worth separating them:

  * a constant or structural error means the FK model is wrong
  * an error that grows with how far the arm is extended is the arm sagging under gravity,
    which the URDF cannot predict

That distinction decides whether joint targets can be trusted straight from IK or have to
be corrected against measurement.
"""
import math
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402

BASE = "base_link"
TIP = "gripper_left_grasping_link"
ARM_TOPIC = "/arm_left_controller/joint_trajectory"
TORSO_TOPIC = "/torso_controller/joint_trajectory"

# (torso, arm1..arm7)
POSTURES = [
    ("zero", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("torso up", [0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("tucked", [0.0, -0.5, -2.4, 0.0, -2.4, 0.0, 0.0, 0.0]),
    ("elbow bent", [0.15, 0.3, -0.8, 0.0, -1.2, 0.0, 0.0, 0.0]),
    ("raised", [0.20, 0.2, -1.4, 0.3, -0.9, 0.0, 0.4, 0.0]),
]


def main():
    rclpy.init()
    node = rclpy.create_node("validate_fk")
    buf = Buffer()
    TransformListener(buf, node)
    arm = node.create_publisher(JointTrajectory, ARM_TOPIC, 10)
    torso = node.create_publisher(JointTrajectory, TORSO_TOPIC, 10)
    chain = ArmChain.from_urdf()

    print(f"chain joints: {chain.joint_names}\n")

    def pump(seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def send(pub, names, values, seconds):
        traj = JointTrajectory()
        traj.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in values]
        point.time_from_start = Duration(sec=int(seconds), nanosec=0)
        traj.points = [point]
        for _ in range(3):
            pub.publish(traj)
            pump(0.15)

    arm_names = [f"arm_left_{i}_joint" for i in range(1, 8)]

    # Read back where the joints actually are. Comparing FK of the *commanded* values
    # against TF conflates two things: whether the kinematic model is right, and whether
    # the controller got there. The torso is slow (0.035 m/s) and is the likely laggard,
    # so the model has to be judged on the joint values the robot actually holds.
    actual = {}

    def on_js(msg):
        for name, pos in zip(msg.name, msg.position):
            actual[name] = pos

    node.create_subscription(JointState, "/joint_states", on_js, 10)
    pump(3.0)

    print(f"{'posture':>12}  {'err vs commanded':>17}  {'err vs actual':>14}  "
          f"{'torso cmd':>9}  {'torso act':>9}")
    rows = []
    for label, values in POSTURES:
        send(torso, ["torso_lift_joint"], [values[0]], 12)
        send(arm, arm_names, values[1:], 5)
        pump(18.0)

        try:
            tf = buf.lookup_transform(BASE, TIP, rclpy.time.Time())
        except Exception as exc:  # noqa: BLE001
            print(f"{label:>12}  TF unavailable: {exc}", flush=True)
            continue
        t = tf.transform.translation
        measured = np.array([t.x, t.y, t.z])

        commanded_err = float(np.linalg.norm(chain.position(values) - measured))

        held = [actual.get(name, values[i])
                for i, name in enumerate(chain.joint_names)]
        actual_err = float(np.linalg.norm(chain.position(held) - measured))

        rows.append((label, commanded_err, actual_err))
        print(f"{label:>12}  {commanded_err:>17.4f}  {actual_err:>14.4f}  "
              f"{values[0]:>9.3f}  {held[0]:>9.3f}", flush=True)

    if rows:
        cmd_errors = [r[1] for r in rows]
        act_errors = [r[2] for r in rows]
        print(f"\nvs commanded joints: max={max(cmd_errors):.4f} "
              f"mean={sum(cmd_errors)/len(cmd_errors):.4f} m")
        print(f"vs actual joints   : max={max(act_errors):.4f} "
              f"mean={sum(act_errors)/len(act_errors):.4f} m")
        if max(act_errors) < 0.01:
            print("\nThe kinematic model is correct. Residual against commanded values is "
                  "the controller not having arrived, not a modelling error.")
        else:
            print("\nModel error remains even against the joint values actually held -- "
                  "the chain itself is wrong somewhere.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
