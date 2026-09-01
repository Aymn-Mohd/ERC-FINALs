#!/usr/bin/env python3
"""Is the arm failing to hold its posture because it is out of torque?

The reach stops 100 mm short with arm_left_4 a third of a radian from its last waypoint,
and each retry ends further away rather than nearer. That is the signature of an arm being
pushed rather than one lagging, and the same joint sagged onto its lower stop in an earlier
grasp. If its effort is at the URDF limit then the posture is beyond what the actuator can
hold and no amount of planning will fix it.
"""
import sys
import time
import xml.etree.ElementTree as ET

import rclpy
from sensor_msgs.msg import JointState

URDF = "/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf"
WATCH = ["torso_lift_joint"] + ["arm_left_%d_joint" % i for i in range(1, 8)]


def main():
    limits = {}
    for joint in ET.parse(URDF).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("effort"):
            limits[joint.get("name")] = float(limit.get("effort"))

    rclpy.init()
    node = rclpy.create_node("effort_probe")
    node.set_parameters([rclpy.parameter.Parameter(
        "use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    samples = []
    node.create_subscription(JointState, "/joint_states",
                             lambda m: samples.append(m), 10)

    end = time.time() + 12
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not samples:
        print("no joint states")
        return 1

    print("%-22s %10s %10s %10s" % ("joint", "position", "effort", "limit"))
    latest = samples[-1]
    index = {n: i for i, n in enumerate(latest.name)}
    for name in WATCH:
        if name not in index:
            continue
        i = index[name]
        effort = latest.effort[i] if i < len(latest.effort) else float("nan")
        limit = limits.get(name, float("nan"))
        flag = ""
        if limit == limit and effort == effort and abs(effort) >= 0.9 * limit:
            flag = "   SATURATED"
        print("%-22s %10.3f %10.2f %10.1f%s"
              % (name, latest.position[i], effort, limit, flag))

    print()
    print("%d samples over %.0f s" % (len(samples), 12.0))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
