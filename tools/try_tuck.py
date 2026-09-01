#!/usr/bin/env python3
"""Command a candidate arm tuck and measure whether it is actually safe to drive with.

    python3 try_tuck.py j1 j2 j3 j4 j5 j6 j7

Sends the pose to both arms (mirrored on joint 1 and 3 for the right arm), then reports:

  * how far each arm link reaches in front of base_footprint -- what would hit the shelf
  * how far it sticks out sideways -- what would clip a shelf upright
  * any contacts currently reported

The default all-zero pose leaves the arm fully extended, which is why
arm_left_6_link collides with the shelf on approach. A collision costs half a point
each time, so this has to be checked rather than assumed.
"""
import math
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from ros_gz_interfaces.msg import Contacts
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_LINKS = [f"arm_left_{i}_link" for i in range(1, 8)] + ["gripper_left_base_link"]
BASE = "base_footprint"


class Tucker(Node):
    def __init__(self, joints):
        super().__init__("try_tuck")
        self.joints = joints
        self.contacts = []
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(Contacts, "/contacts", self._on_contacts, 10)
        self.pub_left = self.create_publisher(
            JointTrajectory, "/arm_left_controller/joint_trajectory", 10)
        self.pub_right = self.create_publisher(
            JointTrajectory, "/arm_right_controller/joint_trajectory", 10)

    def _on_contacts(self, msg):
        for c in msg.contacts:
            pair = (c.collision1.name, c.collision2.name)
            if pair not in self.contacts:
                self.contacts.append(pair)

    def pump(self, seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def send(self, side, values, seconds=4.0):
        traj = JointTrajectory()
        traj.joint_names = [f"arm_{side}_{i}_joint" for i in range(1, 8)]
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in values]
        point.time_from_start = Duration(sec=int(seconds), nanosec=0)
        traj.points = [point]
        pub = self.pub_left if side == "left" else self.pub_right
        for _ in range(5):
            pub.publish(traj)
            self.pump(0.1)

    def extents(self):
        """Max forward (+x) and max lateral (|y|) reach of the arm links, in base frame."""
        best = []
        for link in ARM_LINKS:
            try:
                tf = self.buffer.lookup_transform(BASE, link, rclpy.time.Time())
            except Exception:  # noqa: BLE001 - link may not be published yet
                continue
            t = tf.transform.translation
            best.append((link, t.x, t.y, t.z))
        return best


def main():
    if len(sys.argv) != 8:
        print(__doc__)
        sys.exit(1)
    joints = [float(v) for v in sys.argv[1:8]]

    rclpy.init()
    node = Tucker(joints)
    node.pump(2.0)

    print(f"commanding tuck: {['%.2f' % j for j in joints]}", flush=True)
    node.send("left", joints)
    # Mirror the shoulder pan and upper-arm roll for the right arm.
    mirrored = list(joints)
    mirrored[0] = -joints[0]
    mirrored[2] = -joints[2]
    node.send("right", mirrored)

    node.contacts.clear()
    node.pump(8.0)

    ext = node.extents()
    if not ext:
        print("no TF for arm links")
    else:
        fwd = max(ext, key=lambda e: e[1])
        lat = max(ext, key=lambda e: abs(e[2]))
        print(f"furthest forward : {fwd[0]} at x={fwd[1]:+.3f} m")
        print(f"furthest lateral : {lat[0]} at y={lat[2]:+.3f} m")
        print(f"highest          : {max(ext, key=lambda e: e[3])[0]} "
              f"at z={max(e[3] for e in ext):+.3f} m")
        # Base half-length is 0.36 m; anything beyond that leads the robot into obstacles.
        if fwd[1] > 0.36:
            print(f"  WARNING: reaches {fwd[1]-0.36:+.3f} m beyond the base front")
        else:
            print("  OK: within the base footprint in x")

    if node.contacts:
        print(f"contacts ({len(node.contacts)}):")
        for a, b in node.contacts[:5]:
            print(f"  {a}  <->  {b}")
    else:
        print("contacts: none")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
