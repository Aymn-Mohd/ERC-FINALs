#!/usr/bin/env python3
"""Measure how the omnidirectional base actually responds to /cmd_vel.

Publishes one pure command at a time and reports the resulting motion in the body frame.
Nav2 assumes the base realises the twist it is given; if the mapping is rotated, scaled or
mirrored, closed-loop control chases its own tail and drives away from the goal.

Expected for a correct mecanum base:
    +vx  -> forward   (body dx > 0, dy ~ 0)
    +vy  -> left      (body dy > 0, dx ~ 0)
    +wz  -> CCW       (dyaw > 0, little translation)
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Rig(Node):
    def __init__(self):
        super().__init__("base_kinematics")
        self.pose = None
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)

    def _odom(self, msg):
        self.pose = msg.pose.pose

    def pump(self, seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def snapshot(self):
        self.pump(0.4)
        p = self.pose
        return (p.position.x, p.position.y, yaw_of(p.orientation))

    def drive(self, vx, vy, wz, seconds=2.5):
        x0, y0, yaw0 = self.snapshot()
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            self.cmd.publish(t)
            rclpy.spin_once(self, timeout_sec=0.05)
        for _ in range(12):
            self.cmd.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.05)
        self.pump(1.0)
        x1, y1, yaw1 = self.snapshot()

        # World displacement expressed in the body frame it started in.
        dx_w, dy_w = x1 - x0, y1 - y0
        dx_b = dx_w * math.cos(-yaw0) - dy_w * math.sin(-yaw0)
        dy_b = dx_w * math.sin(-yaw0) + dy_w * math.cos(-yaw0)
        dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
        return dx_b, dy_b, dyaw


def main():
    rclpy.init()
    rig = Rig()
    rig.pump(2.0)
    if rig.pose is None:
        print("ERROR: no /odom")
        return

    for label, (vx, vy, wz) in (
        ("forward  vx=+0.20", (0.20, 0.0, 0.0)),
        ("left     vy=+0.20", (0.0, 0.20, 0.0)),
        ("ccw      wz=+0.40", (0.0, 0.0, 0.40)),
    ):
        dx_b, dy_b, dyaw = rig.drive(vx, vy, wz)
        print(f"{label}  ->  body dx={dx_b:+.3f} m  dy={dy_b:+.3f} m  dyaw={dyaw:+.3f} rad",
              flush=True)

    rig.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
