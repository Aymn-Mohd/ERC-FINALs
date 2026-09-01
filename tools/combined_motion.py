#!/usr/bin/env python3
"""Does the base honour forward and lateral velocity together?

Every earlier characterisation drove one axis at a time. The approach controller commands
both at once during its final drive, and that is where the robot stalls -- commanding
forward motion, reporting no contacts, and not moving. If the omni controller mishandles a
combined twist, or if small commands fall below a deadband, this is where it shows.

Small magnitudes are included deliberately: the final approach ramps down to vx = 0.05,
and a controller that ignores commands that small would look exactly like being stuck.
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


CASES = [
    ("forward only  ", 0.20, 0.00, 0.0),
    ("lateral only  ", 0.00, 0.10, 0.0),
    ("both together ", 0.20, 0.10, 0.0),
    ("as approach   ", 0.05, -0.05, 0.0),   # the exact stalling command
    ("small forward ", 0.05, 0.00, 0.0),
    ("tiny forward  ", 0.02, 0.00, 0.0),
]


def main():
    rclpy.init()
    node = rclpy.create_node("combined_motion")
    state = {}
    node.create_subscription(Odometry, "/odom",
                             lambda m: state.__setitem__("p", m.pose.pose), 10)
    cmd = node.create_publisher(Twist, "/cmd_vel", 10)

    def pump(seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    def snapshot():
        pump(0.4)
        p = state.get("p")
        return (p.position.x, p.position.y, yaw_of(p.orientation)) if p else None

    def drive(vx, vy, wz, seconds=3.0):
        before = snapshot()
        if before is None:
            return None
        x0, y0, yaw0 = before
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = vx, vy, wz
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            cmd.publish(twist)
            rclpy.spin_once(node, timeout_sec=0.05)
        for _ in range(12):
            cmd.publish(Twist())
            rclpy.spin_once(node, timeout_sec=0.05)
        pump(1.0)
        x1, y1, yaw1 = snapshot()
        dx_w, dy_w = x1 - x0, y1 - y0
        return (dx_w * math.cos(-yaw0) - dy_w * math.sin(-yaw0),
                dx_w * math.sin(-yaw0) + dy_w * math.cos(-yaw0),
                math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))

    pump(2.0)
    if state.get("p") is None:
        print("no /odom")
        return

    print(f"{'command':16s} {'vx':>6} {'vy':>6} -> {'dx':>7} {'dy':>7} {'dyaw':>7}")
    for label, vx, vy, wz in CASES:
        result = drive(vx, vy, wz)
        if result is None:
            print(f"{label} no odom")
            continue
        dx, dy, dyaw = result
        moved = "" if (abs(dx) > 0.01 or abs(dy) > 0.01) else "   <-- DID NOT MOVE"
        print(f"{label} {vx:6.2f} {vy:6.2f} -> {dx:+7.3f} {dy:+7.3f} {dyaw:+7.3f}{moved}",
              flush=True)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
