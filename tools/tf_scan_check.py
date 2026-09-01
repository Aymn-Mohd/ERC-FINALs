#!/usr/bin/env python3
"""Verify the scan -> base_footprint transform used by approach_node.

Prints the raw scan range span and the transformed span, so a transform that invents
near points shows up immediately.
"""
import math
import time

import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

BASE = "base_footprint"
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def main():
    rclpy.init()
    node = rclpy.create_node("tf_scan_check")
    buf = Buffer()
    TransformListener(buf, node)
    latest = {}
    node.create_subscription(LaserScan, "/scan_front_raw",
                             lambda m: latest.setdefault("s", m), SENSOR_QOS)

    end = time.time() + 15
    while rclpy.ok() and time.time() < end and "s" not in latest:
        rclpy.spin_once(node, timeout_sec=0.2)
    scan = latest.get("s")
    if scan is None:
        print("no scan")
        return

    print(f"scan frame: {scan.header.frame_id}")
    raw = [r for r in scan.ranges
           if math.isfinite(r) and scan.range_min < r < scan.range_max]
    print(f"raw ranges: n={len(raw)} min={min(raw):.3f} max={max(raw):.3f}")

    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)
    try:
        tf = buf.lookup_transform(BASE, scan.header.frame_id, rclpy.time.Time())
    except Exception as exc:  # noqa: BLE001
        print(f"TF lookup failed: {exc}")
        return

    q, t = tf.transform.rotation, tf.transform.translation
    xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
    xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
    wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
    r00, r01, r02 = 1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)
    r10, r11, r12 = 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)
    r20, r21, r22 = 2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)

    pts = []
    for i, r in enumerate(scan.ranges):
        if not math.isfinite(r) or not (scan.range_min < r < scan.range_max):
            continue
        a = scan.angle_min + i * scan.angle_increment
        lx, ly, lz = r * math.cos(a), r * math.sin(a), 0.0
        bx = r00 * lx + r01 * ly + r02 * lz + t.x
        by = r10 * lx + r11 * ly + r12 * lz + t.y
        bz = r20 * lx + r21 * ly + r22 * lz + t.z
        pts.append((bx, by, bz))

    d = [math.hypot(x, y) for x, y, _ in pts]
    print(f"transformed: n={len(pts)} min_dist={min(d):.3f} max_dist={max(d):.3f}")
    print(f"z span: {min(p[2] for p in pts):.3f} .. {max(p[2] for p in pts):.3f}")

    for label, half in (("+/-0.12 rad", 0.12), ("+/-0.35 rad", 0.35)):
        cone = [(x, y) for x, y, _ in pts if x > 0 and abs(math.atan2(y, x)) <= half]
        if cone:
            dd = [math.hypot(x, y) for x, y in cone]
            print(f"cone {label}: n={len(cone)} min={min(dd):.3f} "
                  f"median_x={sorted(x for x, _ in cone)[len(cone)//2]:.3f}")
        else:
            print(f"cone {label}: empty")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
