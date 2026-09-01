#!/usr/bin/env python3
"""Is the depth VALUE wrong, or the deprojection built on it?

    python3 depth_truth.py <book_model_name>

Both close-range errors point the same way -- the book reads about 0.11 m nearer than it
is in x, and objects deproject about 0.11 m too high in z -- which suggests one underlying
cause rather than two. These are opposite fixes though: a wrong depth reading needs the
sampling corrected, while a good reading placed wrongly needs the geometry corrected.

The test takes a book whose true pose is known from Gazebo, works out where the camera is
from TF, and compares three things:

    * the depth the camera reports over that book
    * the distance from the camera to the book along the optical axis
    * the full 3D position the pipeline produces, against the true one
"""
import math
import subprocess
import sys
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.vision import book_detector as bd  # noqa: E402
from avaa_solution.vision import depth_locator as dl  # noqa: E402

RGB = "/head_front_camera/head_front_camera/color/image_raw"
DEPTH = "/head_front_camera/head_front_camera/depth/image_rect_raw"
CAMERA = "head_front_camera_depth_optical_frame"
INTR = dl.Intrinsics(fx=337.2096, fy=337.2096, cx=320.0, cy=180.0)

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def gz_pose(model):
    """World pose of a Gazebo model as (x, y, z)."""
    out = subprocess.run(
        ["gz", "model", "-m", model, "-p"],
        capture_output=True, text=True, timeout=20).stdout
    lines = [l.strip() for l in out.splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
            return [float(v) for v in line.strip("[]").split()]
    return None


def main():
    model = sys.argv[1]
    truth = gz_pose(model)
    robot = gz_pose("tiago_pro")
    if truth is None or robot is None:
        print("could not read ground truth from Gazebo")
        return
    print(f"{model} true world position: {np.round(truth, 3).tolist()}")

    rclpy.init()
    node = rclpy.create_node("depth_truth")
    buf = Buffer()
    TransformListener(buf, node)
    bridge = CvBridge()
    frames = {}
    node.create_subscription(Image, RGB, lambda m: frames.__setitem__("rgb", m), SENSOR_QOS)
    node.create_subscription(Image, DEPTH, lambda m: frames.__setitem__("d", m), SENSOR_QOS)

    end = time.time() + 15
    while rclpy.ok() and time.time() < end and len(frames) < 2:
        rclpy.spin_once(node, timeout_sec=0.2)
    if len(frames) < 2:
        print("did not receive both image streams")
        return
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)

    rgb = bridge.imgmsg_to_cv2(frames["rgb"], desired_encoding="bgr8")
    depth = bridge.imgmsg_to_cv2(frames["d"], desired_encoding="passthrough")

    colour = model.rsplit("_", 1)[-1]
    books = [b for b in bd.detect_books(rgb) if b.colour == colour]
    if not books:
        print(f"no {colour} book detected in frame")
        return
    centre = rgb.shape[1] / 2.0
    book = min(books, key=lambda b: abs(b.cx - centre))
    print(f"detected {colour} book: bbox={book.bbox} centre=({book.cx:.0f},{book.cy:.0f})")

    # Where the camera is, in world terms.
    try:
        tf = buf.lookup_transform("base_link", CAMERA, rclpy.time.Time())
    except Exception as exc:  # noqa: BLE001
        print(f"TF unavailable: {exc}")
        return
    t = tf.transform.translation
    yaw = robot[2] if len(robot) > 2 else 0.0
    # robot[] is x, y, z; the yaw comes from the second pose line, not read here, so the
    # camera's world position is approximated using the base position plus the TF offset
    # rotated by the robot's heading. Good enough to compare magnitudes.
    print(f"camera in base_link: ({t.x:.3f}, {t.y:.3f}, {t.z:.3f})")

    sampled = dl.sample_depth(depth, book.bbox)
    print(f"\nsampled depth over the book : {sampled:.3f} m"
          if sampled is not None else "\nno usable depth over the book")

    point = dl.locate(book.bbox, depth, INTR)
    if point is not None:
        print(f"deprojected (optical frame) : "
              f"[{point[0]:+.3f} {point[1]:+.3f} {point[2]:+.3f}]")
        base = dl.transform_point(point, tf.transform.rotation, tf.transform.translation)
        print(f"in base_link                : "
              f"[{base[0]:+.3f} {base[1]:+.3f} {base[2]:+.3f}]")

    # True distance from camera to book, in world terms, using the base position and the
    # camera's height. Lateral offset is ignored, so this is a lower bound on the range.
    dx = truth[0] - robot[0]
    dy = truth[1] - robot[1]
    horizontal = math.hypot(dx, dy)
    print(f"\ntrue horizontal base-to-book: {horizontal:.3f} m")
    print(f"book height above camera    : {truth[2] - (robot[2] + t.z):+.3f} m")
    if sampled is not None:
        print(f"sampled depth minus true horizontal range: "
              f"{sampled - horizontal:+.3f} m")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
