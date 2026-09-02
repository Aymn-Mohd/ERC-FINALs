import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from erc_perception import color_vision

TOPIC_RGB = '/head_front_camera/head_front_camera/color/image_raw'
TOPIC_HEAD = '/head_controller/joint_trajectory'
TOPIC_ROW = '/erc/shelf_row_identification'

IMG_WIDTH = 640
IMG_HEIGHT = 360
VFOV_RAD = 0.977

VALID_COLOURS = ('red', 'blue', 'green', 'yellow')
IMAGES_DIR = '/opt/erc_ws/src/erc_images'

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class BookRowNode(Node):
    def __init__(self) -> None:
        super().__init__('erc_book_row')

        self.declare_parameter('book_colour', '')
        self.declare_parameter('tilt_steps', [0.15, -0.15, -0.45, -0.70])
        self.declare_parameter('tilt_settle_sec', 0.6)
        self.declare_parameter('min_area', float(color_vision.MIN_BOOK_AREA))
        self.declare_parameter('max_area', float(color_vision.MAX_BOOK_AREA))
        self.declare_parameter('roi_x_frac', 0.42)
        self.declare_parameter('min_aspect', float(color_vision.MIN_BOOK_ASPECT))
        self.declare_parameter('max_aspect', float(color_vision.MAX_BOOK_ASPECT))

        self.target_colour = str(self.get_parameter('book_colour').value).lower()
        self.tilt_steps = [float(v) for v in self.get_parameter('tilt_steps').value]
        self.settle_sec = float(self.get_parameter('tilt_settle_sec').value)
        self.min_area = float(self.get_parameter('min_area').value)
        self.max_area = float(self.get_parameter('max_area').value)
        self.roi_x_frac = float(self.get_parameter('roi_x_frac').value)
        self.min_aspect = float(self.get_parameter('min_aspect').value)
        self.max_aspect = float(self.get_parameter('max_aspect').value)

        if self.target_colour not in VALID_COLOURS:
            self.get_logger().error(
                f'book_colour must be one of {VALID_COLOURS}, got {self.target_colour!r}. '
                f'Pass it with --ros-args -p book_colour:=<colour>')

        os.makedirs(IMAGES_DIR, exist_ok=True)

        self.bridge = CvBridge()
        self.frame = None

        self.create_subscription(Image, TOPIC_RGB, self._on_image, SENSOR_QOS)
        self.pub_head = self.create_publisher(JointTrajectory, TOPIC_HEAD, 10)
        self.pub_row = self.create_publisher(Int32, TOPIC_ROW, 10)

        self.step_index = 0
        self.step_since: Optional[float] = None
        self.tilt_sent = False
        self.best: Dict[str, Tuple[float, float, np.ndarray, Tuple[int, int, int, int]]] = {}
        self.done = False

        self.create_timer(0.1, self._tick)
        self.get_logger().info(f'book_row_node up — target colour "{self.target_colour}"')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_image(self, msg: Image) -> None:
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'bad frame: {exc}')

    def _send_head_tilt(self, tilt: float) -> None:
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, float(tilt)]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.pub_head.publish(traj)

    def _tick(self) -> None:
        if self.done or self.target_colour not in VALID_COLOURS:
            return

        if self.step_index >= len(self.tilt_steps):
            self._finish()
            return

        now = self._now()
        if self.step_since is None:
            self.step_since = now
            self.tilt_sent = False

        if not self.tilt_sent:
            self._send_head_tilt(self.tilt_steps[self.step_index])
            self.tilt_sent = True
            return

        if now - self.step_since < self.settle_sec:
            return

        if self.frame is None:
            return

        if color_vision.is_looking_at_red_bin(self.frame):
            self.get_logger().info('red bin dominates the frame here — skipping this tilt step',
                                    throttle_duration_sec=2.0)
            self.step_index += 1
            self.step_since = None
            return

        tilt = self.tilt_steps[self.step_index]
        hits = color_vision.detect_all_colours(self.frame, self.min_area, self.max_area,
                                                self.roi_x_frac, self.min_aspect, self.max_aspect)
        for colour, (cx, cy, x, y, w, h) in hits.items():
            elevation = tilt + (IMG_HEIGHT / 2.0 - cy) * (VFOV_RAD / IMG_HEIGHT)
            area = float(w * h)
            prev = self.best.get(colour)
            if prev is None or area > prev[0]:
                self.best[colour] = (area, elevation, self.frame.copy(), (x, y, w, h))
            self.get_logger().info(
                f'{colour}: elevation={elevation:+.3f} rad (tilt step {self.step_index})')

        self.step_index += 1
        self.step_since = None

    def _finish(self) -> None:
        self.done = True
        if not self.best:
            self.get_logger().error('no coloured books detected across the tilt sweep; '
                                     'try adjusting tilt_steps or standoff distance')
            return

        ranked = sorted(self.best.items(), key=lambda kv: kv[1][1], reverse=True)
        self.get_logger().info('row order top-to-bottom: ' + ', '.join(c for c, _ in ranked))
        if len(ranked) < 4:
            self.get_logger().warn(
                f'only {len(ranked)}/4 book colours detected — row numbering may be off; '
                f'widen tilt_steps or move closer/further from the shelf')

        if self.target_colour not in self.best:
            self.get_logger().error(f'target colour "{self.target_colour}" not found')
            return

        row = next(i for i, (c, _) in enumerate(ranked, start=1) if c == self.target_colour)
        area, elevation, frame, (x, y, w, h) = self.best[self.target_colour]

        msg = Int32()
        msg.data = row
        self.pub_row.publish(msg)
        self.get_logger().info(f'target "{self.target_colour}" is row {row} — published to {TOPIC_ROW}')

        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.putText(annotated, timestamp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'{self.target_colour} row {row}', (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out_path = os.path.join(IMAGES_DIR, f'{timestamp}_row_{row}_{self.target_colour}.png')
        cv2.imwrite(out_path, annotated)
        self.get_logger().info(f'saved annotated image to {out_path}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BookRowNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
