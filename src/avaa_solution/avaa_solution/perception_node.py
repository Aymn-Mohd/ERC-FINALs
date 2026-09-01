"""Perception node — reads the column markers and finds books in the head camera feed.

Kept separate from the mission state machine so perception, planning and execution stay
distinct processes, which the Phase 1 rubric marks explicitly.

Two identifications are worth points, each scored twice (topic +1, annotated image +2):

* which shelf column carries the target marker digit
* which row the target-coloured book sits on within that column

This node works both out, publishes them on ``/avaa/perception/*``, and writes the
annotated images. It deliberately does **not** publish to ``/erc/shelf_column_identification``
or ``/erc/shelf_row_identification``: the mission node owns the scoring topics, so there is
exactly one place responsible for what reaches the committee.
"""

import os
from datetime import datetime
from collections import Counter, deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Int32
from tf2_ros import Buffer, TransformListener

from avaa_solution.vision import depth_locator as dl
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from avaa_solution.vision import book_detector as bd
from avaa_solution.vision import marker_reader as mr

TOPIC_RGB = "/head_front_camera/head_front_camera/color/image_raw"
TOPIC_DETECTIONS = "/avaa/perception/books"
TOPIC_TARGET_ROW = "/avaa/perception/target_row"
TOPIC_DEPTH = "/head_front_camera/head_front_camera/depth/image_rect_raw"
TOPIC_DEPTH_INFO = "/head_front_camera/head_front_camera/depth/camera_info"
TOPIC_TARGET_COLUMN = "/avaa/perception/target_column"
TOPIC_TARGET_COLUMN_X = "/avaa/perception/target_column_x"
TOPIC_TARGET_BOOK_POINT = "/avaa/perception/target_book_point"

# Where the grasp controller wants the book expressed.
GRASP_FRAME = "base_link"

# Shelf heights in base_link, top row first, and how far the deprojected height sits
# above the truth. Measured against Gazebo over 100 published points: +152 mm at 0.7-0.9 m
# with 41 mm of spread, +193 at 0.9-1.2, +146 at 1.2-1.6, +121 beyond. The bias is not
# constant enough to name a row on its own -- 0.6 of a row spacing at worst -- but it is
# nowhere near the 660 mm needed to confuse rows two apart, which is what makes it a
# usable check on an answer arrived at a completely different way.
#
# ASSUMPTION: the bias comes from the bounding box sitting high on the visible face of a
# book whose lower edge is occluded by the shelf lip. It is treated as a constant here
# because it does not need to be better than half a row to do this job.
ROW_HEIGHTS_BASE = [1.391, 1.061, 0.731, 0.401]
DEPTH_HEIGHT_BIAS = 0.152

# The camera publishes best-effort; a reliable subscriber receives nothing at all.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("avaa_perception")

        self.declare_parameter("book_colour", "")
        self.declare_parameter("shelf_column_number", 0)
        # Manual override of the column index, for testing without markers in view.
        # -1 (the default) means work it out from the marker digits.
        self.declare_parameter("target_column_index", -1)
        self.declare_parameter("save_images", True)
        # Only src/ is bind-mounted into the container, so this is the deepest path that
        # still lands inside the git repository on the host. See PERCEPTION.md.
        self.declare_parameter("image_dir", "/opt/erc_ws/src/avaa_solution/erc_images")
        self.declare_parameter("min_save_interval_sec", 2.0)
        self.declare_parameter("detect_period_sec", 0.2)

        self.book_colour = str(self.get_parameter("book_colour").value).lower()
        self.target_digit = int(self.get_parameter("shelf_column_number").value)
        self.image_dir = str(self.get_parameter("image_dir").value)
        self.save_images = bool(self.get_parameter("save_images").value)
        self.min_save_interval = float(self.get_parameter("min_save_interval_sec").value)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_header = None
        self.last_column_save: Optional[float] = None
        self.last_book_save: Optional[float] = None
        self.reported_column: Optional[int] = None
        self.reported_row: Optional[int] = None
        # Confident row readings, voted on before anything is latched. See _publish_row.
        self.row_votes = deque(maxlen=15)
        self.started_at = None
        self.row_majority = 0.7

        if self.save_images:
            os.makedirs(self.image_dir, exist_ok=True)

        self.depth_image = None
        # Start from the measured intrinsics rather than waiting for CameraInfo.
        #
        # CameraInfo is not reliably received: the depth image streams at 13 Hz while
        # camera_info can be absent entirely, and a node that starts after the initial
        # publication never sees it. Waiting for it meant _publish_book_point returned at
        # its first guard for the whole run -- silently, because the surrounding log said
        # the book was being tracked. The values are fixed for this camera and identical
        # between the colour and depth streams; a live CameraInfo still overrides them.
        self.intrinsics: Optional[dl.Intrinsics] = dl.Intrinsics(
            fx=337.2096, fy=337.2096, cx=320.0, cy=180.0)
        self.depth_frame: Optional[str] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, TOPIC_RGB, self._on_image, SENSOR_QOS)
        self.create_subscription(Image, TOPIC_DEPTH, self._on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, TOPIC_DEPTH_INFO, self._on_info, SENSOR_QOS)
        self.pub_book_point = self.create_publisher(
            PointStamped, TOPIC_TARGET_BOOK_POINT, 10)
        self.pub_detections = self.create_publisher(Detection2DArray, TOPIC_DETECTIONS, 10)
        self.pub_row = self.create_publisher(Int32, TOPIC_TARGET_ROW, 10)
        self.pub_column = self.create_publisher(Int32, TOPIC_TARGET_COLUMN, 10)
        self.pub_column_x = self.create_publisher(Float32, TOPIC_TARGET_COLUMN_X, 10)

        self.create_timer(float(self.get_parameter("detect_period_sec").value), self._process)

        if self.book_colour and self.book_colour not in bd.COLOURS:
            self.get_logger().error(
                f"book_colour {self.book_colour!r} is not one of {bd.COLOURS}"
            )
        try:
            mr.load_templates()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"could not load marker templates: {exc}")

        self.get_logger().info(
            f"perception up — target marker {self.target_digit}, "
            f"colour {self.book_colour or '(unset)'}, "
            f"images -> {self.image_dir if self.save_images else 'disabled'}"
        )

    # ------------------------------------------------------------------ callbacks

    def _on_image(self, msg: Image) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_header = msg.header
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the node
            self.get_logger().warn(f"could not convert frame: {exc}")

    def _on_depth(self, msg: Image) -> None:
        try:
            # 32FC1, metres. passthrough keeps the float values as they are.
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_frame = msg.header.frame_id
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"could not convert depth frame: {exc}")

    def _on_info(self, msg: CameraInfo) -> None:
        live = dl.Intrinsics.from_k(msg.k)
        if live != self.intrinsics:
            self.intrinsics = live
            self.get_logger().info(
                f"depth intrinsics from CameraInfo: fx={live.fx:.1f} "
                f"cx={live.cx:.1f} cy={live.cy:.1f}"
            )

    def _track_book_without_marker(self, frame, books: List[bd.Book]) -> None:
        """Keep publishing the target book once the marker is out of frame.

        Only valid after the column and row have been established, which is why it does
        nothing until then. Of the target-coloured books in view, it takes the one nearest
        the image centre: the robot has already centred and driven in on the target
        column, so the closest to centre is the one in front of it.
        """
        if self.reported_row is None or not self.book_colour:
            return
        candidates = [b for b in books if b.colour == self.book_colour]
        if not candidates:
            self.get_logger().warn(
                f"no {self.book_colour} book in view at close range",
                throttle_duration_sec=5.0,
            )
            return

        centre = frame.shape[1] / 2.0
        target = min(candidates, key=lambda b: abs(b.cx - centre))
        self.pub_row.publish(Int32(data=self.reported_row))

        # Keep feeding a bearing as well as a point. The marker leaves the frame roughly a
        # metre out, and without a bearing the approach drives the last stretch open-loop
        # and drifts sideways -- one run finished at the end upright of the shelf unit,
        # looking along it rather than at the target column. The book itself is the right
        # thing to steer by once the marker is gone.
        self.pub_column_x.publish(Float32(data=float(target.cx)))
        self._publish_book_point(target)
        self.get_logger().info(
            f"tracking {self.book_colour} book without marker "
            f"({len(candidates)} candidate(s), row {self.reported_row})",
            throttle_duration_sec=5.0,
        )

    def _row_from_height(self, point) -> Optional[int]:
        """Which row a measured height implies, correcting the known upward bias."""
        corrected = float(point[2]) - DEPTH_HEIGHT_BIAS
        best = min(range(len(ROW_HEIGHTS_BASE)),
                   key=lambda i: abs(ROW_HEIGHTS_BASE[i] - corrected))
        return best + 1

    def _cross_check_row(self, point) -> None:
        """Distrust the marker row when the measured height flatly contradicts it.

        The row is counted from the books grouped under a column marker, which needs the
        whole column in frame and gets it wrong when neighbouring books are swept in. One
        run latched row 3 for a book that was on row 1, aimed the head two shelves too
        low, and then reported no red book in view for the rest of the run while standing
        squarely in front of it.

        Being one row out is within what the height bias can explain, so that is left
        alone and only logged. Two rows apart is 660 mm and the height cannot be that
        wrong, so the height wins.
        """
        if self.reported_row is None:
            return
        implied = self._row_from_height(point)
        if implied is None or implied == self.reported_row:
            return
        if abs(implied - self.reported_row) < 2:
            self.get_logger().info(
                f"row {self.reported_row} from the markers, {implied} from the measured "
                f"height; keeping {self.reported_row}", throttle_duration_sec=10.0)
            return
        self.get_logger().warn(
            f"row {self.reported_row} from the markers but the book measures at row "
            f"{implied}, {abs(implied - self.reported_row)} rows away. The markers are "
            f"wrong; switching to {implied}")
        self.reported_row = implied
        self.row_votes.clear()

    def _publish_book_point(self, target: bd.Book) -> None:
        """Publish the target book's 3D position in base_link, for the grasp controller.

        The RGB and depth streams share intrinsics and dimensions exactly, so the box
        found in colour indexes the depth image directly.
        """
        # Say which input is missing rather than returning without a word. The check
        # below this used to be the one that logged, and it could never fire because the
        # guard above already caught the same case. A run then spent its entire timeout
        # printing "centred but the book is not located yet" from the approach while this
        # returned silently on every frame, and nothing anywhere named the depth stream.
        missing = [name for name, value in (
            ("depth image", self.depth_image),
            ("depth intrinsics", self.intrinsics),
            ("depth frame id", self.depth_frame),
        ) if value is None]
        if missing:
            if self.started_at is None:
                self.started_at = self.get_clock().now()
            waited = (self.get_clock().now() - self.started_at).nanoseconds / 1e9
            # After this long it is not a slow start, it is a sensor that never came up.
            # The Gazebo depth camera fails to start every so often and leaves its topic
            # advertised with nothing on it, which otherwise surfaces minutes later as a
            # timeout in the approach with no mention of a camera anywhere.
            if waited > 15.0:
                self.get_logger().error(
                    "no %s after %.0f s. The simulator sometimes starts without its "
                    "depth camera; restart it." % (", ".join(missing), waited),
                    throttle_duration_sec=10.0)
            else:
                self.get_logger().warn(
                    "cannot place the book in 3D, still waiting on: %s"
                    % ", ".join(missing), throttle_duration_sec=5.0)
            return
        point_optical = dl.locate(target.bbox, self.depth_image, self.intrinsics)
        if point_optical is None:
            self.get_logger().warn(
                "no usable depth over the target book", throttle_duration_sec=5.0
            )
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                GRASP_FRAME, self.depth_frame, rclpy.time.Time()
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"no transform {self.depth_frame} -> {GRASP_FRAME}: {exc}",
                throttle_duration_sec=5.0,
            )
            return

        point = dl.transform_point(
            point_optical, tf.transform.rotation, tf.transform.translation
        )
        msg = PointStamped()
        msg.header.frame_id = GRASP_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = (float(v) for v in point)
        self.pub_book_point.publish(msg)
        self._cross_check_row(point)

    def _process(self) -> None:
        if self.latest_frame is None:
            return
        frame = self.latest_frame

        try:
            books = bd.detect_books(frame)
            markers = sorted(mr.read_markers(frame), key=lambda m: m.cx)
            # The markers define the columns. Falling back to gap clustering only when
            # none are visible, since that cannot identify a target column anyway.
            if markers:
                columns = bd.group_by_anchors(
                    books, [m.cx for m in markers], column_max_dx(markers)
                )
            else:
                columns = bd.group_into_columns(books)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"perception failed on this frame: {exc}")
            return

        self._publish_detections(books, columns)

        column_index = self._target_column_index(markers)
        if column_index is None:
            # Close range: the markers sit at 2.26 m and the camera is tilted down onto
            # the books, so they leave the frame entirely. That is expected, and by then
            # the column has already been identified and the robot is parked in front of
            # it -- so keep tracking the book itself rather than going silent exactly when
            # the grasp controller needs a target.
            self._track_book_without_marker(frame, books)
            return

        if column_index != self.reported_column:
            self.get_logger().info(
                f"marker {self.target_digit} is column index {column_index} "
                f"({len(columns)} column(s) in view)"
            )
            self.reported_column = column_index
        self.pub_column.publish(Int32(data=column_index))

        # Where to steer, for the approach controller.
        #
        # Not the column index: that is frame-relative, counting the columns currently in
        # view, so it changes as markers enter and leave the frame -- observed jumping
        # 1, 2, 1, 0 across consecutive frames while the robot drove.
        #
        # Steer by the target BOOK whenever it can be seen, and only fall back to the
        # marker before it has been found. Aiming the head down to keep the books in frame
        # is precisely what drives the markers out of it, so steering by the marker fails
        # exactly when the alignment needs to be at its best. One run ended at the shelf's
        # end upright, looking along the shelves rather than at its column.
        target_book = (bd.find_book(columns, column_index, self.book_colour)
                       if self.book_colour else None)
        steer_x = target_book.cx if target_book is not None else markers[column_index].cx
        self.pub_column_x.publish(Float32(data=float(steer_x)))
        self._save_column_image(frame, books, columns, markers, column_index)

        if not self.book_colour:
            return
        row = bd.row_of(columns[column_index], self.book_colour)

        # Latch the row once resolved.
        #
        # Resolving it needs all four books of the column in frame, which only holds at a
        # distance. By grasping range the column no longer fits in the image and row_of()
        # correctly returns None -- so without a latch perception stops reporting the row
        # exactly when the grasp controller starts needing it. The row cannot change
        # during a run, so the first confident answer stands.
        # Only trust a row read from a genuinely resolved column.
        #
        # With a single marker in view, group_by_anchors has no spacing to measure and
        # falls back to estimating the column width from the marker's own apparent size.
        # That estimate can sweep in books from the neighbouring column, and four books
        # drawn from two columns still look like a complete column to row_of(). It then
        # gets latched and never revised: one run resolved "row 4" from a single-column
        # view and drove to a book that was actually on row 1, the full height of the
        # shelf away.
        #
        # Two markers give a measured spacing, which is the difference between knowing
        # where the column ends and guessing.
        # This applies whether or not a row is already held. Guarding only the first
        # reading was worse than useless: a good two-marker answer would be latched and
        # then overwritten by a single-marker one a few frames later. Observed exactly
        # that -- "on row 4 (2 marker(s) in view)" followed by "on row 3 (1 marker(s) in
        # view)" -- and the grasp used the corrupted value. Row 4 had been correct.
        if row is not None and len(markers) < 2:
            if self.reported_row is None:
                self.get_logger().warn(
                    "row seen but only one marker in view; waiting for a second before "
                    "trusting it", throttle_duration_sec=5.0)
            row = None

        # Vote, then latch, then never change it.
        #
        # The row cannot change during a run. The code above said exactly that in a
        # comment while overwriting reported_row on every confident reading, and since the
        # approach aims the head from the row we publish, that closed a loop: row 2 tilts
        # the head up, which changes which books are in frame, which reads row 4, which
        # tilts it back down. One run flipped between rows 2 and 4 several times a second
        # for the whole approach, rejected 36 sightings as inconsistent with the anchored
        # target, and ended 4.7 m from the shelf having never grasped anything.
        #
        # Publishing nothing until the vote settles is what breaks the loop: with no row,
        # the head stops being re-aimed, the view holds still, and the readings converge.
        # A split vote is reported rather than resolved -- an honest "cannot tell" that
        # holds position is worth more than an answer that alternates.
        if row is not None:
            self.row_votes.append(row)

        if self.reported_row is None:
            if len(self.row_votes) < self.row_votes.maxlen:
                self.get_logger().info(
                    f"reading the row ({len(self.row_votes)} of "
                    f"{self.row_votes.maxlen} samples)",
                    throttle_duration_sec=3.0)
                return
            tally = Counter(self.row_votes)
            winner, count = tally.most_common(1)[0]
            if count < self.row_majority * self.row_votes.maxlen:
                self.get_logger().warn(
                    f"row is ambiguous, holding: {dict(tally)}",
                    throttle_duration_sec=5.0)
                return
            self.reported_row = winner
            self.get_logger().info(
                f"target {self.book_colour} book is on row {winner} "
                f"({count} of {len(self.row_votes)} readings agree)")

        row = self.reported_row
        self.pub_row.publish(Int32(data=row))

        target = bd.find_book(columns, column_index, self.book_colour)
        if target is not None:
            self._publish_book_point(target)
            self._save_book_image(frame, books, target, row)

    # ------------------------------------------------------------------ helpers

    def _target_column_index(self, markers: List[mr.Marker]) -> Optional[int]:
        """Index into ``columns`` of the column carrying the target marker digit.

        Because columns are anchored to the markers, the marker's own left-to-right
        position is the column index.
        """
        override = int(self.get_parameter("target_column_index").value)
        if override >= 0:
            return override
        if not self.target_digit:
            return None
        hits = [i for i, m in enumerate(markers)
                if m.digit == self.target_digit and m.confident]
        if len(hits) != 1:
            return None  # absent, or read twice -- move for a better view
        return hits[0]

    def _publish_detections(self, books: List[bd.Book],
                            columns: List[List[bd.Book]]) -> None:
        msg = Detection2DArray()
        if self.latest_header is not None:
            msg.header = self.latest_header

        column_of = {}
        for index, column in enumerate(columns):
            for book in column:
                column_of[id(book)] = index

        for book in books:
            det = Detection2D()
            det.header = msg.header
            bbox = BoundingBox2D()
            bbox.center.position.x = book.cx
            bbox.center.position.y = book.cy
            bbox.size_x = float(book.w)
            bbox.size_y = float(book.h)
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = book.colour
            hyp.hypothesis.score = 1.0
            det.results.append(hyp)
            det.id = str(column_of.get(id(book), -1))
            msg.detections.append(det)

        self.pub_detections.publish(msg)

    def _due(self, last: Optional[float]) -> bool:
        if not self.save_images:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        return last is None or (now - last) >= self.min_save_interval

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _write(self, image, name: str) -> None:
        try:
            cv2.imwrite(os.path.join(self.image_dir, name), image)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"could not save {name}: {exc}")

    def _save_column_image(self, frame, books, columns, markers, column_index: int) -> None:
        """Annotated frame with a box around the identified column. Worth +2."""
        if not self._due(self.last_column_save):
            return
        self.last_column_save = self._now()

        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]
        caption = f"AVAA {stamp}  column marker={self.target_digit}"
        vis = bd.annotate(frame, books, caption=caption)

        marker = markers[column_index] if column_index < len(markers) else None
        box = column_bbox(columns[column_index], marker)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
        self._write(vis, f"column_{self.target_digit}_{stamp}.png")

    def _save_book_image(self, frame, books, target: bd.Book, row: int) -> None:
        """Annotated frame with a box around the target book. Worth +2."""
        if not self._due(self.last_book_save):
            return
        self.last_book_save = self._now()

        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]
        caption = f"AVAA {stamp}  row={row}  colour={self.book_colour}"
        vis = bd.annotate(frame, books, highlight=target, caption=caption)
        self._write(vis, f"row_{row}_{self.book_colour}_{stamp}.png")


# ---------------------------------------------------------------------- pure helpers


def column_max_dx(markers: List[mr.Marker]) -> float:
    """How far from its marker a book may sit and still belong to that column, in pixels.

    With two or more markers the spacing between them measures a column directly. With
    only one -- which happens as soon as the robot is close enough that the others leave
    the frame -- there is no spacing to measure, so the marker's own apparent width
    provides the scale instead.

    The marker plate is 0.30 m wide and a shelf column is 1.05 m, so a column is 3.5
    marker-widths across and half of that is the radius wanted. Being a ratio of two
    lengths in the same image, it holds at any distance.
    """
    if len(markers) >= 2:
        xs = sorted(m.cx for m in markers)
        spacings = [b - a for a, b in zip(xs, xs[1:])]
        return 0.5 * float(np.median(spacings))
    if markers:
        return 1.75 * float(markers[0].w)
    return float("inf")


def column_bbox(column: List[bd.Book],
                marker: Optional[mr.Marker] = None
                ) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box spanning a column's books and, when given, its marker above.

    Including the marker matters: the points are awarded for a box around the identified
    *column*, and the marker is what identifies it.
    """
    xs: List[int] = []
    ys: List[int] = []
    for book in column:
        xs += [book.x, book.x + book.w]
        ys += [book.y, book.y + book.h]
    if marker is not None:
        xs += [marker.x, marker.x + marker.w]
        ys += [marker.y, marker.y + marker.h]
    if not xs:
        return None

    pad = 6
    x0, x1 = max(0, min(xs) - pad), max(xs) + pad
    y0, y1 = max(0, min(ys) - pad), max(ys) + pad
    return (x0, y0, x1 - x0, y1 - y0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT from a terminal, or SIGTERM from `ros2 launch` shutting the run down.
        # Both are ordinary exits, not faults -- do not spew a traceback over the logs
        # the judges will be reading.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
