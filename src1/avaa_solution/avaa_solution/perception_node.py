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
import math
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Int32
from tf2_ros import Buffer, TransformListener

from avaa_solution.vision import depth_locator as dl
from avaa_solution.vision import shelf_plane as sp
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
TOPIC_BIN_POINT = "/avaa/perception/bin_point"
# The column's position on the shelf, 1-5, as opposed to its index among
# whichever columns happen to be in frame. This is the one the judges want.
TOPIC_SHELF_COLUMN = "/avaa/perception/shelf_column"
# The base's yaw error against the shelf, measured from the depth image.
TOPIC_SHELF_YAW = "/avaa/perception/shelf_yaw"

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
COLUMNS_ON_SHELF = 5
ROW_HEIGHTS_BASE = [1.391, 1.061, 0.731, 0.401]

# The bin rim, in base_link, taken as known rather than measured.
#
# The rules fix the bin on a table: table 140 x 80 x 73 cm, bin 50 x 31 x 21 cm, so the
# rim stands 0.94 m above the floor and base_link sits 0.186 m up. Measured in the
# simulator the rim is at 0.950, which is 10 mm from the arithmetic.
#
# Height is taken as known for the same reason it is taken as known for the shelf rows:
# depth is trustworthy sideways and in range and is not trustworthy vertically. Measured
# against ground truth, two settled readings of the bin came back +18 and +14 mm high,
# and a third taken while the head was still tilting came back +176 mm. A number that
# depends on whether the head has stopped moving is not a number to place a book with.
BIN_RIM_BASE_Z = 0.950 - 0.186
BIN_DEPTH_M = 0.50
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
        self.declare_parameter("columns_left_to_right", True)
        # Inside this range the book is a better bearing than the marker,
        # because the marker has left the top of the frame. Outside it the
        # marker is the only thing that identifies WHICH column.
        self.declare_parameter("book_steering_range_m", 1.60)
        # How far a book may appear to move between frames and still be
        # believed to be the same book. Generous, because the base slides
        # and the head tilts; far short of the gap between columns, which
        # is what this is there to refuse.
        self.declare_parameter("book_jump_px", 120.0)
        self.declare_parameter("save_images", True)
        # Only src/ is bind-mounted into the container, so this is the deepest path that
        # still lands inside the git repository on the host. See PERCEPTION.md.
        self.declare_parameter("image_dir", "/opt/erc_ws/src/avaa_solution/erc_images")
        self.declare_parameter("min_save_interval_sec", 2.0)
        self.declare_parameter("detect_period_sec", 0.2)

        self.book_colour = str(self.get_parameter("book_colour").value).lower()
        self.target_digit = int(self.get_parameter("shelf_column_number").value)
        self.columns_left_to_right = bool(
            self.get_parameter("columns_left_to_right").value)
        self.close_range = float(
            self.get_parameter("book_steering_range_m").value)
        self.book_jump_px = float(self.get_parameter("book_jump_px").value)
        self.image_dir = str(self.get_parameter("image_dir").value)
        self.save_images = bool(self.get_parameter("save_images").value)
        self.min_save_interval = float(self.get_parameter("min_save_interval_sec").value)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_header = None
        self.last_column_save: Optional[float] = None
        self.last_book_save: Optional[float] = None
        self.reported_column: Optional[int] = None
        self.shelf_column: Optional[int] = None
        self.last_book_range: Optional[float] = None
        # Half the image width, and how far off centre a marker may be and still have
        # its digit believed.
        #
        # This was 230 px -- 34 degrees at fx 337.2 -- on the theory that every
        # acquisition happening at 40 to 43 degrees meant the digit was being misread on
        # a badly foreshortened plate. Tried, and it stopped the approach finding a
        # marker at all: 150 seconds of searching, a dozen full turns, not one accepted
        # reading, and the state timed out.
        #
        # What that ruled out is worth more than the change. A marker first ENTERS the
        # frame at the edge, so a search that stops on first sight will always stop
        # there; that part is expected. What is not is that a full sweep never once
        # caught it nearer the middle, and the reason is in the bearing period the
        # approach now measures: 7.7, 5.4, 2.8 seconds between one changed reading and
        # the next. Turning at the 0.35 rad/s search rate, the robot sweeps 150 degrees
        # between frames. The marker is not being read badly at the edge. It is
        # hardly being SAMPLED, and no visual controller can work at that rate.
        #
        # So this stays wide open until perception is fast enough for it to mean
        # anything, and the number to fix is the frame rate, not the gate.
        self.image_centre_px = 320.0
        self.marker_edge_px = 1000.0
        self.last_book_cx: Optional[float] = None
        # The base's heading when last_book_cx was measured, so the next frame can be
        # matched against where the book will have MOVED to rather than where it was.
        self.last_book_yaw: Optional[float] = None
        self.base_yaw: Optional[float] = None
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
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Image, TOPIC_DEPTH, self._on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, TOPIC_DEPTH_INFO, self._on_info, SENSOR_QOS)
        self.pub_book_point = self.create_publisher(
            PointStamped, TOPIC_TARGET_BOOK_POINT, 10)
        self.pub_bin_point = self.create_publisher(
            PointStamped, TOPIC_BIN_POINT, 10)
        self.pub_shelf_column = self.create_publisher(
            Int32, TOPIC_SHELF_COLUMN, 10)
        self.pub_shelf_yaw = self.create_publisher(
            Float32, TOPIC_SHELF_YAW, 10)
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

    def _watch_jump(self, point_optical, target) -> None:
        """Report a book fix that moves further than the robot could have carried it.

        The approach refuses these, correctly, and then has nothing left to drive to:
        watched over ten seconds of APPROACH it rejected seven sightings, one of them
        0.41 m from the last accepted after 0.2 s and another 0.76 m after 1.4 s, with
        the base moving at 0.22 m/s. A gate doing the right thing with bad data leaves
        the drive aiming at a stale target, so the question is what makes the data bad.

        This says what the fix did and what the picture looked like when it did it: the
        jump, the range, the size of the box it was measured over, and how many markers
        were in view. If jumps coincide with the marker count falling, the column
        grouping is losing its anchor; if they coincide with a small box, the depth
        patch is landing on the gap behind the book; if with neither, the two image
        streams are out of step.
        """
        if point_optical is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        previous = getattr(self, "_jump_point", None)
        when = getattr(self, "_jump_at", None)
        self._jump_point = np.asarray(point_optical, dtype=float)
        self._jump_at = now
        if previous is None or when is None:
            return
        gap = now - when
        moved = float(np.linalg.norm(self._jump_point - previous))
        # 0.22 m/s is the approach's top speed; anything past twice that in the time
        # available is not the robot having moved.
        if gap > 1e-3 and moved > 0.06 + 0.45 * gap:
            _, _, w, h = target.bbox
            self.get_logger().warn(
                "the book fix jumped %.0f mm in %.2f s (%.1f m/s) at a range of "
                "%.2f m, box %dx%d px, %d marker(s) in view"
                % (moved * 1000, gap, moved / gap, float(point_optical[2]),
                   w, h, getattr(self, "_markers_seen", -1)),
                throttle_duration_sec=2.0)

    def _tally(self, markers, hits) -> None:
        """Count why the target marker is or is not identified, and say so.

        This is the number that separates a centring that works from one that does not.
        Watched across runs: when a bearing arrives every 0.2 to 0.3 s the state
        converges cleanly -- -301, -199, -77, -14 px, tracking the base rotation exactly
        -- and when it arrives every 0.8 s or slower the error sits still through tens of
        degrees of turn. Perception reads every frame it is given, 30.3 a second
        measured, so the gap is not the frame rate. It is how often a frame yields
        exactly one confident marker carrying the target digit, which is what
        _target_column_index demands before it will publish anything.

        So: of the frames with markers in view, how many gave no confident read of the
        target, how many gave one, and how many gave more than one. The last of those is
        the interesting column, because two markers both read as the target digit means
        the reader is confident and wrong.
        """
        self._markers_seen = len(markers)
        self._seen = getattr(self, "_seen", 0) + 1
        self._digits = getattr(self, "_digits", {})
        self._scores = getattr(self, "_scores", {})
        for m in markers:
            key = "%s%s" % (m.digit, "" if m.confident else "?")
            self._digits[key] = self._digits.get(key, 0) + 1
            self._scores[key] = max(self._scores.get(key, 0.0), float(m.score))
        if not hits:
            self._none = getattr(self, "_none", 0) + 1
        elif len(hits) > 1:
            self._many = getattr(self, "_many", 0) + 1
        else:
            self._one = getattr(self, "_one", 0) + 1
        now = self.get_clock().now().nanoseconds * 1e-9
        since = getattr(self, "_tally_at", None)
        if since is None:
            self._tally_at = now
            return
        if now - since >= 10.0:
            self.get_logger().info(
                "marker %s identified on %d of %d frames in %.0f s: %d found none, "
                "%d found more than one; %d markers in view on the last of them"
                % (self.target_digit, getattr(self, "_one", 0), self._seen,
                   now - since, getattr(self, "_none", 0), getattr(self, "_many", 0),
                   len(markers)))
            if self._digits:
                self.get_logger().info(
                    "  digits read in that window: %s"
                    % ", ".join("%s x%d (best score %.2f)"
                                % (d, n, self._scores.get(d, 0.0))
                                for d, n in sorted(self._digits.items())))
            self._seen = self._one = self._none = self._many = 0
            self._digits = {}
            self._scores = {}
            self._tally_at = now

    def _count_frame(self) -> None:
        """Say how fast this node is actually looking, in SIMULATED frames a second.

        Everything downstream is a visual controller, and a visual controller cannot
        beat its own sensor. The approach measures the interval between changed
        bearings and has reported anything from 0.2 to 7.7 seconds, but that figure is
        contaminated -- a gap spanning a state change inflates it -- so it cannot settle
        the question. This can: it counts frames where they arrive.

        Counted where the work is DONE, not where the frames arrive. The first version
        of this sat in _on_image, which only stores the picture, and so reported the
        camera's 30 Hz as though it were the detection rate. The detector runs on its
        own timer at 5 Hz, which is ample for a controller that needs to turn a few
        degrees between looks -- so this closes off the frame rate as an explanation
        rather than confirming it.

        Per simulated second, because the real-time factor moves by a factor of forty
        over a session and a rate per wall second says as much about how long the
        instance has been up as about the robot.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        self._frames = getattr(self, "_frames", 0) + 1
        since = getattr(self, "_frames_at", None)
        if since is None:
            self._frames_at = now
            return
        if now - since >= 10.0:
            self.get_logger().info(
                "looking at %.1f frames per simulated second (%d in %.0f s)"
                % (self._frames / (now - since), self._frames, now - since))
            self._frames = 0
            self._frames_at = now

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self.base_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _expected_cx(self, width: float) -> Optional[float]:
        """Where the tracked book should appear now, given how far the base has turned.

        Matching the nearest candidate to where the book was LAST seen is a tracker
        that follows its own output, and on a shelf of identically coloured books it
        will hop from one to the next to stay at the same place on the screen. Measured
        in a run: the base turned 22 degrees against Gazebo's ground truth, with odom
        agreeing, and the bearing this publishes moved from 286 px to 286 px. A marker
        or book fixed in the world should have swept 136 px in that time. The camera
        was live throughout -- consecutive frames differed by 30 grey levels -- so the
        detector was seeing new pictures and choosing a different book out of each one.

        The correction is to predict rather than assume. A point at image x sits at a
        bearing atan((x - cx0) / fx) from the camera axis; turn the base left by dpsi
        and that bearing shrinks by dpsi. So the same book will be found at
        cx0 + fx * tan(theta - dpsi), and the nearest candidate to THAT is the one to
        follow.

        Odom is the right source for dpsi and only for dpsi. It cannot see this base
        slide, but tools/turncheck.py measured its yaw against Gazebo over commanded
        turns -- 0.142, 0.270 and 0.407 rad/s reported against 0.142, 0.235 and 0.356
        true -- and over the fifth of a second between two frames even the worst of
        that is a fraction of a pixel.
        """
        if self.last_book_cx is None:
            return None
        if self.last_book_yaw is None or self.base_yaw is None:
            return self.last_book_cx
        focal = self.intrinsics.fx if self.intrinsics is not None else 337.2
        centre = width / 2.0
        turned = math.atan2(math.sin(self.base_yaw - self.last_book_yaw),
                            math.cos(self.base_yaw - self.last_book_yaw))
        theta = math.atan2(self.last_book_cx - centre, focal)
        moved = theta - turned
        # Past a right angle the tangent stops meaning anything, and the book is well
        # out of frame long before that.
        if abs(moved) > 1.2:
            return None
        return centre + focal * math.tan(moved)

    def _track_book_without_marker(self, frame, books: List[bd.Book]) -> None:
        """Keep publishing the target book once the marker is out of frame.

        Only valid after the column and row have been established, AND after the marker
        has identified which book is the target at least once. Both are required and
        only the first used to be checked.

        What this can do is follow a book it already knows. What it cannot do is pick
        one out of several: the marker is the only thing that says which of five
        same-coloured books belongs to the target column, and once it has left the frame
        there is nothing in the picture that distinguishes them.
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

        # Follow the SAME book, not whichever one is nearest the middle of the picture.
        #
        # Nearest-the-centre is only right if the robot is already in front of its
        # column, and it is worst exactly when it matters. Measured on a run that got
        # this far: the robot stood at (2.35, -0.24) with column 3's green book 0.55 m
        # ahead and 0.24 m to its left, and perception handed the grasp a book 0.87 m
        # ahead and 0.72 m to its RIGHT -- column 4's green book, which happened to sit
        # nearer the middle of the frame. The grasp then planned a reach into the wrong
        # column.
        #
        # The marker identified the right book while it was still in frame, so its image
        # position is known. Between frames a book moves a little; it does not jump
        # across a column. So the candidate nearest where the target was last seen is
        # the target, and if nothing is near enough, the honest answer is that the book
        # has been lost rather than that some other book will do.
        if self.last_book_cx is None:
            # The marker has never identified a book, so there is nothing here to
            # recognise the target BY, and nearest-the-centre is a guess.
            #
            # The comment below already says what that guess costs, and the range gate
            # further down was written to stop it -- but it gates the steering BEARING
            # only, and the approach does not navigate on the bearing. It navigates on
            # the 3D point, which was published from any range at all.
            #
            # Measured on the run that found this: from 2.92 m out, with the robot
            # centred on its marker to 5.5 px, this published a book 1.37 m to the
            # side. The approach fixed its target on it, drove sideways with its
            # lateral command saturated for the whole run, ended up between two
            # columns, and went back to searching. The bearing gate never fired,
            # because the bearing was never the problem.
            #
            # Silence is the honest answer. The approach stops and looks again, which
            # is what it is for.
            self.get_logger().warn(
                "%d %s book(s) in view but the marker has not identified which is the "
                "target; not offering a fix on a guess"
                % (len(candidates), self.book_colour),
                throttle_duration_sec=5.0)
            return
        expected = self._expected_cx(float(frame.shape[1]))
        if expected is None:
            # The prediction has left the picture, so this tracker no longer knows
            # which book is the target and cannot get that back on its own. Forget the
            # fix rather than returning None for ever: with last_book_cx cleared, the
            # branch above goes quiet and says why, and the marker path re-seeds the
            # tracker the moment the marker is in frame again.
            #
            # Without this the node deadlocks. Watched in a run: the base swung 60
            # degrees hunting for its column, the prediction went out of frame, and
            # perception published no book point again for the rest of the attempt --
            # the approach squaring back and forth reporting "no metric fix on the
            # book to drive to" until it timed out.
            self.get_logger().warn(
                "the %s book has turned out of frame; waiting for the marker to say "
                "which one is the target again" % self.book_colour,
                throttle_duration_sec=5.0)
            self.last_book_cx = None
            self.last_book_yaw = None
            return
        target = min(candidates, key=lambda b: abs(b.cx - expected))
        jump = abs(target.cx - expected)
        if jump > self.book_jump_px:
            # Do not follow it, and do not remember it.
            #
            # This used to warn and publish anyway, on the reasoning that a large jump
            # is common while the head tilts and the base slides, and that going silent
            # had once starved the approach. Measured, the cost of following is worse.
            # Perception now reports what its own fix does, and every jump it logged in
            # a run came with ZERO or one marker in view:
            #
            #     jumped 1071 mm in 0.20 s (5.4 m/s) at 1.72 m, 0 markers in view
            #     jumped  716 mm in 0.20 s (3.6 m/s) at 1.63 m, 0 markers in view
            #     jumped  717 mm in 0.20 s (3.6 m/s) at 0.90 m, 0 markers in view
            #     jumped 1149 mm in 0.20 s (5.7 m/s) at 1.82 m, 0 markers in view
            #
            # Those are one column spacing, 0.95 m, and a base that moves at 0.22 m/s
            # did not travel them. With no marker to say which column is the target,
            # the nearest candidate to the prediction is simply the neighbouring
            # column's book, and following it overwrites last_book_cx so the tracker
            # can never come back. The approach then refuses the sighting anyway --
            # correctly, its own gate catches 0.41 m in 0.2 s -- so publishing it buys
            # nothing and costs the anchor.
            #
            # Holding the anchor is what lets the true book be recognised when it
            # reappears, which is the whole point of predicting where it went.
            self.get_logger().warn(
                "the %s book is %.0f px from where turning %.0f deg should have put "
                "it; that is another column, so holding the old fix"
                % (self.book_colour, jump,
                   math.degrees(0.0 if (self.base_yaw is None
                                        or self.last_book_yaw is None)
                                else self.base_yaw - self.last_book_yaw)),
                throttle_duration_sec=5.0)
            return
        self.last_book_cx = float(target.cx)
        self.last_book_yaw = self.base_yaw
        self.pub_row.publish(Int32(data=self.reported_row))
        self._publish_book_point(target)

        # Steer by the book only when the robot is CLOSE to it.
        #
        # This exists because the marker leaves the frame about a metre out, and without
        # a bearing the approach drives the last stretch open-loop and drifts sideways.
        # It was publishing at every range, and that turned it from a last-metre aid
        # into a hijack: the bearing is whichever book of the target colour is nearest
        # the middle of the picture, and from three metres back with the head level that
        # is any one of five columns. In one run it published 113 times while the base
        # was still centring on the marker, and the approach spent ninety seconds
        # turning towards a book in the wrong column with the pixel error never
        # improving, because the error was measured against a different book each frame.
        #
        # Beyond this range the marker is the thing to steer by and it should be
        # visible; if it is not, the robot is not looking at its column, and silence is
        # the honest answer. The approach stops and looks again rather than driving
        # confidently at the wrong shelf.
        if self.last_book_range is not None and self.last_book_range <= self.close_range:
            self.pub_column_x.publish(Float32(data=float(target.cx)))
        else:
            self.get_logger().info(
                "the %s book is %s away and the marker is not in frame; not offering a "
                "bearing from a book that far off"
                % (self.book_colour,
                   "an unknown distance" if self.last_book_range is None
                   else "%.1f m" % self.last_book_range),
                throttle_duration_sec=5.0)
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
        self._watch_jump(point_optical, target)
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
        self.last_book_range = float(point[0])
        self._cross_check_row(point)

    def _publish_bin_point(self, frame) -> None:
        """Publish the collection bin's position in base_link, if it is in view.

        Only x and y come from depth. The bin is the same red as a red book -- which is
        why the shape gates in book_detector exist -- and it is large: 500 mm across the
        opening against a book 30 mm wide. So the accuracy that matters here is not the
        accuracy that matters for a grasp, and depth easily clears it: measured against
        ground truth on a settled head, 7 mm and 5 mm in range, 20 mm and 16 mm
        sideways, on a target with 250 mm of margin either side.

        Publishing nothing when the bin is not in view is the point of the topic: the
        delivery controller turns the robot until something arrives.
        """
        found = bd.detect_bin(frame)
        if found is None:
            return
        if self.depth_image is None or self.intrinsics is None or not self.depth_frame:
            return
        point_optical = dl.locate(found.bbox, self.depth_image, self.intrinsics)
        if point_optical is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                GRASP_FRAME, self.depth_frame, rclpy.time.Time())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"no transform {self.depth_frame} -> {GRASP_FRAME} for the bin: {exc}",
                throttle_duration_sec=5.0)
            return
        point = dl.transform_point(
            point_optical, tf.transform.rotation, tf.transform.translation)

        msg = PointStamped()
        msg.header.frame_id = GRASP_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(BIN_RIM_BASE_Z)
        self.pub_bin_point.publish(msg)

    def _process(self) -> None:
        self._count_frame()
        if self.latest_frame is None:
            return
        frame = self.latest_frame

        self._publish_bin_point(frame)

        try:
            books = bd.detect_books(frame)
            markers = sorted(mr.read_markers(frame), key=lambda m: m.cx)
            self._publish_shelf_column(markers)
            self._publish_shelf_yaw()
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
        if target_book is not None:
            # Remember where the RIGHT book was, and the heading it was seen from,
            # so it can be recognised again once the marker that identified it has
            # left the frame. Without the heading the next frame has nothing to
            # predict from and the tracker falls back to following itself.
            self.last_book_cx = float(target_book.cx)
            self.last_book_yaw = self.base_yaw
        # Marker, book and heading together, because the three of them are what
        # separates the possible faults. The bearing published here held at 305, 299,
        # 299 px across 25 degrees of measured base rotation, which cannot be a book
        # standing still in the world. If the marker's own cx moves and the book's does
        # not, find_book is being handed the wrong column; if neither moves while the
        # base does, the detection is older than its timestamp claims.
        self.get_logger().info(
            "steering on the %s book at %.0f px; its marker is at %.0f px, %d "
            "marker(s) in view, base heading %+.1f deg"
            % (self.book_colour or "target",
               float(steer_x), float(markers[column_index].cx), len(markers),
               math.degrees(self.base_yaw) if self.base_yaw is not None else float("nan")),
            throttle_duration_sec=2.0)
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

    def _publish_shelf_yaw(self) -> None:
        """Publish how far the base is turned away from the shelf, from the depth image.

        The approach needs a heading reference and has never had a good one. The laser
        cannot supply it: it sits 209 mm off the floor, where the shelf is an open
        compartment, and measured at 0.74 m from the front it returned 163 beams with
        none inside two metres and 110 of them on the far wall. The overhead marker
        cannot either -- it is small, it drops out constantly, and a pixel bearing says
        nothing about orientation.

        The depth camera can. Measured against Gazebo, six samples from about 2300
        points each: -34.5, -34.5, -35.3, -34.7, -34.2, -34.5 degrees against a true
        -35.9. Better than a degree and a half, and steady.

        The plane it locks onto is usually the shelf's BACK panel rather than its front,
        because at book height most of the view is open shelf. That is fine here and only
        here: the two are parallel, so either gives the same heading. It is emphatically
        not fine for distance, which is why only the angle is published.
        """
        if self.depth_image is None or self.intrinsics is None or not self.depth_frame:
            return
        if self.reported_row is None:
            return
        height = ROW_HEIGHTS_BASE[self.reported_row - 1] if (
            1 <= self.reported_row <= len(ROW_HEIGHTS_BASE)) else None
        if height is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                GRASP_FRAME, self.depth_frame, rclpy.time.Time()).transform
        except Exception:  # noqa: BLE001 - the transform may not be up yet
            return
        q, t = tf.rotation, tf.translation
        xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
        xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
        wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
        rotation = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]])
        result = sp.face_from_depth(
            self.depth_image, self.intrinsics, rotation,
            np.array([t.x, t.y, t.z]), float(height))
        if result is None:
            return
        _, yaw, _ = result
        self.pub_shelf_yaw.publish(Float32(data=float(yaw)))

    def _publish_shelf_column(self, markers: List[mr.Marker]) -> None:
        """Publish WHICH COLUMN of the shelf carries the target marker, 1 to 5.

        This is not _target_column_index, and the difference is a scored point. That one
        returns a position among the columns currently in frame, which is what the
        steering needs and is meaningless to anybody else: driving along the shelf with
        two markers in view it reads 0 or 1, whichever two those are. Published to the
        judges it would be a number with no relation to the shelf.

        The absolute answer needs the whole shelf in one frame: five markers, all
        confident, their digits a permutation of 1 to 5. Anything less and some column is
        off the edge of the image, so counting from the left counts from the wrong place.
        From the start zone the whole unit is in view, which is where this fires.

        Latched once found. The digits are fixed for the run, so a later view from close
        up cannot improve on a reading taken with everything visible, and can easily make
        it worse.

        Which end is column 1 is a parameter, not an assumption. The rules do not say,
        and the same ambiguity applies to the rows. Left to right in an image of the
        robot facing the shelf is the simulator's own numbering -- book_col_1 stands at
        y=+2.0 and book_col_5 at y=-1.9 -- which is the best evidence available.
        """
        if self.shelf_column is not None:
            self.pub_shelf_column.publish(Int32(data=int(self.shelf_column)))
            return
        if not self.target_digit or len(markers) != COLUMNS_ON_SHELF:
            return
        if not all(m.confident for m in markers):
            return
        ordered = sorted(markers, key=lambda m: m.cx)
        digits = [m.digit for m in ordered]
        if sorted(digits) != list(range(1, COLUMNS_ON_SHELF + 1)):
            self.get_logger().warn(
                "five markers in view but their digits are %s, which is not a "
                "permutation of 1-%d; not identifying the column from this frame"
                % (digits, COLUMNS_ON_SHELF), throttle_duration_sec=10.0)
            return
        position = digits.index(self.target_digit) + 1
        if not self.columns_left_to_right:
            position = COLUMNS_ON_SHELF + 1 - position
        self.shelf_column = position
        self.get_logger().info(
            "the whole shelf is in view, markers left to right are %s, so marker %d "
            "is shelf column %d" % (digits, self.target_digit, position))
        self.pub_shelf_column.publish(Int32(data=int(position)))

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
        # Only believe a digit read near the middle of the picture.
        #
        # The camera's half-angle is 43.5 degrees -- fx 337.2 on a 640-wide image -- and
        # a marker at the very edge of that is a flat plate seen at 43 degrees of
        # obliquity, a few pixels of ink, foreshortened to almost nothing.
        #
        # Watched over one run, EVERY acquisition happened there: the approach found
        # "marker 3" at +282, +312, +278, +298, +313, +300 and -313 px, which is 40 to
        # 43 degrees every single time, and then turned 24 degrees without the reading
        # moving. Markers are about 18 degrees apart at that range, so what was being
        # tracked was not one marker being followed but a succession of different ones
        # arriving at the edge and being read, wrongly, as the target.
        #
        # Inside 30 degrees the plate is square enough to read and there is room to
        # turn towards it. SEARCH simply keeps rotating until one is properly in view,
        # which is what it is for.
        limit = self.marker_edge_px
        hits = [i for i, m in enumerate(markers)
                if m.digit == self.target_digit and m.confident
                and abs(m.cx - self.image_centre_px) <= limit]
        self._tally(markers, hits)
        if len(hits) != 1:
            edge = [m.digit for m in markers
                    if m.digit == self.target_digit and m.confident
                    and abs(m.cx - self.image_centre_px) > limit]
            if edge and not hits:
                self.get_logger().info(
                    "marker %s is in view but %.0f px off centre, too oblique to "
                    "trust; turning further" % (self.target_digit, min(
                        abs(m.cx - self.image_centre_px) for m in markers
                        if m.digit == self.target_digit and m.confident)),
                    throttle_duration_sec=3.0)
            return None  # absent, too oblique, or read twice -- move for a better view
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
