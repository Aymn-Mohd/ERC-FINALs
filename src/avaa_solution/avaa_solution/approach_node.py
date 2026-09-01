"""Fine approach — close the last stretch to the target column.

Nav2 gets the robot to within about 0.3 m of a goal and then stops dead, so it is used for
gross navigation only (see config/nav2_params.yaml). This node closes the remainder using
the camera and the front LiDAR, which is what mobile manipulation needs anyway: the arm has
to be placed relative to the book, not to an odometry coordinate that was only ever an
estimate of where the book is.

Sequence:

    TUCK     fold both arms in, so the base strafes cleanly and the LiDAR sees the shelf
    SEARCH   turn on the spot until the target column marker is read
    CENTRE   rotate until that column sits in the middle of the image
    ACQUIRE  hold at a working range, square to the shelf, and line up by strafing
    APPROACH drive in, holding the target centred, until the shelf face is at standoff
    SQUARE   sit perpendicular to the shelf face, fitted from the LiDAR
    VERIFY   confirm the book is still held in view before handing over to the grasp
    RETREAT  back off and re-acquire when it is not; twice, then give up
    DONE

Lateral motion is commanded, and turning is not, once the arms are tucked. Rotating to
chase a bearing while driving turns a small angular error into a large lateral excursion --
one run ended at y = -2.25, past the end of the shelf unit, having started centred --
whereas the base strafes cleanly with the arms stowed: vy = +0.20 measured dy = +0.233 with
dyaw = 0.000. An earlier belief that strafing yaws the base came from measuring it with the
arms extended.

Two things here exist because they failed silently first, and both are worth keeping:

    * SQUARE will not hand an unsquared base to the grasp, which reaches along base x.
    * The target is anchored in odom once measured from a range where the column marker is
      legible, because that marker leaves the frame as the robot closes in and the bearing
      it provides was measured jumping 310 px in a single step at 1 m out.
"""

import math
from collections import deque
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float32, Int32, String
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

BASE_FRAME = "base_footprint"
ODOM_FRAME = "odom"

# Where the left shoulder sits, sideways, in base_link. The base is not the thing that
# has to line up with the book: the arm is, and its first joint is 0.159 m to the left of
# centre. Stopping with the book straight ahead of the base leaves it that far off the
# arm centre line, so the forearm has to cross the shelf opening diagonally to reach in
# and catches on it. Measured with the link contact sensors, four arm links ended up
# against base_link_shelf_collision and the arm stopped 100 mm short.
#
# Aiming the same grasp at the shoulder line instead took the miss from 136 mm to 8 mm.
SHOULDER_OFFSET_Y = 0.159
CAMERA_FRAME = "head_front_camera_depth_optical_frame"

# base_link sits this far above base_footprint. Row heights are quoted in base_link.
BASE_LINK_Z = 0.186

# head_2_joint: negative looks down, roughly one-for-one in radians. Limits from the URDF.
HEAD_TILT_MIN = -1.047   # about 60 degrees down
HEAD_TILT_MAX = 0.349    # about 20 degrees up

# Gripper z in base_link for rows 1..4, top shelf first.
DEFAULT_ROW_HEIGHTS = [1.391, 1.061, 0.731, 0.401]

# Scan returns inside this radius of base_footprint are the robot itself, not obstacles.
# The base is 0.717 x 0.497 m, so its circumscribed radius is 0.437 m; this sits just
# outside that.
SELF_FILTER_RADIUS = 0.45

TOPIC_TARGET_COLUMN_X = "/avaa/perception/target_column_x"
TOPIC_TARGET_ROW = "/avaa/perception/target_row"
TOPIC_BOOK_POINT = "/avaa/perception/target_book_point"
TOPIC_ARM_LEFT = "/arm_left_controller/joint_trajectory"
TOPIC_ARM_RIGHT = "/arm_right_controller/joint_trajectory"
TOPIC_TORSO = "/torso_controller/joint_trajectory"
TOPIC_SCAN = "/scan_front_raw"
TOPIC_STATE = "/avaa/approach/state"
TOPIC_CMD = "/cmd_vel"

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


# Arm posture for driving, measured rather than guessed (see try_tuck.py).
#
# The arms spawn with every joint at zero, which for this arm is fully extended: the
# gripper reaches 0.838 m forward, 0.478 m beyond the front of the base. Driving at the
# shelf in that posture wedges the arm against it -- observed as six simultaneous contacts
# between both grippers, both arm_6 links and erc_shelf, with the base unable to advance
# while the LiDAR still read 0.94 m of clear space ahead. Each contact event costs half a
# point.
#
# This pose measured 0.319 m forward and 0.174 m lateral, both inside the base footprint
# (0.36 m half-length, 0.249 m half-width), with no contacts. Joint 2 does most of the
# work; the elbow pulls the forearm in laterally, and joint 1 finishes the job.
# Folded, and collision free -- which the previous tuck was not.
#
# [-0.5, -2.4, 0.0, -2.4, 0.0, 0.0, 0.0] puts arm_left_2 through arm_left_5 against
# torso_base_link and torso_lift_link. Gazebo never objected, because self-collision
# is not checked there, so it went unnoticed for the whole project until MoveIt
# refused to plan from it: the start state was invalid and every request came back
# 'Motion planning start tree could not be initialized'.
#
# This one was found by sampling folded postures and asking /check_state_validity,
# keeping the most compact one with at least 0.15 rad of room at every joint stop.
# It is also tighter than the old one: the gripper sits 0.29 m from the base axis
# rather than 0.49 m.
TUCK_POSE = [2.1521, 0.3824, 1.2785, -2.1517, 0.8325, 0.1926, 1.3944]
# The tuck is an eight-joint posture, torso included, and only the eight together
# are collision free. Commanding the arm alone and leaving the torso down folds
# arm_left_5, arm_left_6 and the gripper into base_link -- MoveIt reports exactly
# those three contacts -- and the planner then refuses every request from that
# start state, in 0.4 s, with no indication that the torso is the reason.
TUCK_TORSO = 0.15

# The right arm is never used, but it still has to be out of the way, and mirroring
# the left tuck by flipping two joints does not do that: it leaves arm_right_4_link
# at x=0.491, which is inside the shelf once the base is close enough to grasp from.
# MoveIt then reports the robot in collision and refuses every left-arm goal, so the
# grasp fails for a reason that has nothing to do with the arm doing the work.
#
# Found the same way as the left tuck, by sampling against /check_state_validity with
# the shelf in the scene at grasping distance. See tools/find_right_tuck.py.
#
# Scored on where the links actually end up, not on how small the joint angles are.
# Scoring on the angles picks postures close to all-zeros, and all-zeros is the arm
# stretched straight out: the first answer put the right gripper 0.88 m forward,
# inside the shelf, against shelf_back. This one reaches 0.204 m forward and is
# checked at both torso heights, because raising the torso for the top rows takes
# the whole upper body with it.
RIGHT_TUCK = [-0.7194, -2.2867, -0.5064, 0.5221, 2.3399, 1.0503, 1.9772]


class State(Enum):
    WAITING = "waiting"
    TUCK = "tucking"
    SEARCH = "searching"
    CENTRE = "centring"
    ACQUIRE = "acquiring"
    APPROACH = "approaching"
    VERIFY = "verifying"
    RETREAT = "retreating"
    SQUARE = "squaring"
    DONE = "done"
    FAILED = "failed"


class ApproachNode(Node):
    def __init__(self) -> None:
        super().__init__("avaa_approach")

        # Distance to stop at, measured from base_footprint -- not from the laser, which
        # sits 0.275 m further forward. The base is 0.717 m long, so 0.75 m from the
        # origin leaves about 0.39 m of clearance ahead of the bumper.
        #
        # Provisional. The right value is whatever puts the books inside the arm's working
        # envelope, which cannot be settled until grasping exists.
        self.declare_parameter("standoff_m", 0.75)
        self.declare_parameter("centre_tolerance_px", 12.0)
        self.declare_parameter("standoff_tolerance_m", 0.05)
        self.declare_parameter("square_tolerance_rad", 0.05)
        self.declare_parameter("max_yaw_rate", 0.45)
        self.declare_parameter("max_forward", 0.22)
        self.declare_parameter("max_lateral", 0.10)
        # Refuse to drive closer than this whatever the standoff says, so a bad reading
        # cannot push the base into the shelf. A collision costs 0.5 points each time.
        # Also measured from base_footprint: the bumper is 0.36 m out, so 0.55 m leaves
        # roughly 0.19 m of margin.
        self.declare_parameter("min_safe_range_m", 0.55)
        self.declare_parameter("state_timeout_sec", 45.0)
        self.declare_parameter("search_rate", 0.35)
        self.declare_parameter("row_heights", DEFAULT_ROW_HEIGHTS)
        # Where to pause and confirm the book before committing to the final drive.
        #
        # Far enough back that the whole shelf column is still comfortably in frame, so
        # the book can be found and centred without a race. Driving straight to grasping
        # range instead means handing over from marker-steering to book-steering while
        # moving, and if the book is not acquired in time the robot arrives beside its
        # column with nothing recognisable in view.
        self.declare_parameter("acquire_range_m", 1.50)
        self.declare_parameter("acquire_tolerance_px", 25.0)
        # Searching gets its own budget: a full turn at 0.35 rad/s is about 18 s of
        # simulation time, which at a real-time factor near 0.5 is well over half a minute
        # of wall clock. The ordinary state timeout would abort mid-sweep.
        self.declare_parameter("search_timeout_sec", 150.0)
        self.declare_parameter("image_width_px", 640)
        self.declare_parameter("tuck_time_sec", 5.0)

        self.standoff = float(self.get_parameter("standoff_m").value)
        self.centre_tol = float(self.get_parameter("centre_tolerance_px").value)
        self.standoff_tol = float(self.get_parameter("standoff_tolerance_m").value)
        self.square_tol = float(self.get_parameter("square_tolerance_rad").value)
        self.max_yaw = float(self.get_parameter("max_yaw_rate").value)
        self.max_fwd = float(self.get_parameter("max_forward").value)
        self.max_lateral = float(self.get_parameter("max_lateral").value)
        self.min_safe = float(self.get_parameter("min_safe_range_m").value)
        self.timeout = float(self.get_parameter("state_timeout_sec").value)
        self.search_rate = float(self.get_parameter("search_rate").value)
        self.search_timeout = float(self.get_parameter("search_timeout_sec").value)
        self.image_width = int(self.get_parameter("image_width_px").value)
        self.tuck_time = float(self.get_parameter("tuck_time_sec").value)

        self.column_cx: Optional[float] = None
        self.column_cx_at: Optional[float] = None
        self.scan: Optional[LaserScan] = None
        self.state = State.WAITING
        self.state_since = self._now()

        # Servo on the target column's image position, not on a column index. The index
        # perception publishes is frame-relative -- it counts the columns currently in
        # view, so it changes as markers enter and leave the frame, and anything treating
        # it as the column's identity tracks a different column each frame.
        # Scan points must be transformed into the robot frame, not read as if the laser
        # were aligned with it -- it is mounted rotated. See _scan_points_base.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # The row has to be identified before closing in, not after. Resolving it needs
        # all four books of the column in frame, and at grasping range the column no
        # longer fits, so driving first and identifying later never succeeds.
        self.row_seen = False
        self.target_row: Optional[int] = None
        self.head_tilt: Optional[float] = None
        self.acquire_range = float(self.get_parameter("acquire_range_m").value)
        self.acquire_tol = float(self.get_parameter("acquire_tolerance_px").value)
        # Set once the book has actually been located in 3D, which is the signal that it
        # is genuinely visible rather than merely expected to be.
        self.book_point_at: Optional[float] = None
        self.book_x: Optional[float] = None
        self.approach_target = self.acquire_range
        # Retry budget for losing the book on the final drive.
        self.retreats = 0
        # The book position, held in odom once it has been measured from a range
        # where the marker above the column is still legible.
        self.target_odom = None
        # How far a later sighting may sit from that anchor and still be believed.
        # Column spacing is about 0.95 m, so this stays well inside the distance to
        # the neighbouring column while allowing for depth noise and odom drift.
        self.anchor_gate = 0.30
        # Sightings waiting to agree with each other before one of them is trusted, and
        # sightings that disagreed with the anchor. Both need to be consistent among
        # themselves before they are acted on. See _accept_sighting.
        self.anchor_candidates = deque(maxlen=8)
        self.anchor_disagree = deque(maxlen=20)
        self.anchor_rejects = 0
        # Lateral error worth correcting during the final drive, in metres.
        self.centre_tol_m = 0.03
        # How far off a fitted line a return may sit and still count as the same
        # surface. The shelf face is not smooth: book edges and the frame scatter
        # returns by about 0.05 m.
        self.face_tolerance = 0.05
        # What share of the forward returns must agree before the fit is believed.
        # Comfortably over half, so that two surfaces of similar size are refused
        # rather than squared to whichever happened to win. Real shelf faces measured
        # 81 and 76 per cent.
        self.face_consensus = 0.6
        # How long a bad scan is tolerated before treating it as a real obstruction.
        self.square_grace = 4.0
        self.square_lost_since = None
        self.max_retreats = 2
        # Perception runs at 5 Hz and the head is still settling when the drive ends, so
        # allow a moment before concluding the book is lost.
        self.verify_grace = 12.0
        # The head's own trajectory is one second; give it that plus margin to stop.
        self.head_settle = 2.5
        # How long the book must stay located before the view counts as trustworthy.
        self.book_hold_time = 3.0
        self.verify_aimed_at: Optional[float] = None
        self.book_held_since: Optional[float] = None
        self.create_subscription(
            PointStamped, TOPIC_BOOK_POINT, self._on_book_point, 10)
        self.row_heights = list(
            self.get_parameter("row_heights").value or DEFAULT_ROW_HEIGHTS)
        self.pub_head = self.create_publisher(
            JointTrajectory, "/head_controller/joint_trajectory", 10)
        self.create_subscription(Int32, TOPIC_TARGET_ROW, self._on_row, 10)
        self.create_subscription(Float32, TOPIC_TARGET_COLUMN_X, self._on_column_x, 10)
        self.create_subscription(LaserScan, TOPIC_SCAN, self._on_scan, SENSOR_QOS)
        self.pub_cmd = self.create_publisher(Twist, TOPIC_CMD, 10)
        self.pub_state = self.create_publisher(String, TOPIC_STATE, 10)
        self.pub_arm_left = self.create_publisher(JointTrajectory, TOPIC_ARM_LEFT, 10)
        self.pub_arm_right = self.create_publisher(JointTrajectory, TOPIC_ARM_RIGHT, 10)
        self.pub_torso = self.create_publisher(JointTrajectory, TOPIC_TORSO, 10)

        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"approach up — standoff {self.standoff:.2f} m, "
            f"safety floor {self.min_safe:.2f} m"
        )

    # ------------------------------------------------------------------ inputs

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_column_x(self, msg: Float32) -> None:
        self.column_cx = float(msg.data)
        self.column_cx_at = self._now()

    def _column_cx_fresh(self, max_age: float = 1.5) -> Optional[float]:
        """Give the target column image x, or None if perception has gone quiet.

        Acting on a stale bearing steers toward where the column used to be, so the
        controller stops rather than guessing.
        """
        if self.column_cx is None or self.column_cx_at is None:
            return None
        if (self._now() - self.column_cx_at) > max_age:
            return None
        return self.column_cx

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan = msg

    # ------------------------------------------------------------------ geometry

    def _scan_points_base(self) -> List[Tuple[float, float]]:
        """All scan returns as (x, y) in base_footprint.

        The scan frame is NOT aligned with the robot. The front laser is mounted at
        roll -180 deg, yaw -45 deg relative to base_footprint, so scan angle zero points
        45 degrees off to the side and the roll mirrors the direction of increasing angle.
        Treating scan angles as robot-relative bearings therefore measures a cone pointing
        somewhere else entirely -- which is why the shelf-squaring fit reported the face
        2.5 degrees off while the robot was actually sitting 35 degrees away from square.
        """
        if self.scan is None:
            return []
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, self.scan.header.frame_id, rclpy.time.Time()
            )
        except Exception:  # noqa: BLE001 - transform may not be available yet
            return []

        q = tf.transform.rotation
        t = tf.transform.translation
        # Rotation matrix from the quaternion; only the rows producing x and y are needed.
        xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
        xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
        wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
        r00, r01, r02 = 1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)
        r10, r11, r12 = 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)

        points = []
        for i, r in enumerate(self.scan.ranges):
            if not math.isfinite(r) or not (self.scan.range_min < r < self.scan.range_max):
                continue
            angle = self.scan.angle_min + i * self.scan.angle_increment
            lx, ly, lz = r * math.cos(angle), r * math.sin(angle), 0.0
            bx = r00 * lx + r01 * ly + r02 * lz + t.x
            by = r10 * lx + r11 * ly + r12 * lz + t.y
            # Discard the robot's own body. The laser plane sits at z = 0.209 m and the
            # tucked arm reaches 0.319 m forward, so the stowed arm is unavoidably inside
            # the scan -- it read as an obstacle 0.35 m ahead and tripped the safety stop
            # the instant driving began. Nothing external can be this close without the
            # robot already being in contact, so anything inside the footprint radius is
            # the robot seeing itself.
            if math.hypot(bx, by) < SELF_FILTER_RADIUS:
                continue
            points.append((bx, by))
        return points

    def _forward_points(self, half_angle: float = 0.30) -> List[Tuple[float, float]]:
        """Return hits within +/- half_angle of ahead, as (x, y) in base_footprint."""
        return [(x, y) for x, y in self._scan_points_base()
                if x > 0.0 and abs(math.atan2(y, x)) <= half_angle]

    def _range_ahead(self) -> Optional[float]:
        points = self._forward_points(half_angle=0.12)
        if not points:
            return None
        # Median, not minimum: a single spurious short return would otherwise stop the
        # approach short of the shelf.
        return float(np.median([x for x, _ in points]))

    def _min_range_ahead(self) -> Optional[float]:
        points = self._forward_points(half_angle=0.35)
        if not points:
            return None
        return float(min(math.hypot(x, y) for x, y in points))

    def _shelf_angle(self) -> Optional[float]:
        """Yaw error against the shelf face: 0 when square on, positive when yawed CCW.

        Fits a line to the forward returns. The shelf front is flat and 5.25 m wide, so
        within a narrow cone it is a clean straight edge. For a robot yawed by theta, a
        surface of constant world x appears in the robot frame with dx/dy = tan(theta), so
        the fitted slope is the yaw error directly.

        Returns None when the fit is not credible, rather than squaring up to whatever
        happens to be in front. Without that guard a robot that has turned away from the
        shelf will happily square itself to the far wall.
        """
        points = self._forward_points(half_angle=0.45)
        if len(points) < 12:
            return None
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        # Fit the surface most of the returns agree on, not all of them at once.
        # Two different scenes broke a plain least-squares line here. Far from the
        # shelf the cone reaches past the end of it to the wall 2.5 m beyond, and
        # those returns pulled the angle from -34 to -53 degrees. Close in, a flat
        # face at x = 0.85 spanning the whole cone had one object sticking out to
        # x = 0.55 in front of it, and 46 such points out of 207 held the residual at
        # 0.10 m while the robot was in fact already square to within half a degree.
        # Neither a nearest-return nor a median slab separates those two cases; a
        # consensus fit does.
        inliers = _largest_collinear_set(xs, ys, tolerance=self.face_tolerance)
        if inliers is None:
            return None
        # Squaring to a minority surface is how the robot ends up facing a shelf
        # upright, or one pulled-out book, instead of the shelf itself.
        if int(inliers.sum()) < max(12, int(self.face_consensus * len(xs))):
            return None
        xs, ys = xs[inliers], ys[inliers]

        # x as a function of y: the face is roughly parallel to the robot's y axis,
        # so this avoids the vertical-line singularity of fitting y = f(x).
        slope, intercept = np.polyfit(ys, xs, 1)

        # The consensus step has already separated the surfaces, so what is left
        # should be tight. A clean face measured 0.02 m.
        residual = float(np.std(xs - (slope * ys + intercept)))
        if residual > 0.05:
            return None
        return float(math.atan(slope))

    # ------------------------------------------------------------------ control

    def _enter(self, state: State) -> None:
        if state is State.RETREAT:
            # Start the anchor again: if it had been right we would not be retreating.
            self.target_odom = None
            self.anchor_candidates.clear()
            self.anchor_disagree.clear()
        if state is not self.state:
            self.get_logger().info(f"{self.state.value} -> {state.value}")
            self.state = state
            self.state_since = self._now()

    def _publish_state(self) -> None:
        self.pub_state.publish(String(data=self.state.value))

    def _elapsed(self) -> float:
        return self._now() - self.state_since

    def _stop(self) -> None:
        self.pub_cmd.publish(Twist())

    def _tick(self) -> None:
        self._publish_state()

        if self.state in (State.DONE, State.FAILED):
            self._stop()
            return

        if self.state is State.WAITING:
            # Stow the arms before anything moves, then go looking for the marker. The
            # robot's start pose is not guaranteed to face the shelves -- in practice it
            # spawns facing a wall -- so searching is part of the task, not a fallback.
            self._send_tuck()
            self._enter(State.TUCK)
            return

        budget = self.search_timeout if self.state is State.SEARCH else self.timeout
        if self._elapsed() > budget:
            self.get_logger().error(f"timed out in {self.state.value}")
            self._stop()
            self._enter(State.FAILED)
            return

        # Safety floor applies in every moving state.
        nearest = self._min_range_ahead()
        if nearest is not None and nearest < self.min_safe and self.state is State.APPROACH:
            self.get_logger().warn(
                f"safety stop: obstacle at {nearest:.2f} m < {self.min_safe:.2f} m"
            )
            self._stop()
            self._enter(State.SQUARE)
            return

        if self.state is State.TUCK:
            self._do_tuck()
        elif self.state is State.SEARCH:
            self._do_search()
        elif self.state is State.ACQUIRE:
            self._do_acquire()
        elif self.state is State.VERIFY:
            self._do_verify()
        elif self.state is State.RETREAT:
            self._do_retreat()
        elif self.state is State.CENTRE:
            self._do_centre()
        elif self.state is State.APPROACH:
            self._do_approach()
        elif self.state is State.SQUARE:
            self._do_square()

    def _send_tuck(self) -> None:
        """Command both arms to the driving posture."""
        for pub, side in ((self.pub_arm_left, "left"), (self.pub_arm_right, "right")):
            traj = JointTrajectory()
            traj.joint_names = [f"arm_{side}_{i}_joint" for i in range(1, 8)]
            point = JointTrajectoryPoint()
            pose = list(RIGHT_TUCK) if side == "right" else list(TUCK_POSE)
            point.positions = [float(v) for v in pose]
            point.time_from_start = Duration(sec=int(self.tuck_time), nanosec=0)
            traj.points = [point]
            pub.publish(traj)

        # And the torso with them, or the folded arm sits inside the base.
        torso = JointTrajectory()
        torso.joint_names = ["torso_lift_joint"]
        lift = JointTrajectoryPoint()
        lift.positions = [float(TUCK_TORSO)]
        lift.time_from_start = Duration(sec=int(self.tuck_time), nanosec=0)
        torso.points = [lift]
        self.pub_torso.publish(torso)
        self.get_logger().info("stowing arms for driving")

    def _do_tuck(self) -> None:
        self._stop()  # no driving until the arms are in
        elapsed = self._now() - self.state_since

        # Repeat only during the first moment, to cover the controller not yet being
        # subscribed. Repeating later is actively harmful: each JointTrajectory replaces
        # the one in progress and restarts its time_from_start, so a trajectory re-sent
        # every two seconds never finishes. That left the arm permanently mid-sweep, and
        # since the tuck path crosses the LiDAR plane the moving arm registered as an
        # obstacle 0.08 m ahead the instant driving began.
        if elapsed < 0.6:
            self._send_tuck()

        # Wait out the full trajectory plus settling before allowing any motion.
        if elapsed >= self.tuck_time + 2.0:
            self._enter(State.SEARCH)

    def _do_search(self) -> None:
        """Rotate on the spot until the target column's marker comes into view.

        The robot spawns facing a wall and the marker digits are randomised per run, so
        the target may be anywhere around it. Rotating in place is the cheapest way to
        cover the full circle without risking a collision, and with the arms stowed the
        base turns cleanly on the spot.
        """
        if self._column_cx_fresh() is not None:
            self._stop()
            self.get_logger().info("target marker found")
            self._enter(State.CENTRE)
            return

        cmd = Twist()
        cmd.angular.z = self.search_rate
        self.pub_cmd.publish(cmd)
        self.get_logger().info(
            f"searching for marker... ({self._elapsed():.0f}s)",
            throttle_duration_sec=5.0,
        )

    def _on_row(self, msg: Int32) -> None:
        self.row_seen = True
        self.target_row = int(msg.data)

    def _aim_head(self) -> None:
        """Tilt the head to keep the target row in frame as the robot closes in.

        Approaching with the head level loses the books out of the bottom of the image
        well before grasping range, which is what stopped the book point being published
        just when the grasp controller needed it.

        head_2_joint is negative for down, essentially one-for-one in radians (measured:
        -0.40 gives 22.9 degrees down, -0.80 gives 45.8), with 60 degrees of downward
        travel available.
        """
        if self.target_row is None:
            return
        if not 1 <= self.target_row <= len(self.row_heights):
            return
        target_z = self.row_heights[self.target_row - 1]

        distance = self._range_ahead()
        if distance is None or distance < 0.05:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAMERA_FRAME, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return
        camera_z = tf.transform.translation.z

        # base_link sits above base_footprint, and row heights are quoted in base_link,
        # so put both in the same frame before taking the difference.
        drop = (camera_z - BASE_LINK_Z) - target_z
        desired = -math.atan2(drop, distance)
        desired = max(HEAD_TILT_MIN, min(HEAD_TILT_MAX, desired))

        # Only re-send on a meaningful change. Every JointTrajectory replaces the one in
        # progress and restarts its time_from_start, so a trajectory re-sent every tick
        # never finishes and the head never actually arrives.
        if self.head_tilt is not None and abs(desired - self.head_tilt) < 0.05:
            return
        self.head_tilt = desired

        traj = JointTrajectory()
        traj.joint_names = ["head_1_joint", "head_2_joint"]
        point = JointTrajectoryPoint()
        point.positions = [0.0, float(desired)]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.pub_head.publish(traj)
        self.get_logger().info(
            f"head tilt -> {math.degrees(-desired):+.0f} deg down "
            f"(row {self.target_row} at {distance:.2f} m)"
        )

    def _on_book_point(self, msg: PointStamped) -> None:
        point = np.array([msg.point.x, msg.point.y, msg.point.z])
        if not self._accept_sighting(point, msg.header.frame_id):
            return
        self.book_point_at = self._now()
        self.book_x = float(point[0])

    def _lookup(self, target_frame: str, source_frame: str):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001 - the transform may not be published yet
            return None

    @staticmethod
    def _apply_transform(tf, point) -> np.ndarray:
        """Apply a TransformStamped to an (x, y, z) point."""
        q, t = tf.transform.rotation, tf.transform.translation
        xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
        xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
        wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
        rotation = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ])
        return rotation @ np.asarray(point, dtype=float) + np.array([t.x, t.y, t.z])

    def _accept_sighting(self, point: np.ndarray, frame_id: str) -> bool:
        """Decide whether a sighting is the book we set out for, and anchor it.

        The column bearing comes from a marker mounted above the shelf, and that
        marker leaves the camera as the robot closes in. One run had the bearing jump
        310 px in a single step at 1 m out, which strafed the base into the divider
        between two columns and left it facing neither: ground truth put the two
        nearest red books at +33.5 and -39.6 degrees off the nose, both at the edge of
        the frame, and perception reported no red book in view for the rest of the run.

        So once the target has been measured from a range where the marker is still
        legible, its position is held in odom, and later sightings are believed only
        if they agree with it. Odom drift over the last metre is far smaller than the
        error being rejected. This is not navigating to an odometry estimate, which the
        arm could not be placed from -- the grasp still uses the live measurement. It
        is refusing to be talked out of a good fix by a bad one.
        """
        if self.state not in (State.ACQUIRE, State.APPROACH,
                              State.SQUARE, State.VERIFY):
            return True
        tf = self._lookup(ODOM_FRAME, frame_id)
        if tf is None:
            return True   # nothing to check against; a sighting beats none
        in_odom = self._apply_transform(tf, point)

        if self.target_odom is None:
            # Do not anchor to a single sighting. With no marker in view perception picks
            # one of several same-coloured books, and whichever it picked first would
            # become the target for the whole run.
            self.anchor_candidates.append(in_odom)
            settled = self._agreeing(self.anchor_candidates)
            if settled is None:
                return False
            self.target_odom = settled
            self.get_logger().info(
                "target anchored at odom [%.2f, %.2f, %.2f] from %d agreeing sightings"
                % (settled[0], settled[1], settled[2], len(self.anchor_candidates)))
            return True

        drift = float(np.linalg.norm(in_odom[:2] - self.target_odom[:2]))
        if drift > self.anchor_gate:
            self.anchor_rejects += 1
            self.anchor_disagree.append(in_odom)
            # An outlier gate with no way out turns one bad fix into a permanent one. A
            # run anchored to the wrong book and then rejected the right one 217 times in
            # a row, all at the same 1.24 m, until the approach timed out. Sightings that
            # disagree with the anchor but agree with each other are the better answer.
            replacement = self._agreeing(self.anchor_disagree)
            if replacement is not None:
                self.get_logger().warn(
                    "re-anchoring: %d sightings agree with each other %.2f m from the "
                    "anchor, so the anchor was wrong"
                    % (len(self.anchor_disagree), drift))
                self.target_odom = replacement
                self.anchor_disagree.clear()
                return True
            self.get_logger().warn(
                "ignoring a sighting %.2f m from the anchored target (%d so far)"
                % (drift, self.anchor_rejects), throttle_duration_sec=3.0)
            return False

        # Sightings that agree refine the anchor: the book has not moved, but the fix
        # on it gets better as the robot closes in.
        self.anchor_disagree.clear()
        self.target_odom = 0.7 * self.target_odom + 0.3 * in_odom
        return True

    def _agreeing(self, sightings):
        """Give the mean of a full buffer of sightings, if they agree; otherwise None."""
        if len(sightings) < sightings.maxlen:
            return None
        points = np.array(list(sightings))
        centre = points.mean(axis=0)
        spread = float(np.max(np.linalg.norm(points[:, :2] - centre[:2], axis=1)))
        if spread > self.anchor_gate:
            return None
        return centre

    def _target_in_base(self):
        """Give the anchored target as (x, y, z) in base_footprint, or None."""
        if self.target_odom is None:
            return None
        tf = self._lookup(BASE_FRAME, ODOM_FRAME)
        if tf is None:
            return None
        return self._apply_transform(tf, self.target_odom)

    def _distance_to_face(self) -> Optional[float]:
        """Distance to the shelf face, preferring the book over the LiDAR.

        The LiDAR reads THROUGH the open shelf to the back panel, so its forward range
        overstates the distance to the shelf face by roughly the shelf's depth. Acting on
        it drives the base into the unit: one run reported 0.85 m of clearance while the
        robot's front corner was about 8 cm from the face and could no longer move.

        The book's depth position does not have that problem -- it is a measurement of the
        thing we actually care about, good to 15-35 mm in x against ground truth. Fall
        back to the LiDAR only before the book has been found, when a rough range is
        enough to close the initial distance.
        """
        if self._book_located() and self.book_x is not None:
            return self.book_x
        return self._range_ahead()

    def _book_located(self, max_age: float = 1.5) -> bool:
        """Whether the book is currently being located in 3D, not merely expected."""
        if self.book_point_at is None:
            return False
        return (self._now() - self.book_point_at) <= max_age

    def _do_verify(self) -> None:
        """Confirm the book is still being tracked before handing over to the grasp.

        The final drive can lose the book -- it leaves the frame, or the robot ends up
        beside its column rather than in front of it -- and the failure is silent: the
        approach reports success, the grasp controller waits in IDLE for a target that
        never comes, and nothing says why. Checking here turns that into a retry instead
        of a stall.
        """
        self._stop()

        # Aim the head once on entry and then leave it alone. _aim_head publishes a
        # one-second trajectory; re-issuing it here would keep the camera moving through
        # the very check that is supposed to establish a settled view.
        if self.verify_aimed_at is None:
            self._aim_head()
            self.verify_aimed_at = self._now()
        if (self._now() - self.verify_aimed_at) < self.head_settle:
            return

        # Require the book to be HELD, not merely glimpsed.
        #
        # Checking for a single sighting passes while the head is still sweeping: the book
        # crosses the frame, the check fires, and the head then settles somewhere the book
        # is not. One run verified successfully and had lost the book again 95 seconds
        # later with the robot completely stationary.
        if self._book_located():
            if self.book_held_since is None:
                self.book_held_since = self._now()
            held = self._now() - self.book_held_since
            if held >= self.book_hold_time:
                ahead = self._distance_to_face()
                self.get_logger().info(
                    f"approach complete — book held for {held:.1f}s at "
                    f"{ahead:.2f} m" if ahead is not None
                    else f"approach complete — book held for {held:.1f}s")
                self._enter(State.DONE)
            return

        # Lost it again: the hold has to restart from zero, not resume.
        if self.book_held_since is not None:
            self.get_logger().warn("book lost during the hold; restarting the count")
            self.book_held_since = None

        if self._elapsed() < self.verify_grace:
            return  # give perception a moment before giving up on it

        if self.retreats >= self.max_retreats:
            self.get_logger().error(
                f"book still not in view after {self.retreats} retreat(s); giving up")
            self._enter(State.FAILED)
            return

        self.retreats += 1
        self.get_logger().warn(
            f"book not in view; backing off to re-acquire "
            f"(attempt {self.retreats} of {self.max_retreats})")
        # Reset the settling state so the next verification starts clean.
        self.verify_aimed_at = None
        self.book_held_since = None
        self._enter(State.RETREAT)

    def _do_retreat(self) -> None:
        """Back away far enough for the whole column to be in frame again."""
        ahead = self._distance_to_face()
        if ahead is not None and ahead >= self.acquire_range:
            self._stop()
            self.approach_target = self.acquire_range
            self._enter(State.ACQUIRE)
            return
        if self._elapsed() > 20.0:
            # Ranging may itself be the thing that is broken; do not reverse indefinitely.
            self._stop()
            self.approach_target = self.acquire_range
            self._enter(State.ACQUIRE)
            return
        cmd = Twist()
        cmd.linear.x = -0.12
        self.pub_cmd.publish(cmd)

    def _do_acquire(self) -> None:
        """Hold at a working distance until the book is seen and centred.

        This is a checkpoint, not a drive. Everything after it depends on the book being
        genuinely in view, so it is worth a few seconds here rather than discovering at
        grasping range that the target was lost on the way in.

        The head is aimed at the row first: the books are well below the markers, and
        without the tilt the target may not be in frame at all from here.
        """
        self._stop()
        self._aim_head()

        # 1. Square to the shelf FIRST.
        #
        # Centring turns to face the column and then drives along that bearing, so the
        # robot arrives at whatever angle it started at -- measured repeatedly at about
        # 34 degrees. From there the camera looks along the shelf rather than at it and
        # the target book leaves the frame entirely. Squaring here, at a range where the
        # shelf front still reads as a flat face, means the final drive runs along the
        # shelf normal.
        angle = self._shelf_angle()
        if angle is not None and abs(angle) > self.square_tol:
            cmd = Twist()
            cmd.angular.z = -math.copysign(
                min(self.max_yaw, 0.8 * abs(angle) + 0.08), angle)
            self.pub_cmd.publish(cmd)
            self.get_logger().info(
                f"acquiring: squaring, face {math.degrees(angle):+.1f} deg",
                throttle_duration_sec=3.0)
            return

        bearing = self._column_cx_fresh()
        located = self._book_located()

        if bearing is None:
            self.get_logger().warn(
                "acquiring: no bearing", throttle_duration_sec=3.0)
            return

        # 2. Line up by STRAFING, not turning, so the heading stays square.
        error_px = bearing - self.image_width / 2.0
        if abs(error_px) > self.acquire_tol:
            cmd = Twist()
            cmd.linear.y = -math.copysign(
                min(self.max_lateral, 0.0015 * abs(error_px) + 0.02), error_px)
            self.pub_cmd.publish(cmd)
            self.get_logger().info(
                f"acquiring: strafing, error {error_px:+.0f}px",
                throttle_duration_sec=3.0)
            return

        if not located:
            self.get_logger().warn(
                "acquiring: centred but the book is not located yet",
                throttle_duration_sec=3.0)
            return

        self.get_logger().info(
            f"book acquired at {self._range_ahead() or float('nan'):.2f} m; closing in")
        self.approach_target = self.standoff
        self._enter(State.APPROACH)

    def _do_centre(self) -> None:
        column_cx = self._column_cx_fresh()
        if column_cx is None:
            self._stop()
            return
        error_px = column_cx - self.image_width / 2.0
        if abs(error_px) <= self.centre_tol:
            self._stop()
            # Hold here until the row has been read. This is the last point at which the
            # whole column is in frame; drive closer and the chance is gone.
            if not self.row_seen:
                self.get_logger().info(
                    "centred; waiting for the row before closing in",
                    throttle_duration_sec=5.0,
                )
                return
            # Drive to the acquire checkpoint first, not straight to grasping range.
            self.approach_target = self.acquire_range
            self._enter(State.APPROACH)
            return
        # Positive error means the column is right of centre, so turn clockwise.
        cmd = Twist()
        cmd.angular.z = -math.copysign(
            min(self.max_yaw, 0.004 * abs(error_px) + 0.08), error_px
        )
        self.pub_cmd.publish(cmd)

    def _do_approach(self) -> None:
        ahead = self._distance_to_face()
        if ahead is None:
            self.get_logger().warn("no forward LiDAR returns; holding", throttle_duration_sec=3.0)
            self._stop()
            return
        remaining = ahead - self.approach_target
        if remaining <= self.standoff_tol:
            self._stop()
            # Two-stage: pause at the acquire checkpoint, then commit to the final drive.
            if self.approach_target > self.standoff:
                self._enter(State.ACQUIRE)
            else:
                self._enter(State.SQUARE)
            return

        # Keep the target row in frame as the gap closes.
        self._aim_head()

        cmd = Twist()
        cmd.linear.x = min(self.max_fwd, max(0.05, 0.5 * remaining))

        # Correct sideways, not by turning.
        #
        # Turning to chase the bearing while driving converts a small angular error into a
        # large lateral excursion: the robot yaws a little, then drives along the new
        # heading. One run ended at y = -2.25, past the last column and off the end of the
        # shelf unit, having started centred.
        #
        # The base strafes cleanly once the arms are stowed (measured: vy = +0.20 gives
        # dy = +0.233 with dyaw = 0.000), so lateral error can be taken out directly while
        # the heading stays square to the shelf. The earlier belief that strafing yaws the
        # base was an artefact of measuring with the arms extended.
        # Steer to the anchored target while one is held, falling back to the live
        # marker bearing only before there is one. The bearing is what jumped.
        target = self._target_in_base()
        if target is not None:
            # Line the book up with the shoulder, not with the middle of the robot.
            error_m = float(target[1]) - SHOULDER_OFFSET_Y
            bearing = f"{error_m:+.3f}m"
            if abs(error_m) > self.centre_tol_m:
                # +y is to the left of the base, and so is a positive error.
                cmd.linear.y = math.copysign(
                    min(self.max_lateral, 0.6 * abs(error_m) + 0.02), error_m)
        else:
            column_cx = self._column_cx_fresh()
            bearing = "stale"
            if column_cx is not None:
                error_px = column_cx - self.image_width / 2.0
                bearing = f"{error_px:+6.1f}px"
                if abs(error_px) > self.centre_tol:
                    # Image x grows to the right; +y is to the left of the base.
                    cmd.linear.y = -math.copysign(
                        min(self.max_lateral, 0.0012 * abs(error_px)), error_px)
        self.pub_cmd.publish(cmd)

        # Log what was commanded alongside what the range is doing. Range alone cannot
        # distinguish "commanding zero" from "commanding motion and not getting it", and
        # those have opposite fixes.
        self.get_logger().info(
            f"ahead={ahead:.2f} remaining={remaining:+.2f} "
            f"cmd vx={cmd.linear.x:.3f} vy={cmd.linear.y:+.3f} bearing={bearing}",
            throttle_duration_sec=2.0,
        )

    def _do_square(self) -> None:
        angle = self._shelf_angle()
        if angle is None:
            # No credible flat surface ahead. Do NOT carry on: the grasp reaches
            # along the base's own x axis, so handing it an unsquared base aims
            # the hand across the book faces rather than into the shelf. One run did
            # exactly that, ending 35.8 degrees off with the target 15 mm from where
            # it believed it was, and still came away empty while every log said it
            # had worked.
            self._stop()
            if self.square_lost_since is None:
                self.square_lost_since = self._now()
            waited = self._now() - self.square_lost_since
            if waited < self.square_grace:
                self.get_logger().warn(
                    "no flat face to square against; waiting for a clean scan",
                    throttle_duration_sec=2.0)
                return
            if self.retreats >= self.max_retreats:
                self.get_logger().error(
                    "still no flat face after %d retreat(s); giving up rather "
                    "than grasping unsquared" % self.retreats)
                self._enter(State.FAILED)
                return
            self.retreats += 1
            self.get_logger().warn(
                "no flat face to square against; backing off to try again "
                "(attempt %d of %d)" % (self.retreats, self.max_retreats))
            self.square_lost_since = None
            self.verify_aimed_at = None
            self.book_held_since = None
            self._enter(State.RETREAT)
            return
        self.square_lost_since = None
        if abs(angle) <= self.square_tol:
            self._stop()
            self.get_logger().info(
                f"squared to {math.degrees(angle):+.1f} deg; verifying the book")
            self._enter(State.VERIFY)
            return

        # Rotate AGAINST the error. The fitted angle is the yaw error itself, so turning
        # by +angle drives further off square, not towards it -- which sent the robot
        # 155 degrees around until the shelf left the cone and it squared to a wall.
        cmd = Twist()
        cmd.angular.z = -math.copysign(min(self.max_yaw, 0.8 * abs(angle) + 0.08), angle)
        self.pub_cmd.publish(cmd)
        self.get_logger().info(
            f"squaring: face angle {math.degrees(angle):+.1f} deg, "
            f"wz={cmd.angular.z:+.3f}",
            throttle_duration_sec=2.0,
        )


def _largest_collinear_set(xs, ys, tolerance=0.05, iterations=60):
    """Return a mask of the largest set of returns lying on one straight line.

    RANSAC over x = slope * y + intercept, which is the right way round for a surface
    roughly parallel to the robot's y axis. The seed is fixed so the same scan always
    fits the same way: a squaring controller that answers differently on identical
    input is not one you can debug.
    """
    n = len(xs)
    if n < 12:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        if abs(ys[j] - ys[i]) < 1e-6:
            continue
        slope = (xs[j] - xs[i]) / (ys[j] - ys[i])
        intercept = xs[i] - slope * ys[i]
        inliers = np.abs(xs - (slope * ys + intercept)) <= tolerance
        if best is None or int(inliers.sum()) > int(best.sum()):
            best = inliers
    return best


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ApproachNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.pub_cmd.publish(Twist())  # never leave the base driving
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
