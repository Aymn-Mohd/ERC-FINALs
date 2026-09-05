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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float32, Int32, String
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

BASE_FRAME = "base_footprint"
ODOM_FRAME = "odom"
# The approach ticks at 10 Hz -- see create_timer in __init__. Named so the
# diagnostics can turn a commanded rate into an amount of turn asked for.
TICK_PERIOD = 0.1

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

# How far the book faces sit BEHIND the shelf's own front edge.
#
# Measured from the supplied mesh rather than assumed: erc_base_shelf.STL spans 0.35 m
# in the depth direction and the world places it so its front edge is at x=2.755, while
# the books stand with their faces at x=2.820. So a laser range to the shelf face
# understates the distance to the book by this much, and saying so once here is better
# than each caller guessing.
SHELF_FRONT_TO_BOOK_FACE = 0.065

TOPIC_TARGET_COLUMN_X = "/avaa/perception/target_column_x"
TOPIC_SHELF_YAW = "/avaa/perception/shelf_yaw"
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


# The driving posture, chosen to keep the arm INSIDE the base.
#
# The previous one was collision free and that is all it was. Measured from TF with the
# robot standing in it, arm_left_4_link sat at x=+0.449 -- 179 mm in front of a base that
# is 0.54 m across -- and the right gripper at x=-0.516, 246 mm behind it. An elbow
# leading the robot by 180 mm catches every shelf edge it drives past and a hand trailing
# by 250 mm catches the table on the way out, and being collision free does not help with
# either: that check is against the robot itself, and the shelf is not part of the robot.
#
# Both arms are now 0 mm outside the footprint, found by tools/tuck_search.py, which
# hill-climbs from the old posture and accepts a step only if it is both more compact and
# still collision free. Searching for compactness alone does not work: the torso stands in
# the middle of the base, so "inside the footprint" and "inside the robot" are nearly the
# same volume, and all 400 of the zero-overhang postures a free search produced were
# rejected by MoveIt.
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
TUCK_POSE = [0.36, -1.83, 0.47, -2.35, 0.0, -1.2, 0.0]
# The tuck is an eight-joint posture, torso included, and only the eight together
# are collision free. Commanding the arm alone and leaving the torso down folds
# arm_left_5, arm_left_6 and the gripper into base_link -- MoveIt reports exactly
# those three contacts -- and the planner then refuses every request from that
# start state, in 0.4 s, with no indication that the torso is the reason.
TUCK_TORSO = 0.10

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
RIGHT_TUCK = [-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0]


def turn_for(error_rad: float, period: float, share: float = 0.5,
             fastest: float = 0.45, slowest_period: float = 0.25) -> float:
    """Pick a turn rate from how often the bearing that steers it actually arrives.

    A turn controller may not move further between two measurements than the error it
    is correcting, and this one did. The bearing comes from perception, and perception
    processes a frame about every two SIMULATED seconds -- watched in the approach's own
    log, where the reported bearing changed at 468255, 468257 and 468259 and then held
    the same value for the next sample. Turning at the 0.45 rad/s the gain saturated to
    sweeps 0.9 rad in that gap. The head camera sees about 1.0 rad in total.

    So the base swung most of its field of view between one look and the next, the
    marker left the frame, and the state stood with no bearing at all until its
    out-of-view timeout fired twelve seconds later. Three cycles of that in one run, and
    the pixel error logged as unchanged each time because the log and the sensor were
    both slower than the motion.

    A fixed gain cannot fix this, because the period is not a constant of the system:
    the real-time factor in this project has been measured between 0.013 and 0.60, a
    span of forty-five, and the frame rate goes with it. The rate has to come from the
    period. Take ``share`` of the error per measurement, so the turn converges
    geometrically rather than overshooting, and never move faster than ``fastest``.
    """
    period = max(period, slowest_period)
    return float(max(-fastest, min(fastest, share * error_rad / period)))


def sighting_gate(allowance: float, speed: float, gap: float,
                  longest: float = 1.5) -> float:
    """How far the book may honestly have moved in base_link since the last sighting.

    Module level so it can be tested without a simulator. It replaces a gate held in
    ODOM, which is the one frame here that must not be used for this: measured during a
    run held to 17 mm of true error, odom accumulated 813 mm of travel that never
    happened, because holding this base still means driving the wheels against a slide
    and odom faithfully integrates every one of those turns. A target anchored in a
    frame that drifts is a target that walks away, and it did -- one run rejected 24
    consecutive correct sightings as "2.42 m from the anchored target" and then reversed
    2.98 m away from the shelf hunting for a book that was in front of it.

    The job the gate is actually for is narrow: refuse a bearing that has jumped to a
    DIFFERENT book of the same colour. Columns are about 0.95 m apart and rows 0.33 m,
    and those are distances in the robot's own frame right now, not in any accumulated
    one. So compare each sighting against the last accepted one, in base_link, and allow
    for how far the base could have carried the book in between -- which is its own
    driving speed, plus the coast measured at 8 mm per simulated second, plus what
    perception's error looks like at 15-35 mm.

    ``gap`` is capped at ``longest`` because the budget must not grow without limit while
    the book is out of view: a sighting that arrives after a long silence has nothing
    recent to be continuous with, and is better handled by making it agree with several
    others than by widening the gate until it admits the whole shelf.
    """
    return allowance + speed * min(max(0.0, gap), longest)


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
        # Was 0.75, and 0.75 is too far. Measured with tools/reach.py, which finds the
        # arm's greatest shoulder-to-tip distance by sampling and hill climbing -- 1.088
        # m -- and then works out what fraction of it each row and standoff needs:
        #
        #     standoff   row 1   row 2   row 3   row 4
        #     0.60       62%     52%     57%     58%
        #     0.65       66%     57%     61%     62%
        #     0.75       74%     66%     70%     70%
        #     0.85       82%     75%     78%     79%
        #
        # Those are all inside the envelope, so 0.75 looks safe on paper. What the table
        # cannot show is that the base does not stop where it is told. Watched in a run,
        # the grasp measured its target at 1126 mm from the shoulder against a maximum
        # of 1088 -- about 0.27 m beyond where a 0.75 m standoff would have put it --
        # and every one of twenty-four candidate postures failed at the first waypoint
        # because the point simply cannot be occupied.
        #
        # 0.65 keeps the same margin against the shelf that mattered before (the bumper
        # is 0.36 m out, so this leaves about 0.21 m of clearance ahead of it) while
        # putting a comparable overshoot back inside the arm's reach.
        #
        # The better fix is for the grasp to close the gap itself when it finds the book
        # out of reach -- it holds the base already and knows the shortfall to the
        # millimetre. That is worth doing and is not done here.
        self.declare_parameter("standoff_m", 0.65)
        self.declare_parameter("centre_tolerance_px", 12.0)
        self.declare_parameter("standoff_tolerance_m", 0.05)
        # 0.05 rad is 2.9 degrees, and the shelf angle reads to about that, so the
        # squaring hunted along its own tolerance boundary and never declared itself
        # done. Widened to 5 degrees: the reach is re-aimed from perception at the shelf
        # anyway, and the grasp closes its last centimetres on a servo, so arriving two
        # degrees less square costs less than not arriving.
        self.declare_parameter("square_tolerance_rad", 0.09)
        self.declare_parameter("max_yaw_rate", 0.45)
        # How hard to push back against the base's own rotation. At 0 the
        # turn controllers are pure proportional and they ring; the base
        # carries its rate because nothing damps it.
        self.declare_parameter("turn_damping", 0.55)
        self.declare_parameter("max_forward", 0.22)
        self.declare_parameter("max_lateral", 0.10)
        # Refuse to drive closer than this whatever the standoff says, so a bad reading
        # cannot push the base into the shelf. A collision costs 0.5 points each time.
        # Also measured from base_footprint: the bumper is 0.36 m out, so 0.55 m leaves
        # roughly 0.19 m of margin.
        self.declare_parameter("min_safe_range_m", 0.55)
        self.declare_parameter("state_timeout_sec", 45.0)
        # The acquire checkpoint gets its own, longer budget. It is the one
        # state that deliberately holds still and waits -- for the row to be
        # read, and for a metric fix on the book -- and it also drives an
        # arc of up to half a metre at creep speed. 45 s cut it off with
        # 190 mm to go on a run that was converging steadily.
        self.declare_parameter("acquire_timeout_sec", 120.0)
        self.declare_parameter("search_rate", 0.35)
        # How long the target column may be out of view before the robot gives up
        # centring on it and sweeps for it again.
        #
        # Four seconds was far too eager. The overhead marker is small and read from a
        # moving base with a moving head, so it drops out of a frame or two constantly;
        # measured, the approach re-searched six times in three minutes and never got
        # past centring, losing the marker four seconds into each attempt. The bearing
        # it steers by is still required to be fresh -- this only governs when to give
        # up on the column altogether, which is a much rarer event than a missed read.
        # How long to stand still with no bearing before going back to SEARCH.
        #
        # Was 12 s, and standing still for twelve seconds buys nothing: the base cannot
        # find a marker it is not moving to look for, and SEARCH finds one again in two
        # to six seconds every time it is asked. Measured over one run, this state spent
        # 12 s waiting on three separate occasions -- 36 seconds of a 120 second budget
        # doing nothing at all, and it timed out having never reached the shelf.
        #
        # Four seconds is still long enough to ride out a marker that flickers at the
        # edge of the frame, and short enough that a lost marker costs one search rather
        # than a third of the state's clock.
        self.declare_parameter("lost_grace_sec", 4.0)
        # How far the book may sit off the reaching arm's centre line and
        # still be worth handing to the grasp. The pre-grasp is chosen from
        # twelve candidate postures and they run out well before the arm
        # does; 0.25 m is comfortably inside where they still solve.
        self.declare_parameter("lateral_tolerance_m", 0.25)
        # How long to stand squared with no acceptable fix before giving
        # up on the anchor and sweeping for the marker again.
        self.declare_parameter("nofix_grace_sec", 20.0)
        # The closest the robot will drive without a 3D fix on the book.
        # Far enough out that all four rows are still in the camera's
        # reach, so a target it cannot see is a target it can back away
        # from rather than one it is stuck in front of.
        self.declare_parameter("blind_floor_m", 1.60)
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
        # How close to the standing pose counts as arrived, in metres.
        self.declare_parameter("acquire_pose_tolerance_m", 0.06)
        self.declare_parameter("acquire_pose_gain", 0.5)
        # A floor under the speed, because a base with 2 Nm of stiction in
        # its drive does not move at all for a few millimetres a second.
        self.declare_parameter("acquire_creep_speed", 0.07)
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
        self.turn_damping = float(self.get_parameter("turn_damping").value)
        self.max_fwd = float(self.get_parameter("max_forward").value)
        self.max_lateral = float(self.get_parameter("max_lateral").value)
        self.min_safe = float(self.get_parameter("min_safe_range_m").value)
        self.timeout = float(self.get_parameter("state_timeout_sec").value)
        self.acquire_timeout = float(
            self.get_parameter("acquire_timeout_sec").value)
        self.search_rate = float(self.get_parameter("search_rate").value)
        self.lost_grace = float(self.get_parameter("lost_grace_sec").value)
        self.lateral_tol = float(
            self.get_parameter("lateral_tolerance_m").value)
        self.nofix_grace = float(self.get_parameter("nofix_grace_sec").value)
        self.blind_floor = float(self.get_parameter("blind_floor_m").value)
        self.nofix_since: Optional[float] = None
        self.lost_since: Optional[float] = None
        self.unsquared_since: Optional[float] = None
        self.search_timeout = float(self.get_parameter("search_timeout_sec").value)
        self.image_width = int(self.get_parameter("image_width_px").value)
        # The head camera's horizontal focal length in pixels, for turning a bearing
        # in pixels into one in radians. Read from CameraInfo at startup; this is the
        # measured value from the depth stream and is only the fallback.
        self.focal_px = 337.2
        # How often the steering bearing actually arrives, in simulated seconds,
        # measured rather than assumed. See turn_for.
        self.bearing_period = 1.0
        self.bearing_last_at = None
        self.bearing_last_value = None
        self.tuck_time = float(self.get_parameter("tuck_time_sec").value)

        self.column_cx: Optional[float] = None
        self.column_cx_at: Optional[float] = None
        self.column_cx_new_at: Optional[float] = None
        # How long a bearing may go without changing before the controller stops
        # believing it. Bearings arrive at about 5 Hz when perception is publishing at
        # all, so two seconds is many missed measurements, not a hiccup.
        self.bearing_stale = 2.0
        # Per visit to CENTRE: how much turn has been commanded, how much the base
        # has actually turned, and how many ticks asked for nothing at all. A
        # controller that is right but only running a fifth of the time looks exactly
        # like a controller with the wrong gain, and these tell them apart.
        self.centre_asked = 0.0
        self.centre_ticks = 0
        self.centre_idle = 0
        self.centre_yaw0 = None
        self.scan: Optional[LaserScan] = None
        self.yaw_rate = 0.0
        self.odom_yaw = None
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
        self.acquire_pose_tol = float(
            self.get_parameter("acquire_pose_tolerance_m").value)
        self.pose_gain = float(self.get_parameter("acquire_pose_gain").value)
        self.creep_speed = float(
            self.get_parameter("acquire_creep_speed").value)
        # Set once the book has actually been located in 3D, which is the signal that it
        # is genuinely visible rather than merely expected to be.
        self.book_point_at: Optional[float] = None
        self.book_seen_at: Optional[float] = None
        self.book_live = None
        self.shelf_yaw: Optional[float] = None
        self.shelf_yaw_at: Optional[float] = None
        self.book_x: Optional[float] = None
        self.approach_target = self.acquire_range
        # Retry budget for losing the book on the final drive.
        self.retreats = 0
        # The book position, in base_link, from the last sighting that was believed.
        # NOT in odom: see sighting_gate for what odom does to a stored target here.
        self.target_base = None
        self.target_base_at = None
        # How far two consecutive sightings of the same book may sit apart in
        # base_link, over and above the base's own travel between them. Perception
        # measures the book to 15-35 mm, so this is a little over twice its own error.
        self.sighting_allowance = 0.08
        # Past this the target in base_link is no longer about where the robot is
        # now, and a new fix is collected from scratch rather than continued.
        self.sighting_stale = 1.5
        # The speed the gap is multiplied by: the fastest the base drives, 0.22 m/s,
        # plus the coast at 0.008, plus margin. Being generous here costs an outlier
        # admitted; being tight costs the correct sighting refused, which is what the
        # odom version did 24 times in a row.
        self.sighting_speed = 0.35
        # Sightings still have to agree with each other before one is trusted at all,
        # and that spread is judged against this. Column spacing is 0.95 m.
        self.anchor_gate = 0.45
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
        self.create_subscription(
            Float32, TOPIC_SHELF_YAW, self._on_shelf_yaw, 10)
        self.create_subscription(LaserScan, TOPIC_SCAN, self._on_scan, SENSOR_QOS)
        # Odometry is used for ONE thing: the yaw rate, to damp the turns.
        # It is not trusted for position -- the base slides across its wheels
        # without turning them, and during one run held to 17 mm of true error
        # odom had accumulated 813 mm of travel that never happened. A turn is
        # different: turning does rotate the wheels, so the rate is real.
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
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
        value = float(msg.data)
        # Two timestamps, not one, and the difference between them is the whole
        # question. column_cx_at is when a message last ARRIVED, which is what the
        # freshness test wants. column_cx_new_at is when the NUMBER last changed, which
        # is what a controller wants: perception can republish a value it computed
        # several looks ago, and steering on that is steering on the past.
        if self.column_cx is None or abs(value - self.column_cx) > 0.5:
            self.column_cx_new_at = self._now()
        self.column_cx = value
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

    def _on_odom(self, msg: Odometry) -> None:
        self.yaw_rate = float(msg.twist.twist.angular.z)
        q = msg.pose.pose.orientation
        self.odom_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _turn(self, error: float, gain: float, floor: float = 0.08) -> float:
        """Command a damped turn: proportional to the error, against the measured rate.

        Every turn controller here oscillated, and none of them was mistuned. This base
        has no friction across the roller axis, so it does not stop when the command
        stops -- it keeps whatever rate it was given. A pure proportional law on a plant
        with no damping is an oscillator, and that is what the logs show: centring turned
        for ninety seconds without ever landing inside a twelve pixel window, and the
        acquire strafe walked the error from 132 px to 315 px in one direction.

        Subtracting the measured rate supplies the damping the floor does not. The rate
        comes from odometry, which is blind to sliding but not to turning: turning is the
        one thing that actually rotates these wheels.
        """
        wanted = gain * error
        if abs(wanted) > 1e-6:
            wanted += math.copysign(floor, wanted)
        command = wanted - self.turn_damping * self.yaw_rate
        return float(max(-self.max_yaw, min(self.max_yaw, command)))

    def _on_shelf_yaw(self, msg: Float32) -> None:
        self.shelf_yaw = float(msg.data)
        self.shelf_yaw_at = self._now()

    def _heading_error(self, max_age: float = 2.0) -> Optional[float]:
        """How far the base is turned off square to the shelf, best source first.

        The depth camera is the only sensor here that can actually see the shelf. The
        laser is 209 mm off the floor, where the unit is an open compartment: measured
        at 0.74 m from the front, 163 returns in the forward cone, none inside two
        metres, 110 of them on the far wall. Every heading the approach has taken from
        it has been a heading to something else.

        Measured against Gazebo, the depth fit gives -34.5, -34.5, -35.3, -34.7, -34.2,
        -34.5 degrees against a true -35.9 -- better than a degree and a half, steady,
        from about 2300 points. That is the number this controller has been missing; one
        run reached the standoff 38.9 degrees off square, at which angle the camera looks
        along the shelf rather than at it and the book leaves the frame entirely.

        The laser fit stays as the fallback. It is not useless -- it will find whatever
        plane genuinely is in front of the robot, and early in a run, before the head has
        found a row to measure a band around, that is better than nothing.
        """
        if (self.shelf_yaw is not None and self.shelf_yaw_at is not None
                and (self._now() - self.shelf_yaw_at) <= max_age):
            return self.shelf_yaw
        return self._shelf_angle()

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

    def _shelf_face(self, half_angle: float = 0.45):
        """Find the shelf's front face: (perpendicular distance, yaw error), or None.

        The one measurement the whole approach should be built on, and the reason it is
        worth its own method rather than being two loosely related ones.

        The shelf front is flat and 5.25 m wide, which makes it the most reliable thing
        any sensor on this robot can see -- far more reliable than a small overhead
        marker read from a moving base, which is what the approach steered by and which
        dropped out constantly. Four of the Amazon Picking Challenge teams used a laser
        on the base for exactly this: aligning to the shelf, not finding objects.

        Why the NEAREST large surface rather than the largest. _shelf_angle takes the
        biggest consensus set, and through an open shelf the biggest is not always the
        front: the beam passes through the unstocked bottom shelf and the row openings to
        the back panel 0.35 m behind, and when enough of it does, the back panel wins the
        vote. Then the reported range is a third of a metre too far, the drive overshoots
        by that much, and on the top row that is the difference between seeing the book
        and being stuck too close to see it. Measured: the squaring read a face at -20.7
        and then -38.9 degrees at the standoff, and could not fit anything at all a
        moment later.

        Taking the nearest surface that has enough points behind it separates all three
        cases the scan actually contains:

          - the front face: many points, nearest          -> chosen
          - the back panel: many points, 0.35 m further   -> rejected, further away
          - one book pulled proud of the shelf: nearest,
            but only a handful of points                  -> rejected, too few

        Distance and angle come from the same fit, so they cannot disagree about which
        surface they describe -- which two separate helpers reading the same scan could,
        and did.
        """
        points = self._forward_points(half_angle=half_angle)
        if len(points) < 12:
            return None
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        floor = max(12, int(self.face_consensus * len(xs)))
        best = None
        remaining = np.ones(len(xs), dtype=bool)

        # Peel off surfaces one at a time. Three passes is enough for this scene: the
        # front face, the back panel, and whatever else is in the cone.
        for _ in range(3):
            if int(remaining.sum()) < floor:
                break
            sub_x, sub_y = xs[remaining], ys[remaining]
            inliers = _largest_collinear_set(
                sub_x, sub_y, tolerance=self.face_tolerance)
            if inliers is None or int(inliers.sum()) < floor:
                break
            face_x, face_y = sub_x[inliers], sub_y[inliers]
            slope, intercept = np.polyfit(face_y, face_x, 1)
            residual = float(np.std(face_x - (slope * face_y + intercept)))
            if residual <= 0.05:
                # Perpendicular distance from the base to the fitted line. The line is
                # x = slope*y + intercept, so the foot of the perpendicular from the
                # origin is at intercept / sqrt(1 + slope^2).
                distance = abs(float(intercept)) / math.sqrt(1.0 + slope * slope)
                if best is None or distance < best[0]:
                    best = (distance, float(math.atan(slope)), int(inliers.sum()))
            # Drop this surface's points and look for the next one.
            index = np.where(remaining)[0]
            remaining[index[inliers]] = False

        if best is None:
            return None
        distance, angle, count = best
        self.get_logger().debug(
            "shelf face at %.2f m, %+.1f deg, from %d returns"
            % (distance, math.degrees(angle), count))
        return distance, angle

    def _shelf_angle(self) -> Optional[float]:
        """Yaw error against the shelf face: 0 when square on, positive when yawed CCW.

        Now a thin wrapper over _shelf_face, so the angle and the distance can never
        describe two different surfaces. They could before, and did: one helper took the
        largest consensus set and the other took the median of a narrow cone, and through
        an open shelf those are the back panel and the front face respectively.
        """
        face = self._shelf_face()
        return None if face is None else face[1]

    # ------------------------------------------------------------------ control

    def _enter(self, state: State) -> None:
        if state is State.RETREAT:
            # Start the anchor again: if it had been right we would not be retreating.
            self.target_base = None
            self.target_base_at = None
            self.anchor_candidates.clear()
            self.anchor_disagree.clear()
        if state is not self.state:
            self.get_logger().info(f"{self.state.value} -> {state.value}")
            self.state = state
            self.state_since = self._now()
            if state is State.CENTRE:
                # Only on a real transition: these count one visit to the state, and
                # resetting them on a re-entry that is not a change would hide exactly
                # the pattern they exist to show.
                self.centre_asked = 0.0
                self.centre_ticks = 0
                self.centre_idle = 0
                self.centre_yaw0 = self.odom_yaw

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

        budget = self.timeout
        if self.state is State.SEARCH:
            budget = self.search_timeout
        elif self.state is State.ACQUIRE:
            budget = self.acquire_timeout
        if self._elapsed() > budget:
            self.get_logger().error(f"timed out in {self.state.value}")
            self._stop()
            self._enter(State.FAILED)
            return

        # Safety floor applies in every moving state.
        nearest = self._min_range_ahead()
        if nearest is not None and nearest < self.min_safe and self.state is State.APPROACH:
            # Back away, do not hand this to the squaring.
            #
            # It used to enter SQUARE, and SQUARE cannot work from here: fitting a line
            # to the shelf face needs enough of the face in view, and half a metre from
            # it the scan is dominated by whatever single board or upright triggered the
            # stop. Measured on the run that found this, the safety floor tripped at
            # 0.54 m with the book's own fix putting it 1.11 m away, SQUARE reported "no
            # flat face to square against" four times, and the approach gave up --
            # ending a run that was otherwise going well, half a metre from a shelf it
            # had tracked correctly the whole way in.
            #
            # RETREAT already exists for exactly this and hands back to ACQUIRE, which
            # can re-measure and come in again on a range it trusts.
            self.get_logger().warn(
                "safety stop: something at %.2f m, closer than the %.2f m floor; "
                "backing off rather than trying to square from here"
                % (nearest, self.min_safe))
            self._stop()
            self._enter(State.RETREAT)
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
        """Record a sighting, keeping "seen" and "believed" apart.

        These were one thing and they are not the same thing, and merging them starved
        every state that asks whether the book is visible. book_point_at was set only
        when the anchor ACCEPTED a sighting, so _book_located -- which reads as "the
        book is in view" and is used that way in four places -- actually meant "the
        anchor agrees with this". Before the anchor has formed it needs eight agreeing
        candidates, and while it is forming it accepts nothing, so perception could be
        publishing the book on every frame while the approach reported no fix and
        eventually gave up "rather than grasping blind". It was not blind.

        So: seen on every sighting, believed only when the anchor accepts. The states
        that need a trusted target position still ask for one through _target_in_base;
        the states that only need to know whether perception can see the book now get an
        honest answer.
        """
        point = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.book_seen_at = self._now()
        self.book_x = float(point[0])
        self.book_live = point
        if not self._accept_sighting(point, msg.header.frame_id):
            return
        self.book_point_at = self._now()

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
        """Decide whether a sighting is the book we set out for.

        The column bearing comes from a marker mounted above the shelf, and that marker
        leaves the camera as the robot closes in. One run had the bearing jump 310 px in
        a single step at 1 m out, which strafed the base into the divider between two
        columns and left it facing neither: ground truth put the two nearest red books
        at +33.5 and -39.6 degrees off the nose, both at the edge of the frame, and
        perception reported no red book in view for the rest of the run. Refusing that
        jump is the whole purpose here.

        It used to be done by holding the target in odom and rejecting sightings that
        disagreed with it, and that was the wrong frame for the job -- see sighting_gate
        for the measurement, and for the run where it rejected two dozen correct
        sightings and drove away from the shelf.

        Continuity in base_link does the same job without odom. Consecutive sightings of
        one book are a few centimetres apart plus whatever the base drove; a jump to the
        next column is 0.95 m, which no gap between sightings can account for.
        """
        if self.state not in (State.ACQUIRE, State.APPROACH,
                              State.SQUARE, State.VERIFY):
            return True
        if frame_id and frame_id not in (BASE_FRAME, "base_link"):
            tf = self._lookup(BASE_FRAME, frame_id)
            if tf is None:
                return True   # nothing to compare in; a sighting beats none
            point = self._apply_transform(tf, point)

        now = self._now()
        stale = (self.target_base_at is None
                 or (now - self.target_base_at) > self.sighting_stale)
        if self.target_base is None or stale:
            # Do not trust a single sighting. With no marker in view perception picks
            # one of several same-coloured books, and whichever it picked first would
            # become the target for the whole run.
            self.anchor_candidates.append(point)
            settled = self._agreeing(self.anchor_candidates)
            if settled is None:
                return False
            self.target_base = settled
            self.target_base_at = now
            self.anchor_candidates.clear()
            self.get_logger().info(
                "target fixed at [%.2f, %.2f, %.2f] in base_link from %d agreeing "
                "sightings" % (settled[0], settled[1], settled[2],
                               self.anchor_candidates.maxlen))
            return True

        gap = now - self.target_base_at
        step = float(np.linalg.norm(point[:2] - self.target_base[:2]))
        budget = sighting_gate(self.sighting_allowance, self.sighting_speed, gap)
        if step > budget:
            self.anchor_rejects += 1
            self.anchor_disagree.append(point)
            # An outlier gate with no way out turns one bad fix into a permanent one. A
            # run anchored to the wrong book rejected the right one 217 times in a row,
            # all at the same 1.24 m, until the approach timed out. Sightings that
            # disagree with the last accepted one but agree with each other are the
            # better answer.
            replacement = self._agreeing(self.anchor_disagree)
            if replacement is not None:
                self.get_logger().warn(
                    "re-fixing the target: %d sightings agree with each other %.2f m "
                    "from the last accepted one, so that one was wrong"
                    % (len(self.anchor_disagree), step))
                self.target_base = replacement
                self.target_base_at = now
                self.anchor_disagree.clear()
                return True
            self.get_logger().warn(
                "ignoring a sighting %.2f m from the last accepted, %.0f mm allowed "
                "for a %.1f s gap (%d so far)"
                % (step, budget * 1000, gap, self.anchor_rejects),
                throttle_duration_sec=3.0)
            return False

        # Believed. Take it as it stands rather than smoothing towards it: base_link
        # moves with the robot, so an average of sightings taken while driving is an
        # average over places the robot no longer is.
        self.anchor_disagree.clear()
        self.target_base = point
        self.target_base_at = now
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
        """Give the believed target as (x, y, z) in base_footprint, or None.

        It has to be recent to mean anything. base_link travels with the robot, so a
        target held in it goes stale at the speed the base drives -- which is the
        opposite of the old odom version, where the number stayed put and the frame
        underneath it was what moved.
        """
        if self.target_base is None or self.target_base_at is None:
            return None
        if (self._now() - self.target_base_at) > self.sighting_stale:
            return None
        return self.target_base

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
        # The book first. The shelf face is NOT available here.
        #
        # This briefly preferred a laser fit to the shelf front, on the reasoning that a
        # 5.25 m flat surface is the most reliable thing any sensor here can see. It is
        # not visible to this one. The laser sits 209 mm off the floor and at that height
        # the shelf is an open compartment -- measured at 0.74 m from the front, the
        # forward cone returned 163 beams and not one of them inside two metres, with 110
        # of them landing on the far wall 4.5 m away, straight through the unit. See
        # tools/ranges.py.
        #
        # So the range comes from the book when perception can see it, and from the
        # nearest laser return otherwise, which at least measures something solid even if
        # it is an upright rather than the face.
        if self._book_located() and self.book_x is not None:
            return self.book_x
        # The NEAREST return in a wide cone, not the median of a narrow one.
        #
        # _range_ahead takes the median over 0.12 rad, and a narrow cone aimed at a
        # shelf OPENING reads straight through to the back panel: it overstates the
        # distance to the face by about the shelf's depth, which the mesh puts at 0.35 m.
        # The drive to the acquire checkpoint then overshoots by that much, and on the
        # top row that is the difference between seeing the book and not -- at 0.51 m the
        # camera cannot tilt far enough up to find it, and with no sighting there is no
        # fix, and with no fix nothing can drive the robot back out.
        #
        # The minimum over a wider cone measures the shelf's uprights and edges, which
        # are at the face. A spurious short return now stops the approach early rather
        # than late, and early is recoverable.
        return self._min_range_ahead()

    def _book_located(self, max_age: float = 1.5) -> bool:
        """Whether perception can currently see the book in 3D.

        Deliberately the SEEN timestamp, not the accepted one. See _on_book_point.
        """
        if self.book_seen_at is None:
            return False
        return (self._now() - self.book_seen_at) <= max_age

    def _book_trusted(self, max_age: float = 1.5) -> bool:
        """Whether there is an anchored fix on the book, not merely a sighting."""
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
                # Being the right DISTANCE away is not being in front of it.
                #
                # This declared the approach complete on range alone, and the grasp then
                # inherited whatever sideways error was left. Measured on one run: the
                # approach reported "book held for 3.0 s at 0.56 m" and handed over a
                # book 0.639 m to the RIGHT of base_link -- most of a metre off the left
                # shoulder's centre line, at the very edge of its reach. The grasp
                # answered honestly, with none of twelve postures solving, and the
                # failure read as an arm problem.
                #
                # The lateral offset is measured, not inferred: the book's own 3D fix
                # carries it. If it is too large the robot goes back to acquiring, where
                # the pose controller exists precisely to close it.
                target = self._target_in_base()
                if target is not None:
                    sideways = float(target[1]) - SHOULDER_OFFSET_Y
                    if abs(sideways) > self.lateral_tol:
                        self.get_logger().warn(
                            "the book is %.0f mm off the arm's centre line, which is "
                            "more than the %.0f mm the grasp can take; lining up again"
                            % (sideways * 1000, self.lateral_tol * 1000))
                        self.book_held_since = None
                        self._enter(State.ACQUIRE)
                        return
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
        # Hold the heading while reversing. Commanding linear velocity alone on this
        # base is how the acquire back-off arrived side-on to the shelf at -95 degrees:
        # there is no friction across the roller axis, so whatever rotation the robot
        # has it keeps, and reversing blind for twenty seconds is long enough to lose
        # the shelf entirely. Square against it if it can be seen; otherwise damp the
        # rate, which needs nothing but odometry.
        angle = self._heading_error()
        cmd.angular.z = (self._turn(-angle, 0.8) if angle is not None
                         else self._turn(0.0, 0.0, floor=0.0))
        self.pub_cmd.publish(cmd)

    def _do_acquire(self) -> None:
        """Hold at a working distance until the book is seen and centred.

        This is a checkpoint, not a drive. Everything after it depends on the book being
        genuinely in view, so it is worth a few seconds here rather than discovering at
        grasping range that the target was lost on the way in.

        The head is aimed at the row first: the books are well below the markers, and
        without the tilt the target may not be in frame at all from here.

        Nothing is stopped on the way in. This used to open with an unconditional
        _stop(), which publishes a zero twist, and then fall through to publish a turn
        -- so every tick sent stop, turn, stop, turn, and on the ticks that returned
        early it sent only the stop. The base was being asked to hold still as often as
        it was being asked to move. Measured over forty seconds of squaring, the shelf
        angle read -2.9, -3.2, -3.1, -3.6, -2.9, -2.9, -3.1, -3.4 degrees: a controller
        commanding a correction the whole time and a robot not turning at all. Each
        branch below now sends exactly one command, and the branches that want the base
        still say so themselves.
        """
        self._aim_head()

        # 1. Square to the shelf FIRST.
        #
        # Centring turns to face the column and then drives along that bearing, so the
        # robot arrives at whatever angle it started at -- measured repeatedly at about
        # 34 degrees. From there the camera looks along the shelf rather than at it and
        # the target book leaves the frame entirely. Squaring here, at a range where the
        # shelf front still reads as a flat face, means the final drive runs along the
        # shelf normal.
        goal = self._acquire_goal()
        angle = self._heading_error()

        # Square first only while there is no pose to drive to. Once there is, the pose
        # controller owns the heading: arriving square is what it is for, and it has to
        # turn away from square on the way in order to arc across. Squaring on every tick
        # fought it -- measured over one run the two alternated, the lateral offset came
        # in at 20 mm per correction, and the state ran out of time with 190 mm still to
        # go while the shelf angle wandered between -5 and -11 degrees.
        if goal is None:
            # Back off if the reason there is no fix is that the robot is too close.
            #
            # This was a deadlock, and it swallowed a whole run. The forward LiDAR reads
            # THROUGH the open shelf to its back panel, so it overstates the distance to
            # the face by about the shelf's depth, and the drive to the acquire
            # checkpoint overshoots. Measured on the run that found this: the robot
            # finished 0.51 m from a book on the TOP row, and at that range the camera
            # cannot tilt far enough up to see it -- "no green book in view at close
            # range", every frame, for the whole state. No sighting means no fix, no fix
            # means the pose controller has nowhere to drive, and nothing was left that
            # could undo the one thing causing it.
            #
            # Reversing needs no fix on the book. The LiDAR overstates the range, so if
            # even IT says the shelf is nearer than the checkpoint, the robot is much
            # too close, and backing up can only improve the view.
            ahead = self._range_ahead()
            if ahead is not None and ahead < self.acquire_range - self.standoff_tol:
                cmd = Twist()
                cmd.linear.x = -min(self.max_fwd, 0.5 * (self.acquire_range - ahead))
                # Hold the heading while reversing, or the reverse becomes a pirouette.
                #
                # The first version commanded linear velocity alone, on the reasoning
                # that backing straight up needs no steering. It does on this base:
                # there is no friction across the roller axis, so whatever rotation the
                # robot already had it keeps, and nothing was cancelling it. Measured on
                # the run that found this, the robot backed off from 0.74 m to 1.4 m and
                # arrived facing -95 degrees -- side-on to the shelf, with the LiDAR
                # reading 3.92 m down the room and no book in view anywhere. It had
                # solved the problem it was reversing to solve and created a worse one.
                #
                # Squaring against the shelf if it can be seen; otherwise simply damping
                # the rate, which needs nothing but odometry and stops the drift.
                cmd.angular.z = (self._turn(-angle, 0.8) if angle is not None
                                 else self._turn(0.0, 0.0, floor=0.0))
                self.pub_cmd.publish(cmd)
                self.get_logger().warn(
                    "acquiring: no fix and the shelf is only %.2f m ahead, which is "
                    "closer than the %.2f m checkpoint; backing off to see the book"
                    % (ahead, self.acquire_range), throttle_duration_sec=3.0)
                return

            # No fix, and no flat face to square against either: the robot is not
            # looking at the shelf at all. Sweep for the marker again rather than stand
            # here reasoning about a shelf that is not in front of it.
            if angle is None:
                if self.unsquared_since is None:
                    self.unsquared_since = self._now()
                elif self._now() - self.unsquared_since > self.lost_grace:
                    self.get_logger().warn(
                        "no book and no shelf face for %.0f s; the robot is not facing "
                        "the shelf, so searching again" % self.lost_grace)
                    self.unsquared_since = None
                    self._stop()
                    self._enter(State.SEARCH)
                    return
            else:
                self.unsquared_since = None

            # No _stop() before the turn. This is the same fault that made the
            # squaring look dead once before: a zero twist published on the tick, then
            # a turn published on the same tick, and the base given stop, turn, stop,
            # turn. Measured again here, the shelf angle read -5.2 degrees for a
            # minute and a half while a correction was commanded throughout.
            if angle is not None and abs(angle) > self.square_tol:
                # Sized from how often the shelf angle arrives, for the same reason
                # centring is: a fixed gain on an angle that is measured a few times a
                # second sweeps past the target between two looks. Watched in a run,
                # this hunted -7.1, +8.7, -9.0, +7.8, -9.9 degrees and never landed
                # inside its 5.2 degree tolerance, while the state ran out its clock.
                cmd = Twist()
                wanted = turn_for(-angle, self.bearing_period, fastest=self.max_yaw)
                cmd.angular.z = float(max(-self.max_yaw, min(
                    self.max_yaw, wanted - self.turn_damping * self.yaw_rate)))
                self.pub_cmd.publish(cmd)
                self.get_logger().info(
                    "acquiring: squaring to look for the book, face %+.1f deg"
                    % math.degrees(angle), throttle_duration_sec=3.0)
                return

            # Squared, and still no fix. Perception can see the book -- it says so --
            # but every sighting is being refused by the anchor gate, so the approach
            # has a target it will not believe. Standing here squared and blind until
            # the state times out achieves nothing; go and look again, which clears the
            # anchor along with everything else.
            self._stop()
            if self.nofix_since is None:
                self.nofix_since = self._now()
            elif self._now() - self.nofix_since > self.nofix_grace:
                self.get_logger().warn(
                    "squared for %.0f s with no fix on the book that the anchor will "
                    "accept; searching again" % self.nofix_grace)
                self.nofix_since = None
                self.target_base = None
                self.target_base_at = None
                self.anchor_candidates.clear()
                self.anchor_disagree.clear()
                self.anchor_rejects = 0
                self._enter(State.SEARCH)
                return
            self.get_logger().warn(
                "acquiring: no metric fix on the book to drive to",
                throttle_duration_sec=3.0)
            return

        located = self._book_located()

        # No check on the image bearing here any more. It was required before this state
        # steered by pixels, and it outlived the thing it was guarding: the pose
        # controller below drives to a metric fix on the book and never reads a bearing,
        # so a missing one is not a reason to refuse to move. Requiring it stalled the
        # state for its whole timeout every time the head tilted far enough to push the
        # marker out of frame -- which is exactly what the head is tilted down for.

        # 2. Line up by driving to a POSE, not by strafing.
        #
        # Strafing here diverged. Measured over one run, with the controller commanding
        # lateral velocity the whole time, the pixel error went -132, -161, -182, -210,
        # -242, -315 and the robot finished at the end of the shelf unit looking along
        # it. That is not a gain that wants tuning. This base cannot strafe: commanding
        # pure vy yaws it by roughly the magnitude it moves sideways, which is already
        # written down as a design decision elsewhere in this project, and the yaw turns
        # the camera away faster than the translation brings the target in.
        #
        # What replaces it is the textbook pose controller for a base that can only
        # drive and turn. The goal is a pose, not a point: stand at the standoff in front
        # of the book, facing the shelf. In base_footprint the shelf normal is +x once
        # squared, so the goal is (book_x - standoff, book_y) with heading zero, and
        #
        #     rho   distance to it
        #     alpha bearing to it
        #     beta  the heading still owed on arrival, which is -alpha here
        #     v     k_rho * rho
        #     omega k_alpha * alpha + k_beta * beta
        #
        # is asymptotically stable for k_rho > 0, k_beta < 0, k_alpha > k_rho. It curves
        # into the goal and arrives square, which is exactly the manoeuvre a strafe was
        # standing in for.
        self.nofix_since = None
        gx, gy = goal
        rho = math.hypot(gx, gy)
        if rho > self.acquire_pose_tol:
            ahead = self._min_range_ahead()
            if gx > 0 and ahead is not None and ahead < self.min_safe:
                self.get_logger().warn(
                    f"acquiring: {ahead:.2f} m ahead is closer than the goal; holding",
                    throttle_duration_sec=3.0)
                self._stop()
                return
            alpha = math.atan2(gy, gx)
            # Reverse towards a goal that is behind, rather than turning to face it.
            #
            # This control law has a singularity at alpha = pi, and the robot found it.
            # With the goal 1.15 m directly behind, the bearing read -178, -179, -180,
            # +180 degrees on consecutive measurements, so the commanded turn changed
            # sign every time and the base sat still: measured, 1.15 m to go for the
            # whole of the state's two-minute budget, commanding a correction throughout.
            #
            # Driving backwards is also simply the right manoeuvre. Turning 180 degrees
            # to reach a point behind means pointing the camera away from the shelf, and
            # the shelf is the only thing that tells this controller where it is. The
            # standard treatment is to fold the rear half-plane onto the front one and
            # let the speed carry the sign.
            backwards = abs(alpha) > math.pi / 2.0
            if backwards:
                alpha -= math.copysign(math.pi, alpha)
            beta = -alpha
            cmd = Twist()
            # Speed from the DISTANCE to the goal, not from how far ahead it is. Written
            # the second way it stalls exactly when it is needed: a goal 20 mm ahead and
            # 350 mm to the side asked for 9 mm/s, and a base that cannot strafe cannot
            # close a lateral offset without driving. Measured, it crept in at 20 mm per
            # correction and ran out of time with 190 mm still to go.
            speed = float(min(self.max_fwd, max(self.creep_speed,
                                                self.pose_gain * rho)))
            cmd.linear.x = -speed if backwards else speed
            cmd.angular.z = self._turn(1.30 * alpha - 0.40 * beta, 1.0, floor=0.0)
            self.pub_cmd.publish(cmd)
            self.get_logger().info(
                f"acquiring: {'reversing' if backwards else 'driving'} to the pose, "
                f"{rho:.2f} m to go ({gx:+.2f} ahead, {gy:+.2f} across, bearing "
                f"{math.degrees(alpha):+.0f} deg)",
                throttle_duration_sec=3.0)
            return

        # Arrived. Square up now, with the driving finished.
        if angle is not None and abs(angle) > self.square_tol:
            cmd = Twist()
            cmd.angular.z = self._turn(-angle, 0.8)
            self.pub_cmd.publish(cmd)
            self.get_logger().info(
                "acquiring: at the pose, squaring, face %+.1f deg"
                % math.degrees(angle), throttle_duration_sec=3.0)
            return
        self._stop()

        if not located:
            self._stop()
            self.get_logger().warn(
                "acquiring: centred but the book is not located yet",
                throttle_duration_sec=3.0)
            return

        self.get_logger().info(
            f"book acquired at {self._range_ahead() or float('nan'):.2f} m; closing in")
        self.approach_target = self.standoff
        self._enter(State.APPROACH)

    def _acquire_goal(self):
        """Where to stand, in base_footprint: the standoff in front of the book.

        Metres from the book's own 3D fix rather than pixels from its bearing. The pixel
        error says which way to move and nothing about how far, so a controller built on
        it cannot know when to stop; the book point is measured to 15-35 mm in x against
        ground truth and carries the lateral offset directly.
        """
        # The anchored fix if there is one, the live sighting if there is not.
        #
        # Requiring the anchor made this return None for the whole state whenever the
        # anchor had not yet formed -- it needs eight agreeing candidates and accepts
        # nothing while it is collecting them -- so the pose controller had nowhere to
        # drive at exactly the moment it was needed, and the approach stood squared and
        # reporting no fix while perception published the book on every frame.
        #
        # The live sighting is a measurement of the thing itself, good to 15-35 mm in x
        # against ground truth, and perception now follows the same book from frame to
        # frame rather than whichever is nearest the middle of the picture. The anchor is
        # still preferred where it exists, because it averages several sightings; it is
        # no longer a precondition for moving at all.
        if not self._book_located():
            return None
        target = self._target_in_base()
        if target is None:
            if self.book_live is None:
                return None
            target = self.book_live
        # The ACQUIRE range, not the final standoff. This is the checkpoint where the
        # whole column still fits in frame so the row can be read and the book found;
        # driving to the grasping standoff from here closes the distance before the
        # alignment is done, and at 0.65 m the column no longer fits. Measured on the run
        # that found this: the robot arrived 0.65 m from the shelf and 1.67 m along it,
        # with perception reporting no red book in view for the rest of the state.
        return (float(target[0]) - self.acquire_range, float(target[1]))

    def _note_bearing(self, value: float) -> None:
        """Track how often a NEW bearing arrives, which sets how fast we may turn.

        New, not merely republished: the same number arriving twice says the camera has
        not looked again, and turning on it is turning blind.
        """
        now = self._now()
        if self.bearing_last_value is None or value != self.bearing_last_value:
            if self.bearing_last_at is not None:
                gap = now - self.bearing_last_at
                if 0.0 < gap < 10.0:
                    # Slow to trust a fast frame, quick to believe a slow one: an
                    # over-estimate of the rate is what caused the problem.
                    self.bearing_period = (max(gap, self.bearing_period)
                                           if gap > self.bearing_period
                                           else 0.7 * self.bearing_period + 0.3 * gap)
            self.bearing_last_at = now
            self.bearing_last_value = value

    def _bearing_rad(self, error_px: float) -> float:
        return math.atan2(error_px, self.focal_px)

    def _do_centre(self) -> None:
        column_cx = self._column_cx_fresh()
        # Arriving is not the same as being measured again, and steering on the
        # difference is what stalled this state. Watched at 1788546390: perception
        # published nothing at all on target_column_x for seven seconds -- confirmed
        # with `ros2 topic hz`, no messages against a single publisher -- while this
        # state reported a fresh bearing on every tick, zero idle ticks, and a pixel
        # error that would not move through 25 degrees of measured base rotation.
        #
        # _column_cx_fresh tests when a message last ARRIVED. What a controller needs
        # is when the number last CHANGED, because a value republished unchanged while
        # the robot turns is a photograph of where the marker used to be. The age of
        # the change was in the log the whole time, growing 0.2, 0.3, 1.4 s across
        # exactly those samples, and nothing was reading it.
        if column_cx is not None and self.column_cx_new_at is not None:
            if (self._now() - self.column_cx_new_at) > self.bearing_stale:
                column_cx = None
        if column_cx is None:
            # Lost the marker. Stop, and if it stays lost, go and look for it again
            # rather than stand here until the state times out.
            #
            # Standing still was survivable while perception offered a bearing from
            # whichever target-coloured book was nearest the middle of the frame, because
            # something always arrived. Now that bearing is only offered from close
            # range -- it was steering the robot at books in the wrong column -- so a
            # marker that leaves the frame really does mean no bearing at all, and this
            # state has to be able to recover from it.
            self._stop()
            self.centre_ticks += 1
            self.centre_idle += 1
            if self.lost_since is None:
                self.lost_since = self._now()
            elif self._now() - self.lost_since > self.lost_grace:
                self.get_logger().warn(
                    "the target column has been out of view for %.0f s; searching again"
                    % self.lost_grace)
                self.lost_since = None
                self._enter(State.SEARCH)
            return
        self.lost_since = None
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
        #
        # The rate comes from how often the bearing arrives, not from a fixed gain on
        # the pixel error. See turn_for: at the 0.45 rad/s the old gain saturated to,
        # with a bearing that updates about every two simulated seconds, the base swept
        # most of its field of view between one look and the next and lost the marker
        # three times in a single run.
        self._note_bearing(column_cx)
        cmd = Twist()
        wanted = turn_for(-self._bearing_rad(error_px), self.bearing_period,
                          fastest=self.max_yaw)
        cmd.angular.z = float(max(-self.max_yaw, min(
            self.max_yaw, wanted - self.turn_damping * self.yaw_rate)))
        self.pub_cmd.publish(cmd)
        age = (0.0 if self.column_cx_new_at is None
               else self._now() - self.column_cx_new_at)
        self.centre_ticks += 1
        self.centre_asked += abs(cmd.angular.z) * TICK_PERIOD
        if self.centre_yaw0 is None and self.odom_yaw is not None:
            self.centre_yaw0 = self.odom_yaw
        turned = 0.0
        if self.centre_yaw0 is not None and self.odom_yaw is not None:
            turned = abs((self.odom_yaw - self.centre_yaw0 + math.pi)
                         % (2 * math.pi) - math.pi)
        self.get_logger().info(
            "centring: %+.0f px off (%.0f deg), turning at %+.2f rad/s; bearing "
            "changed %.1f s ago, arrives every %.1f s; asked for %.0f deg of turn "
            "over %d ticks (%d of them idle), base has turned %.0f deg"
            % (error_px, math.degrees(self._bearing_rad(error_px)), cmd.angular.z,
               age, self.bearing_period, math.degrees(self.centre_asked),
               self.centre_ticks, self.centre_idle, math.degrees(turned)),
            throttle_duration_sec=3.0)

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

        # Do not close past this without having actually seen the book.
        #
        # Driving in on the LiDAR alone is how the robot cornered itself: it arrived
        # where the target row was out of the camera's reach, and every state after that
        # needed a sighting it could no longer get. Holding here instead costs time and
        # nothing else -- the head is still aimed at the row, perception is still
        # looking, and the moment a fix arrives the drive resumes with a range it can
        # trust.
        if not self._book_located() and ahead < self.blind_floor:
            self._stop()
            self.get_logger().warn(
                "holding at %.2f m: the book has not been seen and driving closer on "
                "the LiDAR alone is how the view gets lost" % ahead,
                throttle_duration_sec=5.0)
            return

        cmd = Twist()
        cmd.linear.x = min(self.max_fwd, max(0.05, 0.5 * remaining))

        # Hold square to the shelf all the way in.
        #
        # This state drove forward and strafed sideways and never once commanded a yaw,
        # on the reasoning that strafing keeps the heading square by itself. It does not
        # on this base: there is no friction across the roller axis, so any rotation the
        # robot picks up it keeps, and a metre of driving is long enough to accumulate a
        # lot of it. Measured on the run that found this, the approach reached the
        # standoff 38.9 degrees off square, and at that angle the head camera is looking
        # along the shelf rather than at it -- so the book left the frame, the squaring
        # could not fit a face either, and the run ended with the target in plain view
        # of nothing.
        #
        # This is the third state in this file to need the same thing. Any state that
        # commands motion on this base has to hold the heading too; the base will not
        # do it for us.
        face = self._heading_error()
        cmd.angular.z = (self._turn(-face, 0.8, floor=0.0) if face is not None
                         else self._turn(0.0, 0.0, floor=0.0))

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
        angle = self._heading_error()
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
                # Out of retreats. Whether that ends the run depends on whether there
                # is anything better to go on than this line fit.
                #
                # The refusal above is right when the base's heading is unknown: the
                # grasp reaches along base x, so an unsquared base aims the hand across
                # the book faces. But the heading is no longer unknown. The acquire
                # state drives to a POSE with a pose controller, and arriving square is
                # what that controller is for; the grasp then re-aims from perception
                # and closes the last centimetres on a servo. So a noisy line fit is no
                # longer the only thing standing between the arm and the book.
                #
                # And this was ending runs that were going well. Measured: the approach
                # tracked the book correctly the whole way in, reached the standoff,
                # read the face at -20.7 degrees, lost the fit, and failed at 94.7 s
                # with a good fix on the book in hand. The scan at the standoff is
                # dominated by the shelf's own openings -- the beam reads through the
                # unstocked bottom shelf to the back panel, so the returns are a mix of
                # front edges and back panel and no line fits them.
                #
                # So: proceed if the book is located, which means perception can see
                # the target and the grasp will re-aim at it anyway. Refuse only when
                # blind, which is the case the original reasoning was about.
                if self._book_located():
                    self.get_logger().warn(
                        "no flat face after %d retreat(s), but the book is in view; "
                        "going on to verify and letting the grasp re-aim, rather than "
                        "ending a run over a line fit"
                        % self.retreats)
                    self.square_lost_since = None
                    self._enter(State.VERIFY)
                    return
                self.get_logger().error(
                    "still no flat face after %d retreat(s) and no fix on the book; "
                    "giving up rather than grasping blind and unsquared"
                    % self.retreats)
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
