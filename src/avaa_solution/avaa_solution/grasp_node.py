"""Grasp controller — take the identified book off the shelf.

Runs once the approach controller has the base in front of the target column.

    SCENE     put the shelf into the planning scene, measured from the book
    PREGRASP  plan to a posture in front of the book, collision free
    OPEN      gripper wide
    ADVANCE   straight line in, along the shelf normal
    CLAMP     close on the spine
    LIFT      small rise to take the weight off the shelf
    WITHDRAW  straight back out, book held
    STOW      return to the driving posture
    DONE

Where the target comes from
---------------------------
Height comes from the identified row, not from depth. Distance and lateral offset come
from depth. Each is used where it is trustworthy: measured against ground truth over 100
published points, depth is good to about 7 mm sideways at grasping range but carries a
systematic upward bias of 121 to 193 mm depending on distance. Rows are 0.33 m apart, so
that vertical error would be up to 0.6 of the gap between shelves; the row identification
pins the height instead.

Who decides what
----------------
The analytic IK in kinematics/arm_chain.py says WHERE the arm should be. MoveIt says HOW
to get there without hitting anything. Pose goals are not used: MoveIt solves those with
KDL, a numerical solver on a redundant eight-joint chain with a 50 ms budget, and it
failed on every point the analytic solver reaches to a tenth of a millimetre.

This split is the whole reason the node is now short. It used to carry a staging state, a
posture-clearance cost, a shelf-opening cost, a joint-limit cost, a sag correction and a
hand-rolled Cartesian interpolator, all of them approximations of a collision checker,
and all of them are gone. What they were working around is recorded in the git history;
the summary is that an arm with four spare degrees of freedom will reach a correct point
by a path straight through the shelf unless something is actually checking.
"""

import math
import threading
from collections import deque
from enum import Enum
from typing import List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PointStamped, Quaternion
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from avaa_solution.kinematics.arm_chain import ArmChain
from avaa_solution.moveit_client import MoveItClient, error_name

TOPIC_TARGET_ROW = "/avaa/perception/target_row"
TOPIC_BOOK_POINT = "/avaa/perception/target_book_point"
TOPIC_STATE = "/avaa/grasp/state"
GRIPPER_TOPIC = "/gripper_left_controller_raw/joint_trajectory"

ARM_JOINTS = [f"arm_left_{i}_joint" for i in range(1, 8)]
CHAIN_JOINTS = ["torso_lift_joint"] + ARM_JOINTS

# Gripper command values. The span curve is roughly span = 0.0285 + 0.80 * joint.
# 0.055 opens the jaws to about 72 mm, against a book 30 mm thick.
#
# 0.040 -- 60.5 mm -- looked like ample clearance and was not. The jaws do not stay
# where they are put while the arm is moving: measured through a reach, the finger went
# from 0.045 to 0.006 between leaving the pre-grasp and arriving at the book, which is a
# span of 32 mm closing on a 30 mm book, and the book was never touched. Standing still
# it holds the commanded position to within 8 mm, so this is the arm's motion
# back-driving a joint held by a proportional velocity loop, not a slow leak. Starting
# wider leaves margin for that, and _hold_gripper keeps re-asserting it.
#
# The supplied clamp node limits the command to 0.069, so this is well inside what the
# organisers allow.
# All the way open. The supplied clamp node limits the command to 0.069, so this is the
# widest the organisers allow, and the margin is the whole game: measured through a
# reach the pads still closed from 0.060 to 0.042, which brings their inner faces to
# about 30 mm -- exactly the thickness of the book -- so instead of passing either side
# of it they met its front corners and shoved it 140 mm deeper into the shelf over
# successive runs. Wide open the inner faces are about 60 mm apart, which tolerates a
# 15 mm aiming error instead of none.
# Open enough to clear the book, and no more.
#
# Measured with tools/fitcheck.py, the pad surfaces sit 11.3 mm from the grasp centre
# line at finger 0.0297 and 29.9 mm at 0.0700, so about 461 mm of pad travel per unit of
# finger. The book's own surface is 15 mm out. That fixes both numbers that matter:
#
#   0.068  pads 29 mm out, 14 mm clearance a side, 30 mm of finger to reach the book
#   0.052  pads 21 mm out,  6 mm clearance a side, 14 mm of finger to reach the book
#
# and the gripper is force limited to about 4 mm/s, so those are 7.6 s and 3.5 s of
# closing. Seven seconds is too long: the base turns a quarter degree a second, which
# drags the pads 21 mm sideways at arm's length while they close, and the last run
# closed on air beside an untouched book. Six millimetres of clearance is enough for a
# reach that arrives within five, and halving the closing time halves the drag.
GRIPPER_OPEN = 0.052
# Below this the jaws are too narrow to take the book and clamping would close on air.
GRIPPER_OPEN_MIN = 0.044
# Measured from TF, not from the span model: at a commanded 0.000 the joint settles at
# 0.0026 with the fingertips 30.4 mm apart, and the book is 30.0 mm thick. The jaws were
# closing around the book with 0.2 mm to spare on each side and gripping nothing, which
# is why a perfectly aimed grasp -- gripper 8 mm from plan, jaws centred on the spine and
# 20 mm inside the front face -- still lifted nothing. -0.001 is the joint's lower limit,
# about 27.7 mm, which actually squeezes.
# A light interference, and a narrow band to hit.
#
# The fingers are position-driven and do not stop on contact: watched through TF they
# close 60.5 -> 37.6 -> 28.7 mm around a book 30.0 mm thick, straight past the point
# where it should have stopped them. So the only grip available is whatever the
# contact solver pushes back with, and how much interference to ask for is the whole
# question.
#
# Measured span curve: 30.4 mm at joint 0.0026, about 0.8 mm of span per 0.001 of
# joint. Both ends of the band fail, in opposite ways:
#
#   -0.0010   1.5 mm a side   ejects the book during the clamp, 2.9 rad of tip
#
# All of that was measured at position_proportional_gain 0.1, where the finger
# controller could not resolve a contact: it crawled toward its target and the book
# was flicked out on the way. At gain 5 the finger stops ON the book -- commanded to
# a span of 29.2 mm it settles at 29.6 with the book undisturbed, which is the book
# holding it open. So the clamp goes back to asking for a firm close, because now
# the controller can push against something instead of sliding past it.
#    0.0009   0.65 mm a side  ejects the book during the clamp, 1.2 rad of tip
#    0.0015   0.25 mm a side  survives the clamp, slips during the withdraw and
#                             topples after 78 mm
GRIPPER_CLAMP = -0.0010

DEFAULT_ROW_HEIGHTS = [1.391, 1.061, 0.731, 0.401]

# Folded, and collision free -- which the previous tuck was not.
#
# [-0.5, -2.4, 0.0, -2.4, 0.0, 0.0, 0.0] puts arm_left_2 through arm_left_5 against
# torso_base_link and torso_lift_link. Gazebo never objected, because self-collision is
# not checked there, so it went unnoticed for the whole project until MoveIt refused to
# plan from it: the start state was invalid and every request came back 'Motion planning
# start tree could not be initialized'.
#
# This one was found by sampling folded postures and asking /check_state_validity,
# keeping the most compact one with at least 0.15 rad of room at every joint stop. It is
# also tighter than the old one: the gripper sits 0.29 m from the base axis rather than
# 0.49 m.
TUCK_POSE = [2.1521, 0.3824, 1.2785, -2.1517, 0.8325, 0.1926, 1.3944]
TUCK_TORSO = 0.15

# The wrist, in base_link: reach along +x, close the fingers across y. Both come from the
# URDF -- the fingers sit at y = +/-0.0288 offset +0.0756 along z in gripper_left_base_link,
# and the grasping frame adds a -pi/2 pitch, which turns those into local x for the
# approach and local y for the finger travel.
GRASP_APPROACH = [1.0, 0.0, 0.0]
GRASP_CLOSING = [0.0, 1.0, 0.0]

# A book is 0.25 m tall and stands on the board below it, so the board sits this far under
# the row height. The 0.02 is half the board thickness.
BOARD_DROP = 0.125 + 0.02

# Height of the left shoulder above base_link when the torso is fully down, measured from
# the chain: arm_left_1 sits at z = 0.677 + torso.
SHOULDER_BASE_Z = 0.677
SHELF_DEPTH = 0.30
SHELF_WIDTH = 4.8


def row_to_height(row: int, heights: List[float], top_down: bool = True) -> Optional[float]:
    """Height of a shelf row, or None if the row is outside the shelf."""
    if not 1 <= row <= len(heights):
        return None
    return heights[row - 1] if top_down else heights[len(heights) - row]


def facing_shelf() -> Quaternion:
    """Identity: the grasping frame reaches along base x and closes across base y."""
    return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


class State(Enum):
    IDLE = "idle"
    SCENE = "scene"
    RAISE = "raising"
    PREGRASP = "pregrasp"
    OPEN = "opening"
    ADVANCE = "advancing"
    CLAMP = "clamping"
    LIFT = "lifting"
    WITHDRAW = "withdrawing"
    STOW = "stowing"
    DONE = "done"
    FAILED = "failed"


class GraspNode(Node):
    def __init__(self) -> None:
        super().__init__("avaa_grasp")

        self.declare_parameter("row_heights", DEFAULT_ROW_HEIGHTS)
        self.declare_parameter("rows_top_down", True)
        # How far in front of the book face the pre-grasp sits. MoveIt plans a
        # collision-free path to it, so this no longer has to be far enough out to keep a
        # dumb joint-space interpolation clear of the shelf.
        self.declare_parameter("standoff_m", 0.15)
        # How far past the book face the GRASPING FRAME is driven, which is not where the
        # jaws are: they sit 29.7 mm behind that frame, measured from TF. At 0.05 a
        # perfect arrival put the jaws 20 mm inside the face, and an arrival 29 mm short
        # in depth -- inside tolerance, and dead on sideways and in height -- closed them
        # 9 mm in FRONT of the book. 0.11 puts them in the middle of a 160 mm book.
        self.declare_parameter("grasp_depth_m", 0.11)
        # No lift before withdrawing.
        #
        # Lifting 30 mm off the shelf before pulling out is the textbook move and it
        # is wrong here: placed to the millimetre with the jaws provably around the
        # book, the book still came out at 90 degrees of tip. A grip this marginal
        # topples the book the moment it is asked to carry its weight, where sliding
        # it straight out leaves the shelf taking the weight the whole way.
        self.declare_parameter("lift_m", 0.0)
        # Slowly. The fingers are position-driven and do not stop on contact, so a
        # fast close arrives at the book with speed and flicks it out; the book has
        # left with a radian of tip while the gripper sat exactly on target. Six
        # seconds gives the contact solver time to push back instead.
        self.declare_parameter("gripper_time_sec", 6.0)
        self.declare_parameter("auto_start", True)
        # A Cartesian path that only gets part way is a blocked reach, not a failure to
        # follow one. Below this, the grasp is abandoned rather than half-attempted.
        self.declare_parameter("min_reach_fraction", 0.9)
        # Arrival is judged per axis, because the axes are not equivalent. Sideways
        # is the tight one: the jaws open to 60.5 mm around a 30 mm book, so much
        # more than 15 mm off centre and a finger meets the front of the book
        # instead of passing it. Depth and height are forgiving -- the book is
        # 160 mm deep and 250 mm tall -- and grasp_depth_m puts the jaws in the
        # middle of it, so being short in depth costs margin rather than the grasp.
        self.declare_parameter("arrival_tol_lateral_m", 0.012)
        self.declare_parameter("arrival_tol_depth_m", 0.030)
        self.declare_parameter("arrival_tol_height_m", 0.030)
        self.declare_parameter("reach_attempts", 6)
        # How much of the reach is left for a second, separately aimed leg.
        #
        # The long leg takes about thirty seconds and the base slides for all of it, so
        # a target aimed once at the start is stale by the time the jaws get there. This
        # stops short, re-aims at the book as it is then, and covers the rest in a few
        # seconds, which is short enough that the drift over it does not matter.
        #
        # 35 mm rather than 60. The base still turns about a quarter of a degree per
        # second even with the wheels given friction, which at arm's length is 3 mm/s
        # sideways, and the jaws have about 13 mm of clearance either side of the book.
        # That buys roughly four seconds between the last look and the jaws closing.
        self.declare_parameter("final_approach_m", 0.035)
        # How far below the middle of the book to grip it.
        #
        # The book is 250 mm tall and 30 mm thick, standing free on a shelf board it
        # cannot slide on, so it is a lever with a very short base. Tipping it needs
        # only F > m g t / 2h: with the pads at mid height that is 0.3 * 9.81 * 0.015
        # / 0.125, about a third of a newton. A parallel gripper closes symmetrically
        # about its own centre line, so any centring error means one pad touches first
        # and pushes -- and a third of a newton is nothing, so the book goes over before
        # the second pad arrives. Every failed run ended with it tipped, not slipped.
        #
        # 45 mm rather than the 80 that the arithmetic wants. At 80 the arm simply
        # cannot get there: twelve pre-grasp postures in a row came back with no IK
        # solution at all, none of them in collision and none over torque, so it is
        # reach and not clutter that runs out. 45 mm still lifts the tipping threshold
        # from 0.35 N to about 0.55 N, and the pads are 34 mm tall so it stays clear of
        # the shelf board the book stands on.
        self.declare_parameter("grasp_below_centre_m", 0.045)
        # How still the book has to look, and for how long, before the jaws close.
        self.declare_parameter("quiet_spread_m", 0.004)
        self.declare_parameter("quiet_for_sec", 2.5)
        self.declare_parameter("quiet_timeout_sec", 25.0)
        # How many postures that reach are compared before choosing one.
        self.declare_parameter("posture_choices", 4)
        # The pre-grasp is a staging point 0.15 m in front of the book, not the
        # grasp. The Cartesian reach that follows targets the book in absolute
        # terms, so a centimetre of error here is corrected rather than carried,
        # and holding it to the same 12 mm as the grasp itself failed runs on a
        # millimetre while the arm sat 13 mm out and stable.
        self.declare_parameter("pregrasp_tol_m", 0.045)
        # Reject postures the arm will not hold. Stalled 0.18 rad short of its last
        # waypoint with no contact anywhere in Gazebo, the arm had arm_left_2 at
        # 42 Nm of 43, arm_left_3 at 27 of 26 and arm_left_4 at 25 of 26 -- three
        # joints saturated simply holding still.
        #
        # The estimate in arm_chain.gravity_torque reads about a quarter of the
        # measured effort, because effort from a position controller includes
        # whatever it is spending on friction and on correcting its own error, not
        # just the load. So the threshold is set against the estimate rather than
        # against the rating: on the posture that failed it reads 6.6 Nm on a 26 Nm
        # joint, and anything at or above a quarter of rated is treated as too dear.
        self.declare_parameter("max_torque_fraction", 0.24)
        # The nearest the pre-grasp may sit to the base.
        # The nearest the pre-grasp may sit to the base. It was 0.42 because the
        # tucked arms reached 0.49 forward and the robot could not work any closer;
        # that was the badly stowed right arm, and both now fold inside 0.31. Working
        # closer is what actually lowers the torque the arm has to hold, which is the
        # thing that has been stopping the reach.
        self.declare_parameter("min_pregrasp_x_m", 0.34)
        # How long to let the arm settle after MoveIt says the trajectory is done.
        # It is not done: the controller reports success when the trajectory time
        # has elapsed, and this arm is still travelling. Judged immediately, a reach
        # measured 86 mm short and 120 mm off sideways, and three retries three
        # seconds apart changed it by 3 mm because nothing had time to move.
        # Waiting for the arm to stop is not free. The base slides about 3.4 mm/s and
        # takes the book with it, and the jaws only have some 14 mm of clearance either
        # side of a 30 mm book, so five seconds of standing still is a third of the
        # budget spent on nothing. Down to 1.8: the base still turns about a quarter of
        # a degree per second whatever the wheels are given, which is 3 mm/s sideways at
        # arm's length, and the window that has to fit inside 13 mm is the last leg plus
        # this settle plus the arrival check. Arriving early is cheap -- the check simply
        # fails and the reach is retried -- while waiting is not.
        self.declare_parameter("settle_sec", 1.8)

        self.row_heights = list(
            self.get_parameter("row_heights").get_parameter_value().double_array_value
        ) or DEFAULT_ROW_HEIGHTS
        self.rows_top_down = bool(self.get_parameter("rows_top_down").value)
        self.standoff = float(self.get_parameter("standoff_m").value)
        self.grasp_depth = float(self.get_parameter("grasp_depth_m").value)
        self.lift = float(self.get_parameter("lift_m").value)
        self.gripper_time = float(self.get_parameter("gripper_time_sec").value)
        self.min_fraction = float(self.get_parameter("min_reach_fraction").value)
        self.tol_lateral = float(self.get_parameter("arrival_tol_lateral_m").value)
        self.tol_depth = float(self.get_parameter("arrival_tol_depth_m").value)
        self.tol_height = float(self.get_parameter("arrival_tol_height_m").value)
        self.reach_attempts = int(self.get_parameter("reach_attempts").value)
        self.final_approach = float(self.get_parameter("final_approach_m").value)
        self.below_centre = float(
            self.get_parameter("grasp_below_centre_m").value)
        self.quiet_spread = float(self.get_parameter("quiet_spread_m").value)
        self.quiet_for = float(self.get_parameter("quiet_for_sec").value)
        self.quiet_timeout = float(self.get_parameter("quiet_timeout_sec").value)
        self.quiet_waited = None
        self.posture_choices = int(
            self.get_parameter("posture_choices").value)
        self.pregrasp_tol = float(self.get_parameter("pregrasp_tol_m").value)
        self.max_torque = float(
            self.get_parameter("max_torque_fraction").value)
        self.min_pregrasp_x = float(
            self.get_parameter("min_pregrasp_x_m").value)
        self.reaches = 0
        self.leg = 0
        self.leg_target = None
        self.settle = float(self.get_parameter("settle_sec").value)
        self.settled_at = None

        self.chain = ArmChain.from_urdf()
        self.state = State.IDLE
        self.row: Optional[int] = None
        self.book: Optional[np.ndarray] = None
        # Roughly a second of frames at 15 Hz. Long enough to bury an outlier, short
        # enough that the target is still current when the arm commits. Depth is noisy:
        # 76 mm of bias with 149 mm of spread at grasping range, so planning from
        # whichever sample arrived last puts the hand anywhere within a hand-width.
        self.book_points = deque(maxlen=15)
        self.book_points_min = 8
        self.joints = {}
        # The targets, held in odom as well as in base_link.
        #
        # The base does not stay where it is put. Measured, it reads 2.5 degrees of yaw
        # immediately after being placed square, and it keeps moving while the arm swings
        # out: a run that planned for the book at base y=+0.159 had it at +0.257 by the
        # time the gripper was closing, a shift of nearly 0.1 m. Everything downstream
        # aims at a target expressed in base_link, so a base that turns takes the target
        # with it and the arm reaches confidently at where the book used to be.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pre_odom = None
        self.grasp_odom = None

        self.pre_target = None
        self.grasp_target = None
        # When perception last saw the book, and how old a sighting may be before it
        # stops being worth trusting. The arm occludes the book on the way in, so this
        # is expected to go stale during the reach itself.
        self.book_at = None
        self.settled_since = None
        self.gripper_held_at = None
        self.reopens = 0
        self.reopen_at = None
        # Generous, because the sighting rate is not guaranteed: perception competes for
        # the same CPU as the planner, and the harness that stands in for it competes
        # with Gazebo's own query service.
        self.book_fresh = 6.0
        self.pre_solution = None

        self.motion_thread: Optional[threading.Thread] = None
        self.motion_result = None
        self.motion_label = ""

        self.create_subscription(Int32, TOPIC_TARGET_ROW, self._on_row, 10)
        self.create_subscription(PointStamped, TOPIC_BOOK_POINT, self._on_book, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.pub_gripper = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
        self.pub_state = self.create_publisher(String, TOPIC_STATE, 10)

        self.moveit = MoveItClient("avaa_grasp_moveit")
        self.get_logger().info("waiting for move_group...")
        if not self.moveit.wait_until_ready(60.0):
            self.get_logger().error(
                "move_group is not running. Start it with "
                "'ros2 launch avaa_solution moveit.launch.py'; without a planner this "
                "controller will not reach into a shelf without hitting it.")
        else:
            self.get_logger().info("move_group connected")

        self.create_timer(0.2, self._tick)
        self.get_logger().info(
            "grasp ready — rows top-down, heights %s" % self.row_heights)

    # ------------------------------------------------------------------ inputs

    def _on_row(self, msg: Int32) -> None:
        self.row = int(msg.data)

    def _on_book(self, msg: PointStamped) -> None:
        """Hold the median of recent sightings, not the latest one."""
        self.book_points.append([msg.point.x, msg.point.y, msg.point.z])
        if len(self.book_points) < self.book_points_min:
            return
        self.book = np.median(np.array(self.book_points), axis=0)
        self.book_at = self.get_clock().now()

    def _on_joints(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self.joints[name] = position

    # ------------------------------------------------------------------ helpers

    def _enter(self, state: State) -> None:
        if state is not self.state:
            self.get_logger().info(f"{self.state.value} -> {state.value}")
            self.state = state

    def _send_gripper(self, value: float) -> None:
        """Command the gripper directly rather than planning for it.

        It has one joint, no collision geometry worth planning around, and the thing it is
        about to touch is deliberately not in the planning scene -- MoveIt would refuse a
        grasp that touches the book, which is the entire objective.
        """
        traj = JointTrajectory()
        traj.joint_names = ["gripper_left_finger_joint"]
        point = JointTrajectoryPoint()
        point.positions = [float(value)]
        point.time_from_start = Duration(
            sec=int(self.gripper_time),
            nanosec=int((self.gripper_time % 1.0) * 1e9))
        traj.points = [point]
        self.pub_gripper.publish(traj)

    def _hold_gripper(self, value: float, period: float = 0.6) -> None:
        """Keep re-asserting a gripper position, briefly, so it is not back-driven.

        A single trajectory point is a promise the controller keeps only as well as the
        joint lets it. Re-sending the same long trajectory every tick would be worse --
        each publish restarts it, so it never completes -- so this sends a short one at
        a low rate, which acts as a hold rather than a move.
        """
        now = self.get_clock().now()
        if (self.gripper_held_at is not None
                and (now - self.gripper_held_at).nanoseconds / 1e9 < period):
            return
        self.gripper_held_at = now
        traj = JointTrajectory()
        traj.joint_names = ["gripper_left_finger_joint"]
        point = JointTrajectoryPoint()
        point.positions = [float(value)]
        point.time_from_start = Duration(sec=0, nanosec=int(0.4 * 1e9))
        traj.points = [point]
        self.pub_gripper.publish(traj)

    def _start(self, label: str, function) -> None:
        """Run a MoveIt call on a worker thread, so this node keeps serving callbacks."""
        self.motion_label = label
        self.motion_result = None
        self.settled_at = None

        def run():
            try:
                self.motion_result = function()
            except Exception as exc:  # noqa: BLE001 - a failed motion is not a crash
                self.get_logger().error("%s raised %s" % (label, exc))
                self.motion_result = (99999, 0.0)

        self.motion_thread = threading.Thread(target=run, daemon=True)
        self.motion_thread.start()

    def _finished(self):
        """(code, fraction) once the motion is done AND the arm has stopped moving.

        Two different meanings of finished. MoveIt returns when the trajectory it sent
        has run out of time; the arm is still catching up for several seconds after that,
        and this one lags far enough that the difference is the whole grasp.
        """
        if self.motion_result is None:
            return None
        if self.settled_at is None:
            self.settled_at = self._now()
            return None
        if self._now() - self.settled_at < self.settle:
            return None
        result = self.motion_result
        return result if isinstance(result, tuple) else (result, 1.0)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _current_joints(self):
        """Give the arm as it is now, in group order, or None if not yet known."""
        try:
            return [self.joints[name] for name in CHAIN_JOINTS]
        except KeyError:
            return None

    def _to_odom(self, point):
        """Put a base_link point into odom, or None if there is no transform yet."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "odom", "base_link", rclpy.time.Time())
        except Exception:  # noqa: BLE001 - the transform may not be up yet
            return None
        return self._apply(tf, point)

    def _from_odom(self, point):
        """Bring an odom point back into base_link, using the base pose as it is NOW."""
        if point is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", "odom", rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        return self._apply(tf, point)

    @staticmethod
    def _apply(tf, point):
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

    def _refresh_targets(self) -> None:
        """Re-aim at the book, preferring what perception can see over what odom claims.

        The base does not merely roll, it slides across its own wheels. They are mecanum
        wheels, modelled the way mecanum wheels have to be -- mu 0.8 along the roller
        axis and mu2 0.0 across it -- so there is no friction whatever in one direction
        and nothing damps a sideways drift once it starts. Measured on a freshly spawned
        robot standing untouched with the arm still: it travels 8 mm and turns 2 degrees
        every 30 seconds, while wheel odometry reports 2 mm and 0.06 degrees.

        That is the whole problem with anchoring the target in odom. Odom is computed
        from wheel rotations, and a wheel that is not turning reports nothing, so odom
        cannot see a slide at all: the frame the target was anchored in is itself
        drifting, and re-deriving from it corrects almost none of the error. Two degrees
        of unseen yaw moves a target at arm's length by about 25 mm, and the book is
        30 mm wide.

        Perception measures the book through a camera bolted to the base, so when the
        base slides the reading changes with it and the correction comes out right.

        There is no odom fallback, because odom is not merely blind here, it is wrong in
        the worst possible way. Holding the base still means driving the wheels against
        the slide, and wheel odometry faithfully integrates every one of those turns
        while the robot does not move: measured during a run held to 17 mm of true
        error, odom had accumulated 813 mm of motion that never happened, and applying
        it moved a correct target most of a metre. So when the sighting is stale --
        which it will be, since the arm occludes the book on the way in -- the right
        thing is to keep aiming where the book last actually was.
        """
        fresh = None
        if self.book is not None and self.book_at is not None:
            age = (self.get_clock().now() - self.book_at).nanoseconds / 1e9
            if age <= self.book_fresh:
                # The median of the last six sightings, not of all fifteen.
                #
                # Fifteen sightings span several seconds, and several seconds ago the
                # base was somewhere else: the median drags the target back towards
                # wherever the book appeared to be while the arm was still swinging. It
                # cost 20 mm sideways on a reach that had already waited for the base to
                # go still, which was enough to fail the 12 mm arrival check and send
                # the arm round again -- and going round again kicks the base, so the
                # next look is worse than the last. Six samples is enough to reject a
                # bad frame and recent enough to mean now.
                recent = list(self.book_points)[-6:]
                fresh = np.median(np.array(recent, dtype=float), axis=0)

        if fresh is not None and self.row is not None:
            height = row_to_height(self.row, self.row_heights, self.rows_top_down)
            if height is not None:
                height -= self.below_centre
                face_x = float(fresh[0])
                y = float(fresh[1])
                pre = np.array([max(face_x - self.standoff, self.min_pregrasp_x),
                                y, height])
                grasp = np.array([face_x + self.grasp_depth, y, height])
                moved = float(np.linalg.norm(grasp - self.grasp_target))
                if moved > 0.005:
                    self.get_logger().info(
                        "the base has slid; re-aimed from perception, target moved "
                        "%.0f mm" % (moved * 1000))
                self.face_x = face_x
                self.pre_target = pre
                self.grasp_target = grasp
                self.pre_odom = self._to_odom(pre)
                self.grasp_odom = self._to_odom(grasp)
                return

        age = ("never" if self.book_at is None else "%.1f s old"
               % ((self.get_clock().now() - self.book_at).nanoseconds / 1e9))
        self.get_logger().warn(
            "no fresh sighting (%s); holding the last measured target rather than "
            "correcting from odom, which counts wheel turns the base did not make"
            % age)

    def _gripper_now(self) -> Optional[np.ndarray]:
        """Where the gripper actually is, from the joints rather than from a promise."""
        try:
            values = [self.joints[name] for name in CHAIN_JOINTS]
        except KeyError:
            return None
        return self.chain.position(values)

    def _arrived(self, target, tolerance=None) -> Optional[bool]:
        """Whether the gripper is at ``target`` closely enough on each axis.

        MoveIt reporting a completed trajectory is not the same as the arm being there.
        This arm lags its commands badly: a Cartesian reach came back planned and executed
        with the gripper 87 mm short in depth and 119 mm off sideways, and the jaws closed
        on air beside the book. The action result says the controller finished, and only
        forward kinematics says where it finished.
        """
        where = self._gripper_now()
        if where is None:
            return None
        offset = np.asarray(where) - np.asarray(target)
        if tolerance is not None:
            return bool(np.all(np.abs(offset) <= tolerance))
        return (abs(float(offset[0])) <= self.tol_depth
                and abs(float(offset[1])) <= self.tol_lateral
                and abs(float(offset[2])) <= self.tol_height)

    def _miss(self, target) -> str:
        where = self._gripper_now()
        if where is None:
            return "position unknown"
        offset = np.asarray(where) - np.asarray(target)
        return ("%+.0f mm depth, %+.0f mm sideways, %+.0f mm height"
                % (offset[0] * 1000, offset[1] * 1000, offset[2] * 1000))

    def _pose(self, point) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in point)
        pose.orientation = facing_shelf()
        return pose

    def _plan_targets(self) -> bool:
        """Work out where the pre-grasp and the grasp are. False if unreachable."""
        if self.row is None or self.book is None:
            return False
        height = row_to_height(self.row, self.row_heights, self.rows_top_down)
        if height is None:
            self.get_logger().error(
                f"row {self.row} is outside 1..{len(self.row_heights)}")
            return False
        height -= self.below_centre

        face_x = float(self.book[0])
        y = float(self.book[1])
        self.face_x = face_x
        # Held back from the book, but never folded back into the robot. Standing
        # closer is what keeps the arm out of torque saturation, and at a 0.58 m
        # standoff a fixed 0.15 m of clearance put the pre-grasp at x=0.31, which no
        # posture reaches at all.
        pre_x = max(face_x - self.standoff, self.min_pregrasp_x)
        self.pre_target = np.array([pre_x, y, height])
        self.grasp_target = np.array([face_x + self.grasp_depth, y, height])

        self.pre_odom = self._to_odom(self.pre_target)
        self.grasp_odom = self._to_odom(self.grasp_target)
        if self.grasp_odom is None:
            self.get_logger().warn(
                "no odom transform; the target cannot be held against base movement")

        self.get_logger().info(
            f"row {self.row} at z={height:.3f}; book face at x={face_x:.3f} "
            f"y={y:.3f}; reaching to x={self.grasp_target[0]:.3f}")
        return True

    def _reachable_and_clear(self, target, attempts: int = 12):
        """Find a posture reaching ``target`` that the planner will accept as a goal.

        The analytic solver picks arbitrarily among the many postures that reach a point
        on an eight-joint arm, and some of them have the elbow inside the shelf. Asking
        it repeatedly and checking each answer costs a few hundred milliseconds and
        removes a failure that was otherwise intermittent and unexplained.
        """
        first = None
        for attempt in range(attempts):
            solution = self.chain.ik(
                target, approach=GRASP_APPROACH, closing=GRASP_CLOSING)
            if solution is None:
                continue
            first = first or solution
            if self.moveit.state_valid(CHAIN_JOINTS, solution) is not False:
                if attempt:
                    self.get_logger().info(
                        "took %d tries to find a posture clear of the shelf"
                        % (attempt + 1))
                return solution
        if first is not None:
            self.get_logger().warn(
                "every posture reaching %s looks to be in collision; "
                "trying the best of them anyway" % np.round(target, 3).tolist())
        return first

    def _extension(self, solution) -> float:
        """How far the arm holds itself out, as a proxy for the load on it.

        The mass-based estimate in arm_chain.gravity_torque is not trustworthy enough to
        choose postures with -- against measured effort it reads a quarter of the load on
        one joint and six times it on another -- but the geometry underneath is not in
        doubt: a joint has to hold the weight of everything beyond it, times how far out
        that weight sits. Summing the horizontal distance of each frame from the shoulder
        needs no masses and gets the ordering right.

        This matters because collision-free and reachable is not the same as holdable.
        Planned 100 per cent clear and executed, a reach still stopped 0.18 rad short with
        arm_left_2, 3 and 4 all at their effort limits and nothing touching anything.
        """
        origins = self.chain.joint_origins(solution)
        shoulder = origins[1]
        return float(sum(
            math.hypot(float(p[0]) - float(shoulder[0]),
                       float(p[1]) - float(shoulder[1]))
            for p in origins[2:]))

    def _affordable(self, solution) -> bool:
        """Whether the arm could hold this posture, by the static torque estimate."""
        torques = self.chain.gravity_torque(solution)
        limits = self.chain.effort_limits()
        for name, torque, limit in zip(CHAIN_JOINTS, torques, limits):
            if name == "torso_lift_joint" or limit <= 0.0:
                continue
            if torque > self.max_torque * limit:
                return False
        return True

    def _torque_share(self, solution) -> float:
        """Worst joint load as a fraction of its rating, by the static estimate.

        The absolute numbers are not trustworthy -- against measured effort this reads a
        quarter of the load on one joint and six times it on another -- so it is no good
        as a threshold. As an ordering between postures for the same target it does work:
        the posture that put the gripper on the book to within a millimetre scored 82 per
        cent, and one that stalled 2.2 rad short scored 159.
        """
        torques = self.chain.gravity_torque(solution)
        limits = self.chain.effort_limits()
        return max((t / l for n, t, l in zip(CHAIN_JOINTS, torques, limits)
                    if n != "torso_lift_joint" and l > 0.0), default=0.0)

    def _worst_torque(self, solution) -> str:
        torques = self.chain.gravity_torque(solution)
        limits = self.chain.effort_limits()
        pairs = [(t / l if l else 0.0, n) for n, t, l in
                 zip(CHAIN_JOINTS, torques, limits) if n != "torso_lift_joint"]
        share, name = max(pairs)
        return "%s at %.0f%% of rated" % (name, share * 100.0)

    def _clear(self, solution) -> bool:
        """Whether a posture is collision free, judged on the whole robot.

        Everything that checks a posture goes through here, so none of them can quietly
        ask about eight joints and get an answer about a robot with its other arm
        somewhere else.
        """
        names, values = self._full_state(solution)
        return self.moveit.state_valid(names, values) is not False

    def _full_state(self, solution):
        """Describe the whole robot as it is now, with the arm moved to ``solution``.

        Both wheels and the other arm matter to a collision check, so a start state that
        only describes eight joints is not a description of this robot.
        """
        names = list(self.joints.keys())
        values = [self.joints[name] for name in names]
        index = {name: position for position, name in enumerate(names)}
        for name, value in zip(CHAIN_JOINTS, solution):
            if name in index:
                values[index[name]] = float(value)
            else:
                names.append(name)
                values.append(float(value))
        return names, values

    def _posture_that_can_reach_in(self, attempts: int = 12):
        """Choose a pre-grasp posture by whether the reach works FROM it.

        Two conditions, and the second is the one that was missing. A posture has to be
        collision free, and it has to be one the arm can travel in a straight line from,
        all the way to the book. On an eight-joint arm most postures satisfy the first
        and only some satisfy the second: the elbow ends up somewhere that has to swing
        through the shelf to let the hand advance, and the reach stops a third of the way
        in with the gripper still outside.

        Testing costs a Cartesian plan per candidate, no motion, and it replaces a
        failure that was intermittent and looked like bad luck.
        """
        best = None
        usable = None
        candidates = []
        rejected = 0
        expensive = 0
        # The first attempt is seeded from where the arm already is. Close to the shelf
        # most solutions for the pre-grasp fold the arm back towards the body and are in
        # collision, so an unbiased search spends its budget on postures that were never
        # going to work: eight random tries found nothing at a 0.60 m standoff while a
        # seeded walk through the same reach was clear at every step.
        seed = self._current_joints()
        for attempt in range(attempts):
            solution = self.chain.ik(
                self.pre_target, seed=seed if attempt == 0 else None,
                approach=GRASP_APPROACH, closing=GRASP_CLOSING,
                prefer=self._posture_cost(float(self.pre_target[2])),
                pin=self._torso_for(float(self.pre_target[2])))
            if solution is None:
                continue
            if not self._clear(solution):
                rejected += 1
                continue
            # The torque estimate is NOT used as a gate. Measured against the real
            # thing it reads 6.6 Nm on a joint drawing 27, and 159 per cent of rated on a
            # joint drawing 2.5 -- wrong in both directions, so a threshold on it rejects
            # good postures and passes bad ones. Kept for reporting, where the ranking
            # within one posture is still informative, and not for deciding.
            _ = expensive
            fraction = self._reach_clearance(
                solution, self.pre_target, self.grasp_target)
            if best is None or fraction > best[0]:
                best = (fraction, solution)
            if fraction >= self.min_fraction:
                # Collect a few that reach, then take the least loaded.
                #
                # Taking the first one that reaches is not enough: postures that reach
                # perfectly well can still be ones the arm cannot hold, and the run that
                # placed the gripper within a millimetre and the run that stalled 2.2 rad
                # short both reported a fully clear reach. What separated them was load.
                #
                # Picking the least EXTENDED was tried instead and is worse -- that
                # measure sums distances over every frame, so it counts links rather than
                # reach, and it chose a posture that arrived 238 mm off.
                share = self._torque_share(solution)
                if usable is None or share < usable[0]:
                    usable = (share, solution, attempt + 1)
                if len(candidates) >= self.posture_choices:
                    break
                candidates.append(solution)
        if usable is not None:
            share, solution, tries = usable
            self.get_logger().info(
                # Just the attempt number. The count of candidates collected is not the
                # count of attempts made -- the search stops collecting once it has
                # enough, so the winning attempt can be numbered higher than the total
                # and the line read "try 6 of 5".
                "pre-grasp posture from try %d that reached, %d considered: worst "
                "joint at %.0f%% of rated"
                % (tries, len(candidates), share * 100.0))
            return solution

        if best is None:
            self.get_logger().error(
                "no usable posture for the pre-grasp in %d tries: %d in collision, "
                "%d beyond what the arm can hold" % (attempts, rejected, expensive))
            return None
        self.get_logger().warn(
            "no posture gives a clear reach; the best of %d is %.0f%% and will be tried"
            % (attempts, best[0] * 100.0))
        return best[1]

    def _torso_for(self, height: float) -> dict:
        """Pin the torso so the shoulder sits level with the target.

        The torso lift is rated 2000 N and the arm joints 26 Nm, so every centimetre of
        height the arm provides instead of the torso is bought at the wrong end of the
        robot. Reaching for a book at z=1.061 with the torso left free, the solver chose
        0.10 and stretched the arm up nearly a metre; the arm then sagged half a metre
        short with arm_left_3 over a radian out.

        The shoulder is aimed ABOVE the target rather than level with it, so the arm
        reaches down. Level was the first guess and it is wrong twice over: with the
        wrist held to reach into a shelf there is often no level solution at all, and
        where there is one the arm is holding its own weight out horizontally. Reaching
        down puts gravity on the same side as the motion. Asked freely, the solver picks
        exactly this: for a row at z=0.731 it settles on a torso of 0.304, a quarter of a
        metre above the book.

        The slack lets it trade a little height for reach, without handing the whole job
        back to the arm.
        """
        ideal = float(np.clip(height - SHOULDER_BASE_Z + 0.25, 0.0, 0.35))
        return {"torso_lift_joint": (ideal, 0.10)}

    def _posture_cost(self, height: float):
        """Prefer postures that get their height from the torso, not from the arm.

        The torso lift is rated 2000 N and the arm joints 26 Nm, so every centimetre of
        height the arm provides instead of the torso is bought at the wrong end of the
        robot. Nothing was expressing that: solving for a book at z=1.061 the IK picked a
        torso of 0.10 and stretched the arm up nearly a metre, and the arm then sagged
        half a metre short of the target with arm_left_3 over a radian out.

        The ideal torso puts the shoulder level with the target, so the arm reaches
        horizontally and holds almost nothing. Where that is outside the torso range the
        cost simply pulls as far that way as it goes.
        """
        ideal = float(np.clip(height - SHOULDER_BASE_Z, 0.0, 0.35))

        def cost(values):
            # Dominant: use the strong joint for height.
            torso = abs(float(values[0]) - ideal)
            # Mild: keep the elbow off its stops, which is where it sags onto.
            crowding = sum(max(0.0, 0.20 - min(v - lo, hi - v))
                           for v, (lo, hi) in zip(values, self.chain.limits))
            return 10.0 * torso + crowding

        return cost

    def _nudge(self, wanted, gain: float = 1.4, limit: float = 0.35) -> bool:
        """Re-command a posture with the standing error added, to beat the stiction.

        Every arm joint carries 2 Nm of Coulomb friction. As the position error shrinks
        the controller torque shrinks with it, and once it falls under the friction the
        joint stops: commanded to a posture and left alone for 55 seconds, the arm went
        from 0.71 rad of total error to 0.22 and was still creeping. Waiting it out would
        take minutes per motion.

        Asking for the error again on top of the target puts the stall point where the
        target is. This is the same idea that failed when it was tried in Cartesian
        space -- there, reaching further needed more extension, which was the thing going
        wrong -- but in joint space it changes nothing about how far the arm has to
        stretch. It only asks each joint to push a little harder.
        """
        actual = self._current_joints()
        if actual is None:
            return False
        error = [w - a for w, a in zip(wanted, actual)]
        if max(abs(e) for e in error) < 1e-3:
            return True
        # Gain above one, because asking for exactly the standing error again does not
        # break friction: three rounds of that moved arm_left_3 from 0.085 rad short to
        # 0.079. But not far above one, because 2.5 overshot and came back by the same
        # amount every round -- +0.133 short, then 0.133 past, then short again -- which
        # is a limit cycle rather than a correction.
        #
        # If the overshoot is in collision the answer is a smaller one, not none: giving
        # up and re-commanding the plain target puts the arm back exactly where it
        # already failed to get past, which is what produced the oscillation.
        overshoot = None
        for scale in (1.0, 0.6, 0.3):
            candidate = [float(np.clip(w + scale * gain * e, lo, hi))
                         for w, e, (lo, hi) in zip(wanted, error, self.chain.limits)]
            candidate = [float(np.clip(c, w - limit, w + limit))
                         for c, w in zip(candidate, wanted)]
            if self._clear(candidate):
                overshoot = candidate
                break
        if overshoot is None:
            self.get_logger().warn(
                "every overshoot posture is in collision; leaving the arm where it is")
            return False
        worst = max(range(len(error)), key=lambda i: abs(error[i]))
        self.get_logger().info(
            "nudging: %s is %+.3f short, asking for %+.3f past it"
            % (CHAIN_JOINTS[worst], error[worst], gain * error[worst]))
        self._start("nudge", lambda: self.moveit.execute_path(
            CHAIN_JOINTS, [actual, overshoot]))
        return True

    def _reach_clearance(self, start_solution, start_point, end_point,
                         steps: int = 4) -> float:
        """How far along a straight line this posture can get, as a fraction.

        The same walk as _straight_path, without building the trajectory, so a candidate
        posture can be judged before the arm is asked to adopt it. It replaces MoveIt's
        compute_cartesian_path for this purpose: that solves IK with KDL at every step and
        returns 0.0 for reaches whose every waypoint the analytic solver finds and
        /check_state_validity passes. Every candidate scored zero and the choice was
        effectively random.
        """
        start_point = np.asarray(start_point, dtype=float)
        end_point = np.asarray(end_point, dtype=float)
        seed = list(start_solution)
        for step in range(1, steps + 1):
            point = start_point + (step / float(steps)) * (end_point - start_point)
            # No posture cost here: the seed from the previous waypoint and the
            # pinned torso already decide the shape, and asking for a cost makes the
            # solver run forty restarts instead of stopping at the first success.
            # Measured, that is 0.78 s per call against 0.02 s for a collision check,
            # and it turned a posture search into a three-minute wait.
            solution = self.chain.ik(point, seed=seed, approach=GRASP_APPROACH,
                                     closing=GRASP_CLOSING,
                                     pin=self._torso_for(float(point[2])))
            if solution is None or not self._clear(solution):
                return (step - 1) / float(steps)
            seed = solution
        return 1.0

    def _straight_path(self, start_solution, start_point, end_point, steps: int = 8):
        """Joint waypoints tracing a straight line in space, each one checked.

        The analytic solver is seeded from the previous waypoint so consecutive postures
        are neighbours and the elbow does not flip halfway along, and every one is put to
        /check_state_validity before it goes anywhere near the arm. Returns None if the
        line cannot be walked, which is a real answer: it means the reach is obstructed.
        """
        start_point = np.asarray(start_point, dtype=float)
        end_point = np.asarray(end_point, dtype=float)
        waypoints = [list(start_solution)]
        seed = list(start_solution)
        for step in range(1, steps + 1):
            point = start_point + (step / float(steps)) * (end_point - start_point)
            solution = self.chain.ik(point, seed=seed, approach=GRASP_APPROACH,
                                     closing=GRASP_CLOSING,
                                     pin=self._torso_for(float(point[2])))
            if solution is None:
                self.get_logger().warn(
                    "no posture reaches %.0f%% of the way along the line"
                    % (100.0 * step / steps))
                return None
            if not self._clear(solution):
                self.get_logger().warn(
                    "the reach is obstructed %.0f%% of the way in"
                    % (100.0 * step / steps))
                return None
            waypoints.append(solution)
            seed = solution
        return waypoints

    def _add_shelf(self) -> bool:
        """Describe the shelf to the planner, as boards rather than as a block.

        The openings between the boards are the only way in, so a single box would make
        every grasp unreachable and a shelf that is not there at all lets the arm plan
        straight through it. Placed from the measured book face, so nothing here needs to
        know where the shelf is in the world.
        """
        centre_x = self.face_x + SHELF_DEPTH / 2.0 - 0.05
        placed = 0
        for index, height in enumerate(self.row_heights):
            if self.moveit.add_box(
                    "shelf_board_%d" % index, "base_link",
                    (centre_x, 0.0, height - BOARD_DROP),
                    (SHELF_DEPTH, SHELF_WIDTH, 0.04)):
                placed += 1
        if self.moveit.add_box("shelf_back", "base_link",
                               (self.face_x + SHELF_DEPTH, 0.0, 0.9),
                               (0.04, SHELF_WIDTH, 1.8)):
            placed += 1
        self.get_logger().info(
            "shelf in the planning scene: %d of %d boxes placed"
            % (placed, len(self.row_heights) + 1))
        return placed == len(self.row_heights) + 1

    # ------------------------------------------------------------------ states

    def _tick(self) -> None:
        self.pub_state.publish(String(data=self.state.value))
        handler = {
            State.IDLE: self._do_idle,
            State.SCENE: self._do_scene,
            State.RAISE: self._do_raise,
            State.PREGRASP: self._do_pregrasp,
            State.OPEN: self._do_open,
            State.ADVANCE: self._do_advance,
            State.CLAMP: self._do_clamp,
            State.LIFT: self._do_lift,
            State.WITHDRAW: self._do_withdraw,
            State.STOW: self._do_stow,
        }.get(self.state)
        if handler:
            handler()

    def _do_idle(self) -> None:
        if not bool(self.get_parameter("auto_start").value):
            return
        if self.row is None or self.book is None:
            return
        if not self._plan_targets():
            self._enter(State.FAILED)
            return
        self._enter(State.SCENE)

    def _do_scene(self) -> None:
        if not self._add_shelf():
            self.get_logger().error("could not describe the shelf to the planner")
            self._enter(State.FAILED)
            return
        # The posture is chosen only now, with the shelf in the scene at the distance
        # just measured. Choosing first and describing the shelf afterwards judged every
        # candidate against whatever geometry was left over from the previous run, which
        # produced a posture reported as 100 per cent clear and then an invalid motion
        # plan the moment the real boards went in.
        self.pre_solution = self._posture_that_can_reach_in()
        if self.pre_solution is None:
            self.get_logger().error(
                "no posture reaches the pre-grasp %s and can then travel into the shelf"
                % np.round(self.pre_target, 3).tolist())
            self._enter(State.FAILED)
            return

        # Get the height first, with the arm still folded.
        #
        # Going straight from the tuck to the pre-grasp asks the planner to raise the
        # torso most of its travel while unfolding the arm past a shelf, and for the
        # middle rows it comes back "invalid motion plan". Raising first turns one hard
        # problem into two easy ones: the torso moves with the arm out of the way, and
        # the arm then unfolds at the height it will work at.
        ideal = self._torso_for(float(self.pre_target[2]))["torso_lift_joint"][0]
        self.raised = [ideal] + list(TUCK_POSE)
        self._enter(State.RAISE)
        self._start("raise", lambda: self.moveit.move_to_joints(
            CHAIN_JOINTS, self.raised, timeout=180.0))

    def _do_raise(self) -> None:
        done = self._finished()
        if done is None:
            return
        code, _ = done
        if code != 1:
            self.get_logger().warn(
                "could not raise the torso first (%s); reaching from where we are"
                % error_name(code))
        else:
            self.get_logger().info(
                "torso at the row, arm still folded; reaching out")
        # Re-aim the pre-grasp before going to it.
        #
        # pre_solution is a joint-space posture, worked out back in the SCENE state, and
        # joint space is measured from the base. A base that has slid forwards in the
        # thirty to sixty seconds since then carries the whole arm forwards with it, so
        # the posture that was a safe 150 mm in front of the book arrives inside it.
        # That is not a theory: watched against ground truth, the book was standing when
        # this move began and lying down when it finished, knocked over by the arm on
        # its way to a pre-grasp that had gone stale.
        self._refresh_targets()
        moved = self.chain.ik(
            self.pre_target, seed=list(self.pre_solution),
            approach=GRASP_APPROACH, closing=GRASP_CLOSING,
            pin=self._torso_for(float(self.pre_target[2])))
        if moved is not None and self._clear(moved):
            shift = float(np.linalg.norm(
                np.asarray(self.chain.fk(list(moved))[:3, 3])
                - np.asarray(self.chain.fk(list(self.pre_solution))[:3, 3])))
            if shift > 0.005:
                self.get_logger().info(
                    "pre-grasp re-aimed %.0f mm for the base having moved"
                    % (shift * 1000))
            self.pre_solution = moved
        elif moved is None:
            self.get_logger().warn(
                "could not re-aim the pre-grasp; going to the one planned earlier")
        else:
            self.get_logger().warn(
                "the re-aimed pre-grasp is not collision free; keeping the planned one")

        self._enter(State.PREGRASP)
        self._start("pre-grasp", lambda: self.moveit.move_to_joints(
            CHAIN_JOINTS, self.pre_solution, timeout=240.0))

    def _do_pregrasp(self) -> None:
        done = self._finished()
        if done is None:
            return
        code, _ = done
        if code != 1:
            self.get_logger().error(
                "could not reach the pre-grasp %s: %s%s"
                % (np.round(self.pre_target, 3).tolist(), error_name(code),
                   " -- " + self.moveit.last_failure
                   if self.moveit.last_failure else ""))
            self._enter(State.FAILED)
            return
        # The reach that follows is a straight line from wherever the arm actually is,
        # so being short here moves sideways there. One run left the pre-grasp 119 mm off
        # in y and the Cartesian path faithfully carried that error into the shelf.
        if self._arrived(self.pre_target, self.pregrasp_tol) is False:
            self.reaches += 1
            if self.reaches < self.reach_attempts:
                self.get_logger().warn(
                    "pre-grasp finished %s; going again (%d of %d)"
                    % (self._miss(self.pre_target), self.reaches, self.reach_attempts))
                if self._nudge(self.pre_solution):
                    return
                self._start("pre-grasp", lambda: self.moveit.move_to_joints(
                    CHAIN_JOINTS, self.pre_solution, timeout=240.0))
                return
            self.get_logger().error(
                "cannot settle at the pre-grasp: %s" % self._miss(self.pre_target))
            self._enter(State.FAILED)
            return
        self.reaches = 0
        self.get_logger().info(
            "at the pre-grasp (%s); opening the gripper" % self._miss(self.pre_target))
        self._send_gripper(GRIPPER_OPEN)
        self.open_at = self.get_clock().now()
        self._enter(State.OPEN)

    def _do_open(self) -> None:
        waited = (self.get_clock().now() - self.open_at).nanoseconds / 1e9
        if waited < self.gripper_time + 0.5:
            return
        self._refresh_targets()
        # Build the reach from the posture that was checked, not from where the arm
        # happens to have stopped. They are a centimetre apart, but the analytic solver
        # seeds from whatever it is given and a different seed lands on a different
        # branch of a redundant arm: the same reach validated 100 per cent clear from the
        # planned posture and 12 per cent from the arm one centimetre away from it.
        # MoveIt tolerates a start that close, so the validated path is the one to run.
        #
        # Start it from the point that posture actually reaches, not from pre_target.
        # Re-aiming moves pre_target -- 51 mm in one run -- and the line was then being
        # drawn from a point the starting posture is not at, so its first step was a
        # jump rather than a step and the whole reach came back obstructed 12 per cent
        # in. The posture and the point it reaches have to be the same thing.
        start_point = np.asarray(self.chain.fk(list(self.pre_solution))[:3, 3],
                                 dtype=float)

        # Stop short of the book, so the last stretch can be aimed separately.
        leg_end = np.asarray(self.grasp_target, dtype=float)
        span = leg_end - start_point
        reach = float(np.linalg.norm(span))
        if reach > self.final_approach + 0.02:
            leg_end = leg_end - (self.final_approach / reach) * span
            self.leg = 1
        else:
            self.leg = 2
        self.leg_target = leg_end

        path = self._straight_path(self.pre_solution, start_point, leg_end)
        if path is None:
            self._enter(State.FAILED)
            return
        self.reach_path = path
        self.get_logger().info(
            "reaching along %d waypoints to %s, stopping %.0f mm short so the last "
            "stretch can be re-aimed"
            % (len(path), np.round(leg_end, 3).tolist(),
               float(np.linalg.norm(np.asarray(self.grasp_target) - leg_end)) * 1000))
        self._enter(State.ADVANCE)
        self._start("reach", lambda: self.moveit.execute_path(CHAIN_JOINTS, path))

    def _do_advance(self) -> None:
        # Keep the jaws open on the way in. They are back-driven by the arm's own
        # motion, and arriving at the book with them already shut is how a reach that
        # lands within 2 mm still closes on nothing.
        self._hold_gripper(GRIPPER_OPEN)
        done = self._finished()
        if done is None:
            return
        code, fraction = done
        if fraction < self.min_fraction:
            # Not a controller failure: the planner is saying the reach is obstructed.
            self.get_logger().error(
                "the reach into the shelf is blocked: only %.0f%% of it is clear"
                % (fraction * 100.0))
            self._enter(State.FAILED)
            return
        if code != 1:
            self.get_logger().error("the reach failed: %s" % error_name(code))
            self._enter(State.FAILED)
            return

        # Planned and executed is not the same as arrived. Report the shortfall in
        # joint space as well: if the joints are where the last waypoint asked and the
        # gripper is not, the kinematics are wrong, and if the joints are short then the
        # controller did not follow.
        current = self._current_joints()
        if current is not None and getattr(self, "reach_path", None):
            gaps = [a - b for a, b in zip(current, self.reach_path[-1])]
            worst = max(range(len(gaps)), key=lambda i: abs(gaps[i]))
            lo, hi = self.chain.limits[worst]
            self.get_logger().info(
                "after the reach, %s wanted %+.3f got %+.3f (gap %+.3f, limits "
                "%+.2f..%+.2f), total %.3f"
                % (CHAIN_JOINTS[worst], self.reach_path[-1][worst], current[worst],
                   gaps[worst], lo, hi, sum(abs(g) for g in gaps)))

        # First leg done: re-aim at the book as it is now, and cover the rest.
        if self.leg == 1:
            before = np.asarray(self.grasp_target, dtype=float)
            self._refresh_targets()
            drifted = float(np.linalg.norm(np.asarray(self.grasp_target) - before))
            here = self._gripper_now()
            start = self._current_joints()
            if here is None or start is None:
                self.get_logger().error("cannot see the arm to aim the last stretch")
                self._enter(State.FAILED)
                return
            remaining = float(np.linalg.norm(np.asarray(self.grasp_target) - here))
            final = self._straight_path(start, here, self.grasp_target, steps=4)
            if final is None:
                self.get_logger().error(
                    "no clear line for the last %.0f mm" % (remaining * 1000))
                self._enter(State.FAILED)
                return
            self.leg = 2
            self.reach_path = final
            self.get_logger().info(
                "at the staging point; the book moved %.0f mm while I reached, "
                "closing the last %.0f mm" % (drifted * 1000, remaining * 1000))
            self._start("reach", lambda: self.moveit.execute_path(CHAIN_JOINTS, final))
            return

        # Wait for the base to go quiet before closing.
        #
        # The base is not disturbed by the world, it is disturbed by this arm. Measured
        # with nothing commanding the base: tucked and idle it drifts about 0.05 deg/s,
        # and after two swings of the arm out to a reach posture and back it is turning
        # at 1.22 deg/s and takes tens of seconds to calm down. The gripper is force
        # limited and takes three to seven seconds to close, so closing while the base is
        # still ringing sweeps the jaws tens of millimetres sideways -- 1.22 deg/s for
        # 3.5 s is 51 mm at arm's length, past a book that is 30 mm thick.
        #
        # There is no need to measure the base to know this. Perception watches the book
        # through a camera bolted to the base, so a base that is moving is a book that
        # appears to move. When the sightings stop changing, the base has settled.
        spread = None
        if len(self.book_points) >= 6:
            recent = np.array(list(self.book_points)[-6:], dtype=float)
            spread = float(np.max(recent, axis=0)[1] - np.min(recent, axis=0)[1])
        now = self._now()
        if spread is not None and spread <= self.quiet_spread:
            if self.settled_since is None:
                self.settled_since = now
        else:
            self.settled_since = None
        if self.settled_since is None or now - self.settled_since < self.quiet_for:
            if self.quiet_waited is None:
                self.quiet_waited = now
            if now - self.quiet_waited < self.quiet_timeout:
                self._hold_gripper(GRIPPER_OPEN)
                return
            self.get_logger().warn(
                "the base has not gone quiet in %.0f s (book still wandering %s); "
                "closing anyway"
                % (self.quiet_timeout,
                   "unknown" if spread is None else "%.0f mm" % (spread * 1000)))
        else:
            self.get_logger().info(
                "the base is quiet -- the book has held to %.0f mm for %.0f s; closing"
                % (spread * 1000, now - self.settled_since))
        self.quiet_waited = None

        # Re-aim before judging arrival, not only before setting off.
        #
        # The reach takes the better part of a minute and the base does not stay still
        # for it. It cannot: the wheels have no friction across the roller axis, so the
        # moment of the extending arm slides the whole robot. Measured against ground
        # truth through one reach, the jaws finished 110 mm short of the book and 244 mm
        # to the side of it, while this controller reported arriving 2 mm from its
        # target -- and both were true, because the target is held in base_link and
        # base_link had moved.
        #
        # Re-aiming here turns that into an ordinary miss, which the retry below already
        # knows how to correct.
        aimed_at = np.array(self.grasp_target, dtype=float)
        self._refresh_targets()
        shifted = float(np.linalg.norm(np.asarray(self.grasp_target) - aimed_at))
        if shifted > 0.005:
            self.get_logger().warn(
                "the book has moved %.0f mm in base_link since the reach was planned; "
                "judging arrival against where it is now" % (shifted * 1000))

        if self._arrived(self.grasp_target) is False:
            self.reaches += 1
            if self.reaches < self.reach_attempts:
                self.get_logger().warn(
                    "reach finished %s; going again (%d of %d)"
                    % (self._miss(self.grasp_target), self.reaches,
                       self.reach_attempts))
                # Push into the standing error rather than replay the same path. The
                # path was followed; the joints just stopped short of its last point.
                # Only when the target has held still, though: if it has moved, that
                # waypoint is aimed at where the book used to be and pushing harder
                # towards it makes the miss worse rather than better.
                if (shifted <= 0.005 and self.reach_path
                        and self._nudge(self.reach_path[-1])):
                    return
                again = self._straight_path(
                    self._current_joints(), self._gripper_now(), self.grasp_target)
                if again is None:
                    self._enter(State.FAILED)
                    return
                self._start("reach", lambda: self.moveit.execute_path(
                    CHAIN_JOINTS, again))
                return
            self.get_logger().error(
                "still %s after %d reaches; not closing on empty air"
                % (self._miss(self.grasp_target), self.reaches))
            self._enter(State.FAILED)
            return

        # Arriving is not the same as arriving ready. Check the jaws are still open
        # wide enough to take the book before closing them, because clamping a gripper
        # that is already nearly shut cannot fail loudly -- it simply reports a
        # successful grasp of nothing, which is exactly what happened.
        finger = self.joints.get("gripper_left_finger_joint")
        if finger is not None and finger < GRIPPER_OPEN_MIN:
            # Wait here rather than going back to the OPEN state. OPEN rebuilds the
            # reach from the pre-grasp posture, and the arm is now deep inside the
            # shelf, so that plan would start from a state the arm is nowhere near.
            if self.reopen_at is None:
                self.reopen_at = self._now()
                self.reopens += 1
                self.get_logger().warn(
                    "the jaws are only %.0f mm apart at the book; re-opening in place "
                    "(attempt %d of 3)"
                    % (27.1 + (finger + 0.001) * 814.6, self.reopens))
                self.gripper_held_at = None
                self._send_gripper(GRIPPER_OPEN)
                return
            if self._now() - self.reopen_at < self.gripper_time + 1.0:
                return
            self.reopen_at = None
            if self.reopens >= 3:
                self.get_logger().error(
                    "the jaws will not stay open; refusing to clamp on air")
                self._enter(State.FAILED)
            return
        self.reopen_at = None

        self.get_logger().info(
            "gripper is at the book (%s), jaws %.0f mm apart; clamping"
            % (self._miss(self.grasp_target),
               27.1 + ((finger if finger is not None else GRIPPER_OPEN) + 0.001) * 814.6))
        self._send_gripper(GRIPPER_CLAMP)
        self.clamp_at = self.get_clock().now()
        self._enter(State.CLAMP)

    def _do_clamp(self) -> None:
        waited = (self.get_clock().now() - self.clamp_at).nanoseconds / 1e9
        if waited < self.gripper_time + 1.0:
            return
        lifted = self.grasp_target + np.array([0.0, 0.0, self.lift])
        self._enter(State.LIFT)
        path = self._straight_path(
            self._current_joints(), self._gripper_now(), lifted, steps=4)
        if path is None:
            self.get_logger().warn("no clear lift; withdrawing without one")
            self.motion_result = (1, 1.0)
            return
        self._start("lift", lambda: self.moveit.execute_path(CHAIN_JOINTS, path))

    def _do_lift(self) -> None:
        done = self._finished()
        if done is None:
            return
        code, fraction = done
        if code != 1:
            self.get_logger().warn(
                "the lift did not complete (%s); withdrawing anyway" % error_name(code))
        self._enter(State.WITHDRAW)

        # Come out the way we went in, measured from where the arm actually is.
        #
        # Aiming at the stored pre-grasp point assumes the arm reached the grasp point
        # exactly and that the base has not moved since, and neither is true: the reach
        # routinely stops tens of millimetres short, and the base slides throughout. A
        # run failed here with "no clear way back out" while the arm was sitting in an
        # ordinary posture inside the shelf, because the line it was asked to check ran
        # from where the arm was to a point that no longer meant anything.
        #
        # Reversing the advance displacement instead is the same motion the arm has
        # already proved it can make, just backwards.
        here = self._gripper_now()
        if here is None:
            self.get_logger().error("cannot see the gripper to withdraw")
            self._enter(State.FAILED)
            return
        back = np.asarray(self.pre_target, dtype=float) - np.asarray(
            self.grasp_target, dtype=float)
        back = back + np.array([0.0, 0.0, self.lift])

        # If the full retreat will not check out, take what is available. Half out of
        # the shelf with the book is worth more than stopping inside it.
        path = None
        for share in (1.0, 0.7, 0.45):
            out = here + share * back
            path = self._straight_path(self._current_joints(), here, out)
            if path is not None:
                if share < 1.0:
                    self.get_logger().warn(
                        "only %.0f%% of the way out is clear; taking that"
                        % (share * 100))
                break
        if path is None:
            self.get_logger().error("no clear way back out with the book")
            self._enter(State.FAILED)
            return
        self._start("withdraw", lambda: self.moveit.execute_path(CHAIN_JOINTS, path))

    def _do_withdraw(self) -> None:
        done = self._finished()
        if done is None:
            return
        code, fraction = done
        if fraction < self.min_fraction:
            self.get_logger().error(
                "could not draw the book clear: %.0f%% of the way out"
                % (fraction * 100.0))
            self._enter(State.FAILED)
            return
        self._enter(State.STOW)
        self._start("stow", lambda: self.moveit.move_to_joints(
            CHAIN_JOINTS, [TUCK_TORSO] + TUCK_POSE, timeout=240.0))

    def _do_stow(self) -> None:
        done = self._finished()
        if done is None:
            return
        code, _ = done
        if code != 1:
            self.get_logger().warn("stow did not complete: %s" % error_name(code))
        self.get_logger().info("book grasped and stowed")
        self._enter(State.DONE)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.moveit.shutdown()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
