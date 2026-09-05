"""A small synchronous client for move_group.

MoveIt 2 in Humble has no Python bindings -- moveit_py arrived later -- so this talks to
move_group over its action and service interfaces directly. It is deliberately small: the
grasp controller needs four things, and everything else MoveIt offers is a distraction.

    plan and move to a joint configuration
    plan and move so the gripper reaches a pose
    move the gripper along a straight line, collision-checked
    tell the planner where the shelf is

Why this exists at all. The first version of this solution drove the arm from its own
analytic IK, which places a gripper accurately and knows nothing about what the arm passes
through on the way. Reaching into a 0.33 m shelf opening, the arm repeatedly arrived at
correct points by paths that went through the shelf: link contact sensors caught
arm_left_6 at world z=0.44, under the lowest shelf surface at 0.587, while its target sat
at 1.247. Four separate heuristics were added to the IK -- off the joint stops, near the
previous posture, out of the shelf volume, inside the target opening -- and each helped
and none of them is a collision checker.

The client owns its own node and spins it on a background thread. Blocking on a future
from inside another node's timer callback is re-entrant spinning, which rclpy refuses, and
the grasp controller is a timer-driven state machine.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence

import rclpy
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
from moveit_msgs.msg import RobotState
from moveit_msgs.msg import RobotTrajectory
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath,
                             GetStateValidity)
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from builtin_interfaces.msg import Duration
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_GROUP = "arm_left_torso"
GRIPPER_GROUP = "gripper_left"
TIP_LINK = "gripper_left_grasping_link"
PLANNING_FRAME = "base_link"


def error_name(code: int) -> str:
    """Turn a MoveItErrorCodes value into something a log reader can act on."""
    known = {
        MoveItErrorCodes.SUCCESS: "success",
        MoveItErrorCodes.PLANNING_FAILED: "planning failed",
        MoveItErrorCodes.INVALID_MOTION_PLAN: "invalid motion plan",
        MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE:
            "the scene changed under the plan",
        MoveItErrorCodes.CONTROL_FAILED: "the controller did not follow the trajectory",
        MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: "no sensor data",
        MoveItErrorCodes.TIMED_OUT: "timed out",
        MoveItErrorCodes.PREEMPTED: "preempted",
        MoveItErrorCodes.START_STATE_IN_COLLISION: "start state is in collision",
        MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS:
            "start state violates the path constraints",
        MoveItErrorCodes.GOAL_IN_COLLISION: "goal is in collision",
        MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS:
            "goal violates the path constraints",
        MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "goal constraints violated",
        MoveItErrorCodes.INVALID_GROUP_NAME: "no such planning group",
        MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: "invalid goal constraints",
        MoveItErrorCodes.NO_IK_SOLUTION: "no IK solution",
    }
    return known.get(code, "error %d" % code)


class MoveItClient:
    """Synchronous access to move_group, on its own node and thread."""

    def __init__(self, name: str = "avaa_moveit_client", use_sim_time: bool = True):
        # use_global_arguments=False, or this node is not called ``name`` at all.
        #
        # Every node in the launch file is started with --ros-args -r __node:=<name>,
        # and a remap on the command line applies to EVERY node created in the process,
        # not just the first. So this one was renamed to match the controller that owns
        # it: two nodes called avaa_deliver in one process, both with rosout publishers,
        # both being spun. ROS says so at startup and it is easy to read past --
        # "Publisher already registered for provided node name" -- and then the delivery
        # controller died with SIGSEGV in the executor thread the moment it finished
        # connecting to move_group, taking the whole launch down with it.
        self.node = rclpy.create_node(name, use_global_arguments=False)
        self.node.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, use_sim_time)
        ])
        # Every client shares one reentrant group, spun by a multi-threaded executor.
        #
        # rclpy's action client is not thread safe, and this class is called from a
        # thread other than the one spinning it -- that is the whole point of it, since
        # the grasp controller is a timer-driven state machine and cannot block. With a
        # single-threaded executor and the default callback group, sending a goal from
        # the caller's thread while the executor thread is servicing the same client is
        # a race, and it does not fail politely: the node died mid-reach with no Python
        # traceback at all, twice, and once took a segfault while printing one.
        #
        # A reentrant group on a multi-threaded executor is the documented arrangement
        # for exactly this: it lets the goal, the feedback and the result be serviced
        # concurrently instead of contending for one callback slot.
        group = ReentrantCallbackGroup()
        self._group = group
        self.move = ActionClient(self.node, MoveGroup, "move_action",
                                 callback_group=group)
        self.execute = ActionClient(self.node, ExecuteTrajectory,
                                    "execute_trajectory", callback_group=group)
        self.cartesian = self.node.create_client(
            GetCartesianPath, "compute_cartesian_path", callback_group=group)
        self.apply_scene = self.node.create_client(
            ApplyPlanningScene, "apply_planning_scene", callback_group=group)
        self.validity = self.node.create_client(
            GetStateValidity, "check_state_validity", callback_group=group)

        # One caller at a time, so two states can never have goals in flight together.
        self._lock = threading.Lock()
        self.last_failure = ""
        self.last_contacts = []
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ lifecycle

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Whether move_group is up. It takes a few seconds longer than the sim does."""
        return (self.move.wait_for_server(timeout_sec=timeout)
                and self.cartesian.wait_for_service(timeout_sec=timeout)
                and self.apply_scene.wait_for_service(timeout_sec=timeout)
                and self.validity.wait_for_service(timeout_sec=timeout))

    def shutdown(self) -> None:
        self._executor.shutdown()
        self.node.destroy_node()

    # ------------------------------------------------------------------ the scene

    def add_box(self, name: str, frame: str, centre, size,
                orientation: Optional[Quaternion] = None) -> bool:
        """Put a box in the planning scene, replacing any box of the same name."""
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(v) for v in size]

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in centre)
        pose.orientation = orientation or Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        obj = CollisionObject()
        obj.header.frame_id = frame
        obj.id = name
        obj.primitives = [box]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.apply_scene.call_async(request)
        result = self._wait(future, timeout=10.0)
        return bool(result and result.success)

    def remove_object(self, name: str) -> bool:
        obj = CollisionObject()
        obj.id = name
        obj.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        result = self._wait(self.apply_scene.call_async(request), timeout=10.0)
        return bool(result and result.success)

    def state_valid(self, joint_names: Sequence[str], values: Sequence[float],
                    group: str = ARM_GROUP) -> Optional[bool]:
        """Whether a joint configuration is collision free, according to the planner.

        The analytic IK says where the arm reaches; it has no idea what the arm is
        touching when it gets there. On a redundant eight-joint arm most targets have
        many solutions and some of them put the elbow inside the shelf, so the solver
        returns one of those about as often as not and the planner then refuses it as a
        goal -- instantly, and with a generic failure that names nothing.

        Checking here turns "sometimes the grasp will not plan" into "that posture is no
        good, try another one".
        """
        # A complete state, not a diff. Sending only the eight arm joints and letting
        # is_diff fill the rest evaluates the candidate against a robot whose other arm
        # is wherever the default puts it -- straight out -- so every posture comes back
        # in collision. The grasp reported "none of 12 postures is even collision free"
        # for points that are clear, which is the same trap that made reach_fraction
        # return zero for every candidate.
        state = RobotState()
        state.joint_state.name = list(joint_names)
        state.joint_state.position = [float(v) for v in values]
        state.is_diff = False

        request = GetStateValidity.Request()
        request.robot_state = state
        request.group_name = group
        result = self._wait(self.validity.call_async(request), timeout=10.0)
        if result is None:
            self.last_contacts = []
            return None
        # Keep what it collided WITH, not merely that it did.
        #
        # "none of 12 postures is even collision free" is a report with no next step in
        # it. The service already says which two bodies touched and it was being thrown
        # away, so a day went into narrowing down by elimination -- is it the arm, the
        # distance, the shelf, the driving posture -- what one line of the answer would
        # have named. It is nearly always the OTHER arm: left where it spawns,
        # gripper_right_base_link sits at x=+0.86 in base_link, which at a 0.68 m
        # standoff is 0.18 m inside the shelf.
        self.last_contacts = [
            "%s/%s" % (c.contact_body_1, c.contact_body_2)
            for c in getattr(result, "contacts", [])]
        return bool(result.valid)

    def why_invalid(self) -> str:
        """Name what the last state_valid call collided with, for a log line."""
        contacts = getattr(self, "last_contacts", [])
        if not contacts:
            return "no contact reported"
        unique = []
        for pair in contacts:
            if pair not in unique:
                unique.append(pair)
        return ", ".join(unique[:3])

    # ------------------------------------------------------------------ motion

    def move_to_joints(self, joint_names: Sequence[str], values: Sequence[float],
                       group: str = ARM_GROUP, tolerance: float = 0.01,
                       timeout: float = 120.0):
        """Plan and execute to a joint configuration. Returns an error code."""
        constraints = Constraints()
        for name, value in zip(joint_names, values):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(value)
            joint.tolerance_above = tolerance
            joint.tolerance_below = tolerance
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return self._send_move(group, constraints, timeout)

    def move_to_pose(self, position, orientation: Quaternion,
                     group: str = ARM_GROUP, frame: str = PLANNING_FRAME,
                     link: str = TIP_LINK, position_tolerance: float = 0.01,
                     orientation_tolerance: float = 0.1, timeout: float = 120.0):
        """Plan and execute so ``link`` reaches a pose. Returns an error code."""
        constraints = Constraints()

        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [float(position_tolerance)]

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            float(v) for v in position)
        pose.orientation = orientation

        wanted = PositionConstraint()
        wanted.header.frame_id = frame
        wanted.link_name = link
        wanted.constraint_region.primitives = [region]
        wanted.constraint_region.primitive_poses = [pose]
        wanted.weight = 1.0
        constraints.position_constraints = [wanted]

        facing = OrientationConstraint()
        facing.header.frame_id = frame
        facing.link_name = link
        facing.orientation = orientation
        facing.absolute_x_axis_tolerance = orientation_tolerance
        facing.absolute_y_axis_tolerance = orientation_tolerance
        facing.absolute_z_axis_tolerance = orientation_tolerance
        facing.weight = 1.0
        constraints.orientation_constraints = [facing]

        return self._send_move(group, constraints, timeout)

    def reach_fraction(self, waypoints: List[Pose], start_names: Sequence[str],
                       start_values: Sequence[float], group: str = ARM_GROUP,
                       link: str = TIP_LINK, step: float = 0.01) -> float:
        """How much of a straight line would be achievable from a given posture.

        Plans without executing and without the arm having to be there, which makes it a
        test rather than an attempt. The pre-grasp posture is chosen with it: on a
        redundant arm most postures reach the pre-grasp point, and only some of them can
        then travel in a straight line into a 0.33 m shelf opening. Picking one that
        cannot is how a grasp gets 31 per cent of the way in and stops.
        """
        request = GetCartesianPath.Request()
        request.header.frame_id = PLANNING_FRAME
        request.group_name = group
        request.link_name = link
        request.waypoints = waypoints
        request.max_step = step
        request.jump_threshold = 5.0
        request.avoid_collisions = True
        # The complete joint state, not a diff. A partial start_state with is_diff set
        # is silently unusable here: every candidate posture came back 0.0 while the same
        # request without a start_state planned 90 per cent of the way. Whatever the
        # merge is meant to do, it does not do it, and a zero fraction is indistinguishable
        # from a genuinely blocked reach.
        request.start_state.joint_state.name = list(start_names)
        request.start_state.joint_state.position = [float(v) for v in start_values]
        request.start_state.is_diff = False

        result = self._wait(self.cartesian.call_async(request), timeout=30.0)
        if result is None or result.fraction < 0.0:
            return 0.0
        return float(result.fraction)

    def straight_line(self, waypoints: List[Pose], group: str = ARM_GROUP,
                      link: str = TIP_LINK, step: float = 0.01,
                      timeout: float = 120.0):
        """Move the gripper along a straight line, checking collisions as it goes.

        This is the reach into the shelf. A joint-space plan between two points either
        side of a shelf opening bows the arm sideways and the bow goes through the shelf;
        asking for the straight line keeps the hand on the one path that fits.

        Returns (error code, fraction of the path achieved). A fraction below 1.0 means
        the planner ran out of room, which is information: it says the reach is blocked
        rather than that the arm failed to follow it.
        """
        request = GetCartesianPath.Request()
        request.header.frame_id = PLANNING_FRAME
        request.group_name = group
        request.link_name = link
        request.waypoints = waypoints
        request.max_step = step
        # NOT zero, which disables jump checking. On a redundant eight-joint arm the
        # IK solutions along a path can flip the elbow between one waypoint and the
        # next, and MoveIt will happily string those into a trajectory: the reported
        # fraction stays high, the controller executes it, and the gripper arrives
        # 116 mm sideways off a path that only asked to move in x.
        #
        # With a threshold, a discontinuity truncates the path instead. That shows up
        # as a low fraction, which is a fact worth knowing rather than a silent
        # sweep across the front of the shelf.
        request.jump_threshold = 5.0
        request.avoid_collisions = True

        result = self._wait(self.cartesian.call_async(request), timeout=30.0)
        if result is None:
            return MoveItErrorCodes.FAILURE, 0.0
        if result.fraction < 0.0:
            return result.error_code.val, 0.0
        if not result.solution.joint_trajectory.points:
            return result.error_code.val, float(result.fraction)

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = result.solution
        code = self._send_action(self.execute, goal, timeout)
        return code, float(result.fraction)

    def execute_path(self, joint_names: Sequence[str],
                     waypoints: List[Sequence[float]],
                     arm_speed: float = 0.22, torso_speed: float = 0.030,
                     timeout: float = 240.0) -> int:
        """Execute a joint-space path the caller has already worked out.

        MoveIt still does the execution -- splitting the trajectory across the arm and
        torso controllers, and monitoring it -- but not the kinematics. Its Cartesian
        planner solves IK with KDL at every step, and on this redundant eight-joint chain
        KDL fails: a reach whose every waypoint checks out as collision free, walked and
        verified one by one against /check_state_validity, came back from
        compute_cartesian_path as 2 per cent achievable.

        Timed from what the joints can do rather than from what they are rated for, and
        that had to be measured again. The old figures -- 0.3 rad/s for the arm, 0.035
        m/s for the torso, against a rated 1.95 to 3.95 -- were taken when
        position_proportional_gain was still 0.1 and the arm could not follow anything
        it was told. Re-measured with the gain fixed, a 0.9 rad move asked for in one
        second settles in 1.6, which is 0.55 rad/s. Asking for more than that does not
        get more, it gets overshoot: at 0.70 rad/s with a 0.15 s floor between
        waypoints, arm_left_1 was asked for 2.318 rad and went to 4.101.

        But speed was the wrong thing to optimise, and 0.45 was a mistake made for a
        reason that turned out to be backwards. The reach was sped up to spend less time
        exposed to base drift. The base drift is caused by the arm: tucked and idle the
        base turns 0.05 deg/s, and after two swings of the arm it is turning 1.22 and
        has moved 211 mm. The disturbance goes with how hard the arm accelerates, not
        with how long it takes, so hurrying makes the very thing it was meant to escape.

        It shows up as a loop that cannot converge. With the base measurably still --
        the book holding to 1 mm through the camera -- a reach finished 112 mm from its
        target, because the target had moved that far while the arm was travelling. The
        correction for that is another arm motion, which moves the base again by about
        as much as it corrects.

        So: gently. 0.22 rad/s and a half second between waypoints.

        Speed matters here for a reason that has nothing to do with impatience. The base
        slides continuously -- the wheels have no friction across the roller axis -- at
        around 3.4 mm/s, and it takes the target with it. A reach that takes 34 seconds
        is a reach whose aim is 115 mm stale by the time it arrives. Halving the time
        halves that error, and it is the cheapest halving available.
        """
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = list(joint_names)
        moment = 0.0
        previous = None
        for values in waypoints:
            if previous is not None:
                arm_move = max(
                    (abs(a - b) for name, a, b in zip(joint_names, values, previous)
                     if name != "torso_lift_joint"), default=0.0)
                torso_move = max(
                    (abs(a - b) for name, a, b in zip(joint_names, values, previous)
                     if name == "torso_lift_joint"), default=0.0)
                # Every waypoint asks for zero velocity, so a short segment is a demand
                # for hard acceleration, and hard acceleration is exactly what shoves the
                # base. Half a second is deliberately unhurried.
                moment += max(0.50, arm_move / arm_speed, torso_move / torso_speed)
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in values]
            point.velocities = [0.0] * len(values)
            point.time_from_start = Duration(
                sec=int(moment), nanosec=int((moment % 1.0) * 1e9))
            trajectory.joint_trajectory.points.append(point)
            previous = values

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        return self._send_action(self.execute, goal, timeout)

    # ------------------------------------------------------------------ plumbing

    def _send_move(self, group: str, constraints: Constraints, timeout: float):
        goal = MoveGroup.Goal()
        goal.request.group_name = group
        goal.request.goal_constraints = [constraints]
        goal.request.num_planning_attempts = 10
        # Five seconds was not enough for the harder rows, where the torso has to
        # travel most of its range while the arm unfolds past a shelf.
        goal.request.allowed_planning_time = 15.0
        # Scaled down hard. This arm covers about 3 rad in 28 s and its controller aborts
        # trajectories it cannot keep up with, which presents as an arm that simply does
        # not move.
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        return self._send_action(self.move, goal, timeout)

    def _send_action(self, client: ActionClient, goal, timeout: float) -> int:
        with self._lock:
            return self._send_action_locked(client, goal, timeout)

    def _send_action_locked(self, client: ActionClient, goal, timeout: float) -> int:
        """Send a goal and wait for its result, saying which step failed if one does.

        Returning a bare FAILURE for every problem here made three different faults look
        identical -- server missing, goal rejected, plan failed -- which is exactly the
        kind of silence this project has lost the most time to.
        """
        self.last_failure = ""
        if not client.wait_for_server(timeout_sec=20.0):
            self.last_failure = "action server never appeared"
            return MoveItErrorCodes.FAILURE
        # Accepting a goal is cheap, but move_group only gets to it when it is next
        # scheduled, and it is competing with Gazebo, the controllers and perception on
        # a machine already running the simulation below real time. Fifteen seconds was
        # enough until the base holder was added and then a run failed with "no reply to
        # the goal request" while the arm was visibly sitting at the pre-grasp it had
        # just been asked for. This is a queueing delay, not a planning failure, so it
        # deserves a timeout that reflects the load rather than the work.
        handle = self._wait(client.send_goal_async(goal), timeout=45.0)
        if handle is None:
            self.last_failure = "no reply to the goal request"
            return MoveItErrorCodes.FAILURE
        if not handle.accepted:
            self.last_failure = "goal rejected by move_group"
            return MoveItErrorCodes.FAILURE
        wrapper = self._wait(handle.get_result_async(), timeout=timeout)
        if wrapper is None:
            self.last_failure = "no result within %.0f s" % timeout
            return MoveItErrorCodes.TIMED_OUT
        result = getattr(wrapper, "result", None)
        if result is None:
            self.last_failure = "result message had no payload"
            return MoveItErrorCodes.FAILURE
        code = getattr(result, "error_code", None)
        if code is None:
            self.last_failure = "result payload had no error_code: %s" % type(result)
            return MoveItErrorCodes.FAILURE
        return int(code.val)

    @staticmethod
    def _wait(future, timeout: float):
        """Wait on a future that another thread is spinning."""
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if future.done():
                try:
                    return future.result()
                except Exception:  # noqa: BLE001 - a failed call is not a crash
                    return None
            time.sleep(0.02)
        return None
