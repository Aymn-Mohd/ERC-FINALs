#!/usr/bin/env python3
"""Can the arm pick up a book when the target is perfect?

    python3 ideal_grasp.py <book_colour>

Perception is now close: measured against Gazebo the robot ends square, the row is right
and the grasp point lands inside the book. The book still does not move. That leaves two
very different causes -- a targeting error of a few centimetres, or an arm that never
does what it is told -- and the fix for one is nothing like the fix for the other.

So this removes perception from the loop. It reads the target book straight out of
Gazebo, publishes that as the row and the book point, and runs the real grasp controller
against it while watching the arm. If the book moves, the mechanics are sound and the
remaining work is in perception. If it does not, the mechanics are the problem and no
amount of perception accuracy would have helped.

Run it with the robot already standing in front of the shelf.
"""
import math
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, String

sys.path.insert(0, "/opt/erc_ws/src/avaa_solution")
from avaa_solution.kinematics.arm_chain import ArmChain  # noqa: E402

ARM_JOINTS = ["arm_left_%d_joint" % i for i in range(1, 8)]
CHAIN_JOINTS = ["torso_lift_joint"] + ARM_JOINTS
FINGER = "gripper_left_finger_joint"
BASE_Z = 0.186          # base_link sits this far above the world origin
BOOK_HALF_DEPTH = 0.08  # book centre to its front face

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


def gz_pose(model):
    out = subprocess.run(["gz", "model", "-m", model, "-p"],
                         capture_output=True, text=True, timeout=25).stdout
    lines = [l.strip() for l in out.splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
            return ([float(v) for v in line.strip("[]").split()],
                    [float(v) for v in lines[i + 1].strip("[]").split()])
    return None, None


def gz_pose_retry(model, attempts=4):
    """gz declines a pose query often enough that one refusal must not end the run."""
    for _ in range(attempts):
        p, r = gz_pose(model)
        if p is not None:
            return p, r
        time.sleep(1.0)
    return None, None


def nearest_book(colour):
    # Ask more than once. Gazebo drops query requests when it is busy, and a single
    # dropped listing here reports "no red book found" and throws away a run that was
    # otherwise ready to go -- twice in one session.
    names = []
    for _ in range(8):
        try:
            listing = subprocess.run(["gz", "model", "--list"], capture_output=True,
                                     text=True, timeout=25).stdout
        except Exception:  # noqa: BLE001
            listing = ""
        # ``colour`` may be a full model name instead, so a specific row can be targeted.
        names = [l.strip(" -") for l in listing.splitlines()
                 if "book_col" in l and (colour in l or colour == l.strip(" -"))]
        if names:
            break
        time.sleep(0.5)
    base, _ = gz_pose_retry("tiago_pro")
    if base is None:
        return None
    best = None
    for name in names:
        p, r = gz_pose_retry(name, attempts=4)
        if p is None:
            continue
        d = math.hypot(p[0] - base[0], p[1] - base[1])
        if best is None or d < best[0]:
            best = (d, name, p, r)
    return best


TUCK_POSE = [2.1521, 0.3824, 1.2785, -2.1517, 0.8325, 0.1926, 1.3944]
# The right arm is never used and it still has to be out of the shelf. Left at the pose
# it spawns in, gripper_right_base_link sits at x=+0.86 in base_link, which at a 0.68 m
# standoff is 0.18 m inside the shelf: every pre-grasp posture then comes back in
# collision -- arm_right_4/5/6/7 against shelf_board_2 -- and the grasp fails before it
# has moved. Tucked, the same link sits at x=-0.52, behind the robot.
RIGHT_TUCK = [-0.7194, -2.2867, -0.5064, 0.5221, 2.3399, 1.0503, 1.9772]
TUCK_TORSO = 0.15


def tuck_the_arms(node):
    """Fold the arm in before starting, the way the real approach does.

    Without this the fixture starts from whatever pose the robot spawned in, which is not
    the pose the grasp controller ever sees in a real run. Started from there, the arm
    jammed against the shelf on its very first move and the fixture was measuring its own
    setup rather than the grasp.
    """
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration

    pubs = {
        side: node.create_publisher(
            JointTrajectory, "/arm_%s_controller/joint_trajectory" % side, 10)
        for side in ("left", "right")
    }
    torso_pub = node.create_publisher(
        JointTrajectory, "/torso_controller/joint_trajectory", 10)
    deadline = time.time() + 10
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if all(p.get_subscription_count() > 0 for p in pubs.values()):
            break

    torso = JointTrajectory()
    torso.joint_names = ["torso_lift_joint"]
    lift = JointTrajectoryPoint()
    lift.positions = [float(TUCK_TORSO)]
    lift.time_from_start = Duration(sec=20, nanosec=0)
    torso.points = [lift]
    torso_pub.publish(torso)

    for side, pub in pubs.items():
        traj = JointTrajectory()
        traj.joint_names = ["arm_%s_%d_joint" % (side, i) for i in range(1, 8)]
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in
                           (RIGHT_TUCK if side == "right" else TUCK_POSE)]
        point.time_from_start = Duration(sec=20, nanosec=0)
        traj.points = [point]
        pub.publish(traj)
    print("tucking both arms before the grasp...", flush=True)

    # The simulation runs below real time, so a 20 s trajectory needs more than 20 s of
    # wall clock to finish. At the measured 0.7 real time factor this is about 32 s of
    # simulated time, which is enough with room to spare.
    end = time.time() + 46
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)


def run_grasp(depth=None):
    """Start the real grasp controller, optionally overriding how deep it reaches.

    The jaw pads extend well forward of the fingertip link origins, so the depth that
    puts them around a book rather than against its face is not something to derive from
    mesh bounds. Sweeping it is quicker and the answer is measured rather than argued.
    """
    extra = "" if depth is None else " -p grasp_depth_m:=%f" % depth
    cmd = ("source /opt/erc_ws/install/setup.bash && "
           "exec python3 -u /opt/erc_ws/install/avaa_solution/lib/avaa_solution/"
           "grasp --ros-args -p use_sim_time:=true" + extra +
           " > /tmp/ideal_grasp.log 2>&1")
    return subprocess.Popen(["/entrypoint.sh", "bash", "-c", cmd])


def main():
    colour = sys.argv[1] if len(sys.argv) > 1 else "red"
    depth = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    chain = ArmChain.from_urdf()
    print("grasp depth: %s" % ("default" if depth is None else "%.3f m" % depth))

    found = nearest_book(colour)
    if found is None:
        print("no %s book found" % colour)
        return
    distance, name, truth, truth_rpy = found
    robot, rpy = gz_pose_retry("tiago_pro")
    yaw = rpy[2]

    # The book in base_link, from ground truth: rotate the world offset by -yaw.
    dx, dy = truth[0] - robot[0], truth[1] - robot[1]
    bx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    by = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    face_x = bx - BOOK_HALF_DEPTH
    bz = truth[2] - BASE_Z

    # Which competition row that height is, against the heights the grasp node uses.
    heights = [1.391, 1.061, 0.731, 0.401]
    row = min(range(4), key=lambda i: abs(heights[i] - bz)) + 1

    print("target      : %s at %.2f m" % (name, distance))
    print("robot yaw   : %+.1f deg" % math.degrees(yaw))
    print("book in base: face x=%.3f  y=%+.3f  z=%.3f" % (face_x, by, bz))
    print("row          : %d (nominal z=%.3f, truth z=%.3f, error %+.3f m)"
          % (row, heights[row - 1], bz, bz - heights[row - 1]))
    print()

    rclpy.init()
    node = rclpy.create_node("ideal_grasp")
    pub_row = node.create_publisher(Int32, "/avaa/perception/target_row", 10)
    pub_point = node.create_publisher(
        PointStamped, "/avaa/perception/target_book_point", 10)
    state = {"joints": None, "grasp": "?"}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: state.__setitem__("joints", m), SENSOR_QOS)
    node.create_subscription(String, "/avaa/grasp/state",
                             lambda m: state.__setitem__("grasp", m.data), 10)

    tuck_the_arms(node)

    # Take the "before" pose after the tuck, not before it.
    #
    # The tuck is the fixture's own arm motion, and it is not gentle: starting from a
    # failed grasp it sweeps the arm out of the shelf, and on one run that knocked the
    # target book onto the floor. The run then measured the book against a pose recorded
    # before any of that happened, found it had not moved since, and reported NOT MOVED
    # for a book that was never on the shelf to begin with.
    settled, settled_rpy = gz_pose_retry(name)
    if settled is not None and settled_rpy is not None:
        shifted = math.dist(truth[:3], settled[:3])
        if shifted > 0.02:
            print("WARNING: %s moved %.3f m during the tuck, before the grasp started"
                  % (name, shifted))
        if abs(settled[2] - truth[2]) > 0.05:
            print("WARNING: %s is not at shelf height any more (z=%.3f); "
                  "this run cannot mean anything" % (name, settled[2]))
        truth, truth_rpy = settled, settled_rpy

    grasp = run_grasp(depth)
    time.sleep(4)

    refreshed = [0.0]
    last_fk = [None]
    last_finger = [None]
    print("%-11s %6s  %-40s  %8s %8s %8s  %6s" %
          ("state", "torso", "arm 1..7", "fk x", "fk y", "fk z", "finger"))
    seen = set()
    start = time.time()
    try:
        # A whole grasp is scene, posture search, raise, pre-grasp, open, advance,
        # clamp, lift, withdraw and stow, each of them a planned and executed
        # trajectory, on a simulation running below real time. Measured, it reaches
        # "advancing" around 130 s in. At 170 s this fixture was killing the controller
        # part way into the shelf and then reporting that the book had not moved, which
        # is true and says nothing about the grasp.
        while time.time() - start < run_seconds:
            # Recompute from live ground truth, the way perception would.
            #
            # Taking one snapshot at startup and publishing it forever is not what the
            # real pipeline does, and it is not harmless: the base keeps moving after it
            # is placed -- 2.5 degrees of yaw immediately, and around 7 by the time the
            # arm is out -- so a target fixed in base_link at startup is 85 mm from the
            # book by the time the gripper gets there. That is the fixture being wrong,
            # not the grasp, and it was being read as the grasp being wrong.
            if time.time() - refreshed[0] > 4.0:
                refreshed[0] = time.time()
                live, live_rpy = gz_pose_retry(name, attempts=1)
                base, base_rpy = gz_pose_retry("tiago_pro", attempts=1)
                if live is not None and base is not None:
                    yaw_now = base_rpy[2]
                    dx, dy = live[0] - base[0], live[1] - base[1]
                    face_x = (dx * math.cos(-yaw_now) - dy * math.sin(-yaw_now)
                              - BOOK_HALF_DEPTH)
                    by = dx * math.sin(-yaw_now) + dy * math.cos(-yaw_now)
                    bz = live[2] - BASE_Z

            msg = PointStamped()
            msg.header.frame_id = "base_link"
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.point.x, msg.point.y, msg.point.z = face_x, by, bz
            pub_point.publish(msg)
            pub_row.publish(Int32(data=row))

            rclpy.spin_once(node, timeout_sec=0.2)
            js = state["joints"]
            if js is not None:
                index = {n: i for i, n in enumerate(js.name)}
                if all(n in index for n in CHAIN_JOINTS):
                    actual = [js.position[index[n]] for n in CHAIN_JOINTS]
                    finger = js.position[index[FINGER]] if FINGER in index else float("nan")
                    fk = chain.fk(actual)[:3, 3]
                    last_fk[0] = fk
                    last_finger[0] = finger
                    now = state["grasp"]
                    stamp = "%.0f" % (time.time() - start)
                    key = (now, stamp)
                    if now not in seen or key not in seen:
                        print("%-11s %6.3f  %-40s  %8.3f %8.3f %8.3f  %6.4f" %
                              (now, actual[0],
                               " ".join("%+.2f" % v for v in actual[1:]),
                               fk[0], fk[1], fk[2], finger), flush=True)
                        seen.add(now)
                        seen.add(key)
            if state["grasp"] in ("done", "failed"):
                break
            time.sleep(1.0)
    finally:
        grasp.terminate()
        subprocess.run(["pkill", "-f", "avaa_solution/lib"], capture_output=True)
        node.destroy_node()
        rclpy.shutdown()

    time.sleep(2)
    after, after_rpy = gz_pose_retry(name)
    if after is None or after_rpy is None:
        print()
        print("=== judged against Gazebo ===")
        print("could not read %s back; no verdict" % name)
        return
    moved = math.dist(truth[:3], after[:3])
    tipped = max(abs(a - b) for a, b in zip(truth_rpy[:2], after_rpy[:2]))
    print()
    print("=== judged against Gazebo ===")
    print("%s moved %.3f m, tipped %.2f rad" % (name, moved, tipped))
    # Displacement is not a pick, and neither is displacement plus staying upright.
    #
    # This fixture has now called two different failures a success. A book swept onto
    # its side travelled 0.14 m with the fingers closed on air. Then a book shoved
    # 0.085 m deeper into the shelf, still sitting at shelf height with the gripper
    # fully closed and empty, was reported as PICKED UP because it had moved far enough
    # and had not tipped far enough. Both were penalties in the competition and both
    # read as wins here.
    #
    # A pick means the robot ended up holding the book. The only honest test of that is
    # where the book finished relative to the hand, so that is what decides it.
    # Where the gripper finished, in the world. ROS is shut down by now, so this uses
    # the last pose the monitor computed rather than asking TF again.
    held = None
    robot, robot_rpy = gz_pose_retry("tiago_pro")
    if last_fk[0] is not None and robot is not None:
        yaw = robot_rpy[2]
        gx, gy, gz_ = last_fk[0]
        hand = (robot[0] + gx * math.cos(yaw) - gy * math.sin(yaw),
                robot[1] + gx * math.sin(yaw) + gy * math.cos(yaw),
                gz_ + BASE_Z)
        held = math.dist(hand, after[:3])
        print("book finished %.3f m from the gripper, and %+.3f m in height"
              % (held, after[2] - truth[2]))

    # Near the hand is not in the hand. A run finished with the book 56 mm from the
    # gripper and was called a pick, while the finger sat at 0.067 -- wide open -- and
    # the book had not risen a millimetre. It had simply been knocked over and landed
    # next to an open gripper. So the jaws have to be closed on something of about the
    # right thickness as well: fully shut is air, fully open is nothing held.
    finger = last_finger[0]
    closed_on_something = finger is not None and 0.004 < finger < 0.055
    if held is not None and held < 0.12 and closed_on_something:
        print("RESULT: PICKED UP — the book finished in the hand, jaws at %.4f"
              % finger)
    elif held is not None and held < 0.12:
        print("RESULT: NOT HELD — the book ended up beside the gripper, jaws at %s"
              % ("unknown" if finger is None else "%.4f" % finger))
    elif moved <= 0.02:
        print("RESULT: NOT MOVED")
    elif tipped > 0.35:
        print("RESULT: KNOCKED OVER — swept, not grasped")
    else:
        print("RESULT: DISTURBED — moved %.3f m but left behind, not carried" % moved)


if __name__ == "__main__":
    main()
