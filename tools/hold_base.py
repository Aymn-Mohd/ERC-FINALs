#!/usr/bin/env python3
"""Hold the base still while the arm works.

The base does not stay where it is put, and not because anything is driving it. The
wheels are mecanum, and they are modelled the way mecanum wheels have to be: mu 0.8
along the roller axis and mu2 0.0 across it. With no friction at all in one direction
there is nothing to damp a sideways drift, so once the base has any lateral motion it
keeps it. Measured on a freshly spawned robot standing untouched with the arm still, it
travels 8 mm and turns 2 degrees every 30 seconds. Measured after tools/place_robot.py
teleports it, 140 mm and 21 degrees every 30 seconds, and that does not decay.

Commanding zero velocity does not help: zero cmd_vel locks the wheels, and the base
simply slides across the locked wheels in the direction that has no friction.

So this closes the loop. It is a fixture, not part of the solution: it reads the true
pose from the simulator, which the robot itself cannot do. The solution needs the same
station keeping driven by perception instead -- the point here is only to stop the test
rig from generating a drift ten times larger than the real one and then blaming the
grasp for missing.

    python3 hold_base.py [seconds] [x y yaw_deg]

Given a goal it drives to it and holds there; given none it holds wherever it starts.
Pass the goal when the base has to be square to the shelf: it drifts about 2 degrees in
the few seconds between tools/place_robot.py teleporting it and this latching, which is
enough to matter at arm's length.

Measured holding: 3 mm and 0.7 degrees, against 140 mm and 21 degrees uncontrolled.
"""
import math
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist

# Reading the true pose costs about a second, so the loop runs near 0.8 Hz and the
# gains have to suit that: at gain 2.0 a 10 degree error commanded 0.16 rad/s for a
# whole 1.25 s step, which is 11 degrees of correction for a 10 degree error, and the
# base sat there oscillating. Keep the product of gain and step below one.
GAIN_LINEAR = 0.5
GAIN_ANGULAR = 0.6
# Damping on measured velocity, not just error on position.
#
# A position controller on a loop this slow can only choose which way to be wrong: a
# tight deadband makes it oscillate, a loose one lets it drift, and both show up at the
# gripper as the target moving. What actually matters is the rate -- the jaws have about
# 14 mm of clearance either side of the book, and it is the millimetres per second
# between re-aiming and closing that spend it. Opposing the measured velocity attacks
# that directly, and unlike the position term it does not need a setpoint to chase.
# Off. Differencing the pose over a 1.5 s loop and feeding that back as a velocity
# command is derivative feedback through a long delay, and it did exactly what that
# predicts: the base oscillated out to 503 mm and plus or minus 21 degrees, far worse
# than the drift it was meant to remove. Kept at zero rather than deleted because the
# gating below is the part worth keeping if it is ever tried again with a faster loop.
DAMP_LINEAR = 0.0
DAMP_ANGULAR = 0.0
# Below this the base is where it is wanted; chasing further only adds motion.
# Deliberately loose. A tight deadband makes this fight every millimetre, and with a
# loop running near 0.7 Hz that fight is an oscillation: measured at the jaws during a
# reach, the base was swinging the target through 13 to 52 mm sideways while the log
# reported it "held" to 34 mm. The grasp re-aims from perception and can absorb a
# steady offset; what it cannot absorb is the floor moving under it. So this now only
# intervenes when the base is genuinely running away.
# Tighter again now the wheels have friction across the roller axis. The loose
# deadband existed to stop the holder fighting a base that slid out from under it
# whatever it did; with the sliding gone it can afford to hold properly.
# Wide enough that it is not constantly nudging. Below about 40 mm of error the
# proportional output falls under what the drive will actually move for -- commanded
# 0.02 m/s the base simply sits there -- so chasing smaller errors just adds motion
# without removing any. A steady offset costs nothing anyway: perception measures the
# book through a camera on the base, so a base parked 40 mm off still aims correctly.
# What has to be small is the movement between re-aiming and closing.
# Only step in when the base has genuinely wandered. The grasp now waits for the book
# to stop moving in the camera before it closes the jaws, and a holder that is forever
# nudging is a base that is never quiet -- it would starve the grasp of the stillness it
# is waiting for. A steady offset costs nothing, because perception re-aims through it.
DEADBAND_LINEAR = 0.060
DEADBAND_ANGULAR = 0.050
# Slow enough that a correction never yanks a book out of the fingers.
MAX_LINEAR = 0.15
MAX_ANGULAR = 0.4


def gz(*args, timeout=25):
    return subprocess.run(["gz", *args], capture_output=True, text=True,
                          timeout=timeout).stdout


def pose_retry(model="tiago_pro", attempts=12):
    """Gazebo's query service drops requests when it is busy, so ask again.

    Under the load of a grasp run -- Gazebo below real time, move_group planning,
    perception and this holder all on the same machine -- a single `gz model -p` fails
    often enough that reading the pose once and giving up killed the holder on startup
    and left the base free to slide through the whole run.
    """
    for _ in range(attempts):
        here, rpy = pose(model)
        if here is not None:
            return here, rpy
        time.sleep(0.3)
    return None, None


def pose(model="tiago_pro"):
    lines = [l.strip() for l in gz("model", "-m", model, "-p").splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
            try:
                return ([float(v) for v in line.strip("[]").split()],
                        [float(v) for v in lines[i + 1].strip("[]").split()])
            except ValueError:
                return None, None
    return None, None


def wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def clamp(value, limit):
    return max(-limit, min(limit, value))


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
    wanted = None
    if len(sys.argv) > 4:
        wanted = (float(sys.argv[2]), float(sys.argv[3]),
                  math.radians(float(sys.argv[4])))

    rclpy.init()
    node = rclpy.create_node("hold_base")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    waited = time.time()
    while time.time() - waited < 3:
        rclpy.spin_once(node, timeout_sec=0.1)

    start, start_rpy = pose_retry()
    if start is None:
        print("no robot pose; is Gazebo up?")
        return
    goal_x, goal_y, goal_yaw = wanted or (start[0], start[1], start_rpy[2])
    print("holding x=%.3f y=%.3f yaw=%.2f deg for %.0f s"
          % (goal_x, goal_y, math.degrees(goal_yaw), seconds))
    sys.stdout.flush()

    began = time.time()
    previous = None
    reported = [time.time()]
    worst_linear = worst_angular = 0.0
    while time.time() - began < seconds:
        here, rpy = pose()
        if here is None:
            continue
        error_x, error_y = goal_x - here[0], goal_y - here[1]
        error_yaw = wrap(goal_yaw - rpy[2])
        worst_linear = max(worst_linear, math.hypot(error_x, error_y))
        worst_angular = max(worst_angular, abs(error_yaw))

        # The error is in the world; cmd_vel is in the base.
        yaw = rpy[2]
        forward = error_x * math.cos(-yaw) - error_y * math.sin(-yaw)
        sideways = error_x * math.sin(-yaw) + error_y * math.cos(-yaw)

        command = Twist()
        if not rclpy.ok():
            break

        # Straight proportional control on all three, which works again now the wheels
        # have a little friction across the roller axis. At mu2 exactly zero the base
        # slid out from under any correction; at 0.80 it could drive forwards but could
        # not turn at all -- commanded 0.20 rad/s it managed 1 degree in six seconds --
        # and the holder walked away from its goal. At 0.12 all four motions work:
        # forward, backward, strafe and rotate.
        if math.hypot(error_x, error_y) > DEADBAND_LINEAR:
            command.linear.x = clamp(GAIN_LINEAR * forward, MAX_LINEAR)
            command.linear.y = clamp(GAIN_LINEAR * sideways, MAX_LINEAR)
        if abs(error_yaw) > DEADBAND_ANGULAR:
            command.angular.z = clamp(GAIN_ANGULAR * error_yaw, MAX_ANGULAR)

        pub.publish(command)
        rclpy.spin_once(node, timeout_sec=0.01)
        # Reading the true pose is a Gazebo service call, and calling it flat out keeps
        # Gazebo's service thread busy enough that move_group misses its own deadlines.
        # The drift is millimetres per second; it does not need chasing any harder.
        # Each pose read spawns a process and hits Gazebo's own service thread. With
        # this holder, the fixture and a monitor all polling at once the simulation fell
        # from 0.65 to 0.008 of real time and nothing moved at all. The drift is
        # millimetres per second; once every second and a half is plenty.
        time.sleep(1.5)

        if time.time() - reported[0] > 15.0:
            reported[0] = time.time()
            print("  held to %5.1f mm, %+5.2f deg"
                  % (math.hypot(error_x, error_y) * 1000, math.degrees(error_yaw)))
            sys.stdout.flush()

    try:
        pub.publish(Twist())
    except Exception:  # noqa: BLE001 - shutting down mid-publish is not a failure
        pass
    print("worst error while holding: %.1f mm, %.2f deg"
          % (worst_linear * 1000, math.degrees(worst_angular)))
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
