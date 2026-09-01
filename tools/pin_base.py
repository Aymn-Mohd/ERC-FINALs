#!/usr/bin/env python3
"""Pin the base in place, so a grasp can be tested without the floor moving under it.

tools/hold_base.py drives the wheels to hold station and that is the right shape of
answer for a real robot, but it cannot win here. The wheels are mecanum, modelled with
mu 0.8 along the roller axis and mu2 0.0 across it, so there is no friction at all in one
direction. With the arm tucked the holder keeps the base inside 20 mm. With the arm
extended into the shelf the moment of the reaching arm beats it: measured through one
grasp, the error grew from 19 mm to 263 mm and from 2 to 25 degrees while the holder was
commanding full correction the whole way.

So this pins the base by teleport instead. It is unashamedly a fixture -- a real robot
cannot teleport -- and its only job is to take base drift out of the experiment so that
the grasp itself can be measured. What it costs is one Gazebo service call per second,
and what it buys is the difference between testing the grasp and testing the floor.

    python3 pin_base.py <seconds> <x> <y> <yaw_deg>
"""
import math
import subprocess
import sys
import time

TOLERANCE_M = 0.004
TOLERANCE_RAD = 0.004


def gz(*args, timeout=20):
    try:
        return subprocess.run(["gz", *args], capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return ""


def pose(model="tiago_pro", attempts=6):
    for _ in range(attempts):
        lines = [l.strip() for l in gz("model", "-m", model, "-p").splitlines()]
        for i, line in enumerate(lines):
            if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
                try:
                    return ([float(v) for v in line.strip("[]").split()],
                            [float(v) for v in lines[i + 1].strip("[]").split()])
                except ValueError:
                    pass
        time.sleep(0.25)
    return None, None


def put(x, y, yaw):
    return gz("service", "-s", "/world/erc_world/set_pose",
              "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
              "--timeout", "3000",
              "--req", 'name: "tiago_pro", position: {x: %f, y: %f, z: 0.0}, '
                       'orientation: {x: 0, y: 0, z: %f, w: %f}'
                       % (x, y, math.sin(yaw / 2.0), math.cos(yaw / 2.0)))


def wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
    if len(sys.argv) > 4:
        gx, gy, gyaw = float(sys.argv[2]), float(sys.argv[3]), math.radians(float(sys.argv[4]))
    else:
        here, rpy = pose()
        if here is None:
            print("no robot pose; is Gazebo up?")
            return
        gx, gy, gyaw = here[0], here[1], rpy[2]

    print("pinning base at x=%.3f y=%.3f yaw=%.2f deg for %.0f s"
          % (gx, gy, math.degrees(gyaw), seconds))
    sys.stdout.flush()

    began = time.time()
    corrections = 0
    worst = 0.0
    reported = time.time()
    while time.time() - began < seconds:
        here, rpy = pose(attempts=2)
        if here is None:
            time.sleep(0.5)
            continue
        drift = math.hypot(gx - here[0], gy - here[1])
        turn = abs(wrap(gyaw - rpy[2]))
        worst = max(worst, drift)
        if drift > TOLERANCE_M or turn > TOLERANCE_RAD:
            put(gx, gy, gyaw)
            corrections += 1
        if time.time() - reported > 20.0:
            reported = time.time()
            print("  worst drift between corrections %.1f mm, %d corrections so far"
                  % (worst * 1000, corrections))
            sys.stdout.flush()
            worst = 0.0
        time.sleep(1.0)

    print("done: %d corrections" % corrections)


if __name__ == "__main__":
    main()
