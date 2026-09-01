#!/usr/bin/env python3
"""At what distance from the shelf can the arm actually complete a grasp?

The arm reaches its planned posture in open floor and not at the shelf, yet no joint
origin in that posture goes past the shelf face. So the obstruction is not the elbow
going through the front of the unit; it is more likely that at 0.90 m the arm is close to
fully extended and swings wide on the way, and there is a divider standing about 0.05 m
proud of the book faces for it to catch.

This places the base at a series of distances and runs the real grasp controller against a
target taken from Gazebo, so the only thing changing is how far back the robot stands.
"""
import subprocess
import sys
import time

STANDOFFS = [float(v) for v in sys.argv[1:]] or [0.90, 0.78, 0.66]


def sh(command, timeout=400):
    return subprocess.run(["bash", "-lc", command], capture_output=True,
                          text=True, timeout=timeout).stdout


def main():
    results = []
    for standoff in STANDOFFS:
        print("=" * 60)
        print("standoff = %.2f m" % standoff, flush=True)
        sh("sim restart --fast --headless >/dev/null 2>&1", timeout=300)
        time.sleep(26)

        # Tuck BEFORE placing. The robot spawns with its arms straight out -- all
        # joints at zero puts the gripper at (0.984, 0.493, 0.229) -- so a base
        # teleported to grasping distance plants the arm inside the shelf, and every
        # command after that jams before it can start.
        print(sh("docker exec erc_sim /entrypoint.sh bash -c "
                 "'source /opt/erc_ws/install/setup.bash && "
                 "python3 -u /tmp/tuck_arm.py' 2>&1 | tail -2",
                 timeout=120).strip(), flush=True)
        placed = sh("docker exec erc_sim /entrypoint.sh python3 /tmp/place_robot.py "
                    "red %f 2>&1 | tail -2" % standoff)
        print(placed.strip(), flush=True)

        sh("docker exec erc_sim /entrypoint.sh bash -c "
           "'source /opt/erc_ws/install/setup.bash && timeout 400 python3 -u "
           "/tmp/ideal_grasp.py red > /tmp/ideal.log 2>&1'", timeout=400)

        log = sh("docker exec erc_sim bash -c 'strings /tmp/ideal.log'")
        verdict, closest = "no verdict", ""
        for line in log.splitlines():
            if line.startswith("RESULT:"):
                verdict = line.strip()
        grasp = sh("docker exec erc_sim bash -c 'strings /tmp/ideal_grasp.log'")
        gaps = [int(m) for m in
                __import__("re").findall(r"(\d+) mm to go", grasp)]
        if gaps:
            closest = "closest approach %d mm (from %d)" % (min(gaps), gaps[0])
        print("   %s" % closest, flush=True)
        print("   %s" % verdict, flush=True)
        results.append((standoff, closest, verdict))

    print()
    print("=" * 60)
    print("%-10s %-32s %s" % ("standoff", "closest approach", "outcome"))
    for standoff, closest, verdict in results:
        print("%-10.2f %-32s %s" % (standoff, closest, verdict))


if __name__ == "__main__":
    main()
