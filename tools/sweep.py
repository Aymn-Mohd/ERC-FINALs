#!/usr/bin/env python3
"""Find the grasp depth that puts the jaws around a book instead of against it.

Runs on the WSL host: restarts the simulator, places the robot squarely in front of a
book, and runs the real grasp controller at each depth with a target taken from Gazebo,
so the only thing varying is the depth.
"""
import re
import subprocess
import sys
import time

# Taken from the command line so a sweep can be split across several invocations;
# a full one runs longer than a single tool call is allowed to last.
DEPTHS = [float(v) for v in sys.argv[1:]] or [0.05, 0.09, 0.13, 0.17]


def sh(command, timeout=300):
    return subprocess.run(["bash", "-lc", command], capture_output=True,
                          text=True, timeout=timeout).stdout


def main():
    results = []
    for depth in DEPTHS:
        print("=" * 60)
        print("grasp_depth = %.3f m" % depth, flush=True)
        sh("sim restart --fast --headless >/dev/null 2>&1", timeout=300)
        time.sleep(26)

        placed = sh("docker exec erc_sim /entrypoint.sh python3 /tmp/place_robot.py "
                    "red 0.90 2>&1 | tail -2")
        print(placed.strip(), flush=True)

        sh("docker exec erc_sim /entrypoint.sh bash -c "
           "'source /opt/erc_ws/install/setup.bash && timeout 220 python3 -u "
           "/tmp/ideal_grasp.py red %f > /tmp/ideal.log 2>&1'" % depth, timeout=400)

        log = sh("docker exec erc_sim bash -c 'strings /tmp/ideal.log'")
        verdict = "no verdict"
        for line in log.splitlines():
            if line.startswith("RESULT:"):
                verdict = line.strip()
            if "moved" in line and "tipped" in line:
                verdict = line.strip() + " | " + verdict
        arrival = ""
        m = re.search(r"gripper (arrived[^\n]*|stalled[^\n]*)", log)
        if m:
            arrival = m.group(0)
        print("   %s" % arrival, flush=True)
        print("   %s" % verdict, flush=True)
        results.append((depth, arrival, verdict))

    print()
    print("=" * 60)
    print("%-8s %s" % ("depth", "outcome"))
    for depth, arrival, verdict in results:
        print("%-8.3f %s" % (depth, verdict))


if __name__ == "__main__":
    main()
