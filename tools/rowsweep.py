#!/usr/bin/env python3
"""Which shelf rows can the arm actually reach into, and from how far back?

Middle rows work: arrivals of 8 mm, and of -29 mm depth with 0 mm sideways and 1 mm of
height. The bottom row does not -- the forearm meets base_link_shelf_collision with the
gripper still 115 mm out, on a straight-line path, from a pre-grasp the arm reaches
cleanly. That is a reachability limit rather than a bug, and it decides what the solution
can promise.

Targets a named book so the row is chosen rather than left to the randomiser, and reports
what happened for each.
"""
import re
import subprocess
import sys
import time

CASES = sys.argv[1:] or [
    "book_col_3_row_2", "book_col_3_row_3", "book_col_3_row_4", "book_col_3_row_5",
]
STANDOFF = 0.72


def sh(command, timeout=500):
    return subprocess.run(["bash", "-lc", command], capture_output=True,
                          text=True, timeout=timeout).stdout


def full_name(prefix):
    listing = sh("docker exec erc_sim /entrypoint.sh gz model --list 2>/dev/null")
    for line in listing.splitlines():
        name = line.strip(" -")
        if name.startswith(prefix):
            return name
    return None


def main():
    results = []
    for prefix in CASES:
        print("=" * 62, flush=True)
        sh("sim restart --fast --headless >/dev/null 2>&1", timeout=300)
        time.sleep(26)
        name = full_name(prefix)
        if name is None:
            print("%s: not in this world" % prefix)
            continue
        row = prefix.rsplit("_", 1)[-1]
        print("%s  (world row %s)" % (name, row), flush=True)

        sh("docker exec erc_sim /entrypoint.sh bash -c "
           "'source /opt/erc_ws/install/setup.bash && python3 -u /tmp/tuck_arm.py 40'",
           timeout=120)
        sh("docker exec erc_sim /entrypoint.sh python3 /tmp/place_robot.py %s %f"
           % (name, STANDOFF), timeout=90)
        sh("docker exec erc_sim /entrypoint.sh bash -c "
           "'source /opt/erc_ws/install/setup.bash && timeout 400 python3 -u "
           "/tmp/ideal_grasp.py %s > /tmp/ideal.log 2>&1'" % name, timeout=460)

        grasp = sh("docker exec erc_sim bash -c 'strings /tmp/ideal_grasp.log'")
        outcome = sh("docker exec erc_sim bash -c 'strings /tmp/ideal.log'")
        verdict = next((l.strip() for l in outcome.splitlines()
                        if l.startswith("RESULT:")), "no verdict")
        arrived = re.search(r"gripper arrived: [^\n]+", grasp)
        gaps = [int(m) for m in re.findall(r"(\d+) mm to go", grasp)]
        detail = arrived.group(0) if arrived else (
            "stalled at %d mm" % min(gaps) if gaps else "no progress logged")
        print("   %s" % detail, flush=True)
        print("   %s" % verdict, flush=True)
        results.append((prefix, detail, verdict))

    print()
    print("=" * 62)
    for prefix, detail, verdict in results:
        print("%-22s %-46s %s" % (prefix, detail[:46], verdict))


if __name__ == "__main__":
    main()
