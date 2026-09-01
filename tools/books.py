#!/usr/bin/env python3
"""List the books on the shelf, with their true poses.

The layout is randomised on every launch -- the colour in a given row and column
changes, and so does the exact sideways position -- so a run has to start by asking
what is actually there. Grouped by column, since a grasp is aimed at one column.

    tools/in-sim books.py
"""
import subprocess
import sys
import time


def gz(*args, timeout=25):
    try:
        return subprocess.run(["gz", *args], capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return ""


def pose(model, attempts=5):
    """Gazebo drops query requests when it is busy, so ask more than once."""
    for _ in range(attempts):
        lines = [l.strip() for l in gz("model", "-m", model, "-p").splitlines()]
        for i, line in enumerate(lines):
            if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
                try:
                    return [float(v) for v in line.strip("[]").split()]
                except ValueError:
                    return None
        time.sleep(0.3)
    return None


def main():
    names = []
    for _ in range(8):
        names = sorted(l.strip(" -") for l in gz("model", "--list").splitlines()
                       if "book_col" in l)
        if names:
            break
        time.sleep(0.5)
    if not names:
        print("no books; is the simulator up? try tools/simready.sh")
        return 1

    # Rows are numbered top-down in the arena but the grasp counts reachable rows,
    # so show both rather than make the reader work it out.
    graspable = {"row_2": "grasp row 1", "row_3": "grasp row 2",
                 "row_4": "grasp row 3", "row_5": "grasp row 4"}
    column = None
    for name in names:
        col = name.split("_row_")[0]
        if col != column:
            column = col
            print()
            print(col.replace("book_", "").replace("_", " "))
        p = pose(name)
        if p is None:
            print("    %-30s (could not read)" % name)
            continue
        row = "row_" + name.split("_row_")[1].split("_")[0]
        print("    %-30s x=%.3f y=%+.3f z=%.3f   %s"
              % (name, p[0], p[1], p[2], graspable.get(row, "")))
    print()
    print("The layout is randomised each launch. Pass a whole name or just a colour:")
    print("    tools/in-sim place_robot.py <colour|name> 0.68")
    print("    tools/in-sim ideal_grasp.py <colour|name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
