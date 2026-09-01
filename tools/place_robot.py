#!/usr/bin/env python3
"""Put the robot squarely in front of a chosen book, for controlled grasp experiments.

    python3 place_robot.py <book_colour> [standoff]

The approach is not reliable enough yet to be a test fixture. Waiting for it to succeed
before every grasp experiment means most attempts never reach the thing being tested, and
a run that ends beside the wrong column tells you nothing about the fingers.

This teleports the base to a pose the approach is *trying* to reach -- squared to the
shelf, centred on the target book, at a chosen standoff -- so that everything downstream
of the base can be tested on its own. It is a fixture, not part of the solution.
"""
import math
import subprocess
import sys
import time


def gz(*args, timeout=25):
    return subprocess.run(["gz", *args], capture_output=True, text=True,
                          timeout=timeout).stdout


def pose(model):
    lines = [l.strip() for l in gz("model", "-m", model, "-p").splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("[") and i + 1 < len(lines) and lines[i + 1].startswith("["):
            try:
                return ([float(v) for v in line.strip("[]").split()],
                        [float(v) for v in lines[i + 1].strip("[]").split()])
            except ValueError:
                return None, None
    return None, None


def main():
    colour = sys.argv[1] if len(sys.argv) > 1 else "red"
    standoff = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    # Line the book up with the arm, not with the middle of the robot. The left
    # shoulder sits 0.159 m to the left of base_link, so a book centred on the base is
    # that far off the arm centre line and the forearm has to cross the shelf opening
    # diagonally to reach it.
    shoulder = float(sys.argv[3]) if len(sys.argv) > 3 else 0.159

    # ``colour`` may instead be a full model name, so that a particular row can be
    # tested rather than whichever row the randomiser put that colour on.
    # Ask more than once. Gazebo drops query requests when it is busy, and a dropped
    # listing here does not fail politely -- the empty list falls straight through to an
    # IndexError and the run is gone before it started.
    names = []
    for _ in range(8):
        names = [l.strip(" -") for l in gz("model", "--list").splitlines()
                 if "book_col" in l and (colour in l or colour == l.strip(" -"))]
        if names:
            break
        time.sleep(0.5)
    if not names:
        print("no %s books" % colour)
        return

    # Pick the one nearest the middle of the shelf, so the base has room either side.
    books = []
    for name in names:
        for _ in range(5):
            p, _ = pose(name)
            if p:
                books.append((abs(p[1]), name, p))
                break
            time.sleep(0.4)
    books.sort()
    if not books:
        print("could not read any %s book back from Gazebo" % colour)
        return
    _, name, book = books[0]

    # Square to the shelf, which faces -x, so the robot looks along +x at yaw 0.
    x = book[0] - standoff
    y = book[1] - shoulder
    request = ('name: "tiago_pro", position: {x: %f, y: %f, z: 0.0}, '
               'orientation: {x: 0, y: 0, z: 0, w: 1}' % (x, y))
    out = gz("service", "-s", "/world/erc_world/set_pose",
             "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
             "--timeout", "3000", "--req", request)
    print("placing in front of %s at [%.3f, %.3f], yaw 0" % (name, x, y))
    print("set_pose said: %s" % out.strip())

    after, rpy = pose("tiago_pro")
    if after:
        print("robot now at [%.3f, %.3f] yaw %.1f deg"
              % (after[0], after[1], math.degrees(rpy[2])))
        print("book %s at [%.3f, %.3f, %.3f]" % (name, book[0], book[1], book[2]))
        print("so the book should be %.3f m ahead and %+.3f m to the side"
              % (book[0] - after[0], book[1] - after[1]))


if __name__ == "__main__":
    main()
