#!/usr/bin/env python3
"""Where on the shelf is the arm actually touching?

The shelf collision is the same STL as its visual, so in principle the openings are open.
But a non-convex mesh is often approximated by its convex hull, which would make the whole
unit solid and every opening unreachable -- and that would explain an arm that jams with
four links against the shelf while no joint origin has passed its face.

Contact positions decide it. Touching the front face of a shelf board is normal geometry.
Touching thin air in the middle of an opening means the hull.
"""
import re
import subprocess
import time

LINKS = ["arm_left_%d_link" % i for i in range(4, 8)]
TOPIC = "/world/erc_world/model/tiago_pro/link/%s/sensor/%s_contact/contact"


def sh(command, timeout=120):
    return subprocess.run(["bash", "-lc", command], capture_output=True,
                          text=True, timeout=timeout).stdout


def main():
    print("book faces sit at world x = 2.82; shelf boards span x 2.82 to 3.02")
    print("row surfaces are at world z = 0.587, 0.917, 1.247, 1.577")
    print()
    for link in LINKS:
        out = sh("docker exec erc_sim /entrypoint.sh bash -c "
                 "'timeout 4 gz topic -e -t %s 2>/dev/null | head -60'"
                 % (TOPIC % (link, link)), timeout=40)
        if not out.strip():
            continue
        # Contact points come through as position { x: .. y: .. z: .. } blocks.
        points = re.findall(
            r"position\s*{\s*x:\s*([-\d.e]+)\s*y:\s*([-\d.e]+)\s*z:\s*([-\d.e]+)",
            out)
        if not points:
            continue
        print("%s: %d contact point(s)" % (link, len(points)))
        for x, y, z in points[:4]:
            x, y, z = float(x), float(y), float(z)
            if x < 2.80:
                verdict = "IN FRONT of the shelf face -- nothing solid should be here"
            elif x > 3.02:
                verdict = "behind the shelf"
            else:
                verdict = "within the shelf depth"
            print("    (%.3f, %.3f, %.3f)  %s" % (x, y, z, verdict))


if __name__ == "__main__":
    main()
