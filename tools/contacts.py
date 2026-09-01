#!/usr/bin/env python3
"""Which part of the arm is stuck on what?

The arm reaches its planned posture in open floor and jams at the shelf, moving 0.002
rad/s, with no joint origin passing the shelf face. Rather than keep inferring the
geometry, this asks the simulator: every arm link carries a contact sensor.

Places the robot, starts a grasp, waits for it to jam, then reads every arm link contact
topic in turn and prints what each one is touching.
"""
import re
import subprocess
import time

LINKS = (["arm_left_%d_link" % i for i in range(1, 8)]
         + ["gripper_left_base_link", "torso_lift_link", "base_link"])
TOPIC = ("/world/erc_world/model/tiago_pro/link/%s/sensor/%s_contact/contact")


def sh(command, timeout=400):
    return subprocess.run(["bash", "-lc", command], capture_output=True,
                          text=True, timeout=timeout).stdout


def main():
    sh("sim restart --fast --headless >/dev/null 2>&1", timeout=300)
    time.sleep(26)
    print(sh("docker exec erc_sim /entrypoint.sh python3 /tmp/place_robot.py red 0.78 "
             "2>&1 | tail -2").strip())

    sh("docker exec -d erc_sim /entrypoint.sh bash -c "
       "'source /opt/erc_ws/install/setup.bash && timeout 200 python3 -u "
       "/tmp/ideal_grasp.py red > /tmp/ideal.log 2>&1'")
    print("grasp started; waiting for it to jam", flush=True)
    time.sleep(75)

    print()
    print("contacts while the arm is stuck:")
    found = False
    for link in LINKS:
        topic = TOPIC % (link, link)
        out = sh("docker exec erc_sim /entrypoint.sh bash -c "
                 "'timeout 4 gz topic -e -t %s 2>/dev/null | head -40'" % topic,
                 timeout=30)
        if not out.strip():
            continue
        others = set(re.findall(r'collision2\s*{\s*name:\s*"([^"]+)"', out))
        if not others:
            others = set(re.findall(r'name:\s*"([^"]+)"', out))
        others = {o for o in others if "tiago_pro" not in o or link not in o}
        if others:
            found = True
            print("   %-24s touching:" % link)
            for other in sorted(others)[:6]:
                print("        %s" % other)
    if not found:
        print("   none reported (the sensors publish only while touching)")

    print()
    print(sh("docker exec erc_sim bash -c 'strings /tmp/ideal_grasp.log "
             "| grep \"to go\" | tail -2'").strip())


if __name__ == "__main__":
    main()
