#!/usr/bin/env python3
"""Extract the kinematic chain from the URDF, base_link -> gripper_left_grasping_link.

MoveIt is installed but has no SRDF for this robot, so there is no configured planning
group to ask for IK. Building the chain directly is the alternative, and it needs each
joint's parent, child, origin, axis and limits.
"""
import sys
import xml.etree.ElementTree as ET

URDF = "/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf"
TARGET = "gripper_left_grasping_link"
ROOT = "base_link"


def main():
    tree = ET.parse(URDF)
    root = tree.getroot()

    joints = {}
    parent_of = {}
    for j in root.findall("joint"):
        name = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        axis_el = j.find("axis")
        axis = axis_el.get("xyz") if axis_el is not None else None
        limit = j.find("limit")
        lo = limit.get("lower") if limit is not None else None
        hi = limit.get("upper") if limit is not None else None
        joints[name] = dict(type=j.get("type"), parent=parent, child=child,
                            xyz=xyz, rpy=rpy, axis=axis, lower=lo, upper=hi)
        parent_of[child] = name

    # Walk up from the target to the root.
    chain = []
    link = TARGET
    while link != ROOT:
        jname = parent_of.get(link)
        if jname is None:
            print(f"chain broken at link {link}", file=sys.stderr)
            break
        chain.append(jname)
        link = joints[jname]["parent"]
    chain.reverse()

    print(f"chain {ROOT} -> {TARGET}: {len(chain)} joints\n")
    for name in chain:
        j = joints[name]
        moving = j["type"] not in ("fixed",)
        mark = "*" if moving else " "
        lim = ""
        if moving and j["lower"] is not None:
            lim = f"  limits[{float(j['lower']):+.3f}, {float(j['upper']):+.3f}]"
        print(f" {mark} {name}")
        print(f"     type={j['type']:9s} axis={j['axis']}{lim}")
        print(f"     xyz=({j['xyz']})  rpy=({j['rpy']})")

    movers = [n for n in chain if joints[n]["type"] not in ("fixed",)]
    print(f"\nmoving joints ({len(movers)}): {movers}")


if __name__ == "__main__":
    main()
