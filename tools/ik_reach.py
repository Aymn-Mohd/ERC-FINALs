#!/usr/bin/env python3
"""Offline IK design check for left-arm shelf grasps (no ROS / MoveIt).

Reports whether analytic IK can reach pre-grasp and walk into the book for each
stocked row at typical face distances. Run on the host:

    PYTHONPATH=src/avaa_solution python tools/ik_reach.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "avaa_solution"))

from avaa_solution.kinematics.arm_chain import ArmChain, DEFAULT_URDF  # noqa: E402

TUCK = [0.15, 2.1521, 0.3824, 1.2785, -2.1517, 0.8325, 0.1926, 1.3944]
APP, CLO = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
HEIGHTS = [1.391, 1.061, 0.731, 0.401]
SHOULDER_BASE_Z = 0.677
BELOW, STANDOFF, DEPTH = 0.045, 0.15, 0.11


def torso_pin(z: float) -> dict:
    ideal = float(np.clip(z - SHOULDER_BASE_Z + 0.25, 0.0, 0.35))
    return {"torso_lift_joint": (ideal, 0.10)}


def main() -> int:
    chain = ArmChain.from_urdf(DEFAULT_URDF)
    print("URDF:", DEFAULT_URDF)
    print("tuck FK:", np.round(chain.fk(TUCK)[:3, 3], 4).tolist())
    print()
    print(f"{'row':>3} {'face':>5} {'pre':>5} {'grasp':>5} {'line':>6}")
    for row, h in enumerate(HEIGHTS, 1):
        z = h - BELOW
        pin = torso_pin(z)
        for face in (0.70, 0.75, 0.80, 0.85):
            pre = np.array([face - STANDOFF, 0.0, z])
            grasp = np.array([face + DEPTH, 0.0, z])
            sol = chain.ik(pre, seed=TUCK, approach=APP, closing=CLO, pin=pin)
            if sol is None:
                print(f"{row:3d} {face:5.2f} {pre[0]:5.2f} {grasp[0]:5.2f}  FAIL")
                continue
            seed = list(sol)
            ok = 0
            for step in range(1, 5):
                pt = pre + (step / 4.0) * (grasp - pre)
                nxt = chain.ik(pt, seed=seed, approach=APP, closing=CLO, pin=pin)
                if nxt is None:
                    break
                seed = nxt
                ok = step
            print(f"{row:3d} {face:5.2f} {pre[0]:5.2f} {grasp[0]:5.2f}  {ok}/4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
