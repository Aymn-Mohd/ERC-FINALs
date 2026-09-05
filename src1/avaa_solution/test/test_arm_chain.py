"""Unit tests for the arm kinematics.

The URDF is part of the competition image, so these are skipped rather than failed where
it is absent.
"""

import math
import os

import numpy as np
import pytest

from avaa_solution.kinematics.arm_chain import (
    DEFAULT_URDF,
    ArmChain,
    axis_rotation,
    rpy_to_matrix,
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_URDF), reason="robot URDF not present"
)


@pytest.fixture(scope="module")
def chain():
    return ArmChain.from_urdf()


# ----------------------------------------------------------------- maths helpers


def test_identity_rotation():
    assert np.allclose(rpy_to_matrix(0, 0, 0), np.eye(3))


def test_yaw_rotates_x_onto_y():
    got = rpy_to_matrix(0, 0, math.pi / 2) @ np.array([1, 0, 0])
    assert np.allclose(got, [0, 1, 0], atol=1e-9)


def test_roll_rotates_y_onto_z():
    got = rpy_to_matrix(math.pi / 2, 0, 0) @ np.array([0, 1, 0])
    assert np.allclose(got, [0, 0, 1], atol=1e-9)


def test_axis_rotation_matches_yaw_about_z():
    assert np.allclose(
        axis_rotation(np.array([0, 0, 1.0]), 0.7), rpy_to_matrix(0, 0, 0.7), atol=1e-9
    )


def test_rotations_are_orthonormal():
    r = rpy_to_matrix(0.3, -1.1, 2.0)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)


# ----------------------------------------------------------------- the chain


def test_chain_has_the_expected_moving_joints(chain):
    assert chain.joint_names == [
        "torso_lift_joint",
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        "arm_left_4_joint",
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
    ]


def test_fk_at_zero_matches_the_measured_pose(chain):
    # Cross-checked against TF in the running simulator: the gripper sits at
    # (0.983, 0.493, 0.227) in base_link with every joint at zero.
    assert np.allclose(chain.position([0.0] * 8), [0.983, 0.493, 0.227], atol=2e-3)


def test_torso_raises_the_gripper_one_for_one(chain):
    low = chain.position([0.0] * 8)
    high = chain.position([0.30] + [0.0] * 7)
    assert np.isclose(high[2] - low[2], 0.30, atol=1e-6)
    assert np.allclose(high[:2], low[:2], atol=1e-9)


def test_fk_rejects_the_wrong_number_of_values(chain):
    with pytest.raises(ValueError):
        chain.fk([0.0] * 7)


def test_clamp_respects_limits(chain):
    extreme = [1e3] * len(chain.joint_names)
    for value, (lo, hi) in zip(chain.clamp(extreme), chain.limits):
        assert lo <= value <= hi


# ----------------------------------------------------------------- IK


@pytest.mark.parametrize("target", [
    [0.80, 0.0, 1.391],   # shelf row 1, book centred on the base
    [0.80, 0.0, 1.061],   # row 2
    [0.80, 0.0, 0.731],   # row 3
    [0.80, 0.0, 0.401],   # row 4
])
def test_every_shelf_row_is_reachable(chain, target):
    solution = chain.ik(target)
    assert solution is not None, f"no IK solution for {target}"
    assert np.allclose(chain.position(solution), target, atol=5e-3)


def test_ik_solution_respects_joint_limits(chain):
    solution = chain.ik([0.80, 0.0, 1.061])
    assert solution is not None
    for value, (lo, hi) in zip(solution, chain.limits):
        assert lo - 1e-6 <= value <= hi + 1e-6


def test_ik_round_trips_a_reachable_pose(chain):
    pose = [0.10, 0.4, -1.0, 0.2, -0.8, 0.1, 0.3, -0.2]
    target = chain.position(pose)
    solution = chain.ik(target, seed=pose)
    assert solution is not None
    assert np.allclose(chain.position(solution), target, atol=5e-3)


def test_ik_returns_none_when_out_of_reach(chain):
    # Far beyond the arm's span; must report failure rather than a nearest-point guess.
    assert chain.ik([5.0, 0.0, 1.0]) is None
