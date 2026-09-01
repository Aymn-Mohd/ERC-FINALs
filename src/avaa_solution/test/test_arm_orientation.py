"""The wrist must be pinned, not just the fingertip position.

These exist because the grasp failed four times in a row while every log said it had
worked. The arm has seven joints for three position constraints, so a position-only
solve resolves the spare freedom arbitrarily: it put the gripper exactly on target with
the approach axis 78 degrees out, reaching in from the side, and the fingers closed past
the corner of the book without touching it.
"""

import math

import numpy as np

import pytest

from avaa_solution.kinematics.arm_chain import ArmChain, _angle_between

APPROACH = [1.0, 0.0, 0.0]
CLOSING = [0.0, 1.0, 0.0]

# Directly in front, one target per shelf row, at the depth a grasp actually reaches to.
ROW_TARGETS = [
    [0.836, 0.020, 1.391],
    [0.836, 0.020, 1.061],
    [0.836, 0.020, 0.731],
    [0.836, 0.020, 0.401],
]


@pytest.fixture(scope="module")
def chain():
    return ArmChain.from_urdf()


def axes(chain, solution):
    """Return the approach and finger-closing axes for a solution, in base_link."""
    pose = chain.fk(solution)
    return pose[:3, 0], pose[:3, 1]


def test_angle_between_is_bounded_at_the_poles():
    # acos of a dot product that floating point pushed past 1.0 would raise instead.
    assert _angle_between(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])) == pytest.approx(0.0)
    assert _angle_between(np.array([1.0, 0, 0]),
                          np.array([-1.0, 0, 0])) == pytest.approx(math.pi)


@pytest.mark.parametrize("target", ROW_TARGETS)
def test_oriented_solve_reaches_every_row(chain, target):
    solution = chain.ik(target, approach=APPROACH, closing=CLOSING)
    assert solution is not None, "no oriented solution for %s" % target
    assert np.linalg.norm(chain.position(solution) - np.array(target)) < 0.005


@pytest.mark.parametrize("target", ROW_TARGETS)
def test_oriented_solve_points_into_the_shelf(chain, target):
    solution = chain.ik(target, approach=APPROACH, closing=CLOSING)
    approach_axis, closing_axis = axes(chain, solution)
    assert _angle_between(approach_axis, np.array(APPROACH)) < 0.26
    # The jaws are symmetric, so either way round the fingers is the same grasp.
    assert min(_angle_between(closing_axis, np.array(CLOSING)),
               _angle_between(closing_axis, -np.array(CLOSING))) < 0.26


def test_closing_axis_is_sign_agnostic(chain):
    """Asking for the fingers the other way round must not change the solve."""
    target = ROW_TARGETS[-1]
    forward = chain.ik(target, approach=APPROACH, closing=CLOSING)
    reversed_ = chain.ik(target, approach=APPROACH, closing=[0.0, -1.0, 0.0])
    assert forward is not None and reversed_ is not None
    for solution in (forward, reversed_):
        _, closing_axis = axes(chain, solution)
        assert min(_angle_between(closing_axis, np.array(CLOSING)),
                   _angle_between(closing_axis, -np.array(CLOSING))) < 0.26


def test_position_only_solve_still_works_for_free_space(chain):
    """Waypoints away from the shelf do not need a wrist, and must not start failing."""
    target = [0.6, 0.2, 1.0]
    solution = chain.ik(target)
    assert solution is not None
    assert np.linalg.norm(chain.position(solution) - np.array(target)) < 0.005


def test_impossible_orientation_returns_none_rather_than_a_wrong_wrist(chain):
    """A silent near-miss on orientation is what caused the failures; refuse instead."""
    # Approach and closing must be perpendicular; they are gripper axes. Asking for
    # them parallel cannot be satisfied by any pose.
    assert chain.ik(ROW_TARGETS[-1], approach=[1.0, 0.0, 0.0],
                    closing=[1.0, 0.0, 0.0]) is None


def test_unreachable_target_returns_none(chain):
    assert chain.ik([3.0, 0.0, 1.0], approach=APPROACH, closing=CLOSING) is None
