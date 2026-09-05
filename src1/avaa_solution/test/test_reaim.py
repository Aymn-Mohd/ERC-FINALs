"""Tests for the bound on re-aiming the grasp target, and for the arm's reach.

Both come from one failed grasp, and the bound went through two versions because the
first one was built on a belief about the base that turned out to be wrong.

The base COASTS. Measured against Gazebo over eight consecutive windows with a zero
twist published at 20 Hz throughout (tools/coast.py): 8.1, 7.4, 6.9, 8.5, 8.7, 6.8, 7.7
and 7.8 mm per simulated second, all on one heading, agreement 0.98 of 1.0. The wheel
model asks for exactly that -- mu2 is 0 across the roller axis, so nothing damps a
slide, and commanding zero wheel speed asks the wheels not to turn rather than asking
the base to stop.

So most of the corrections the failed run made were the robot having really travelled,
and a bound that refused them would be refusing the truth. Only one was impossible: it
arrived 0.6 s after the previous one asking to move the book 169 mm, which is 280 mm a
second against a coast of 7.7. That one put the target past the end of the arm, and the
servo then rejected two hundred consecutive solves and timed out 442 mm away.
"""

import math

import numpy as np
import pytest

from avaa_solution.grasp_node import ARM_MAX_REACH, reaim_budget
from avaa_solution.kinematics.arm_chain import ArmChain

ALLOWANCE = 0.06
RATE = 0.012

# The measured coast, in metres per simulated second, over eight windows.
COAST = [0.0081, 0.0074, 0.0069, 0.0085, 0.0087, 0.0068, 0.0077, 0.0078]

# The four corrections the failed run made, read off its log: how far each moved the
# target and how long after the target had last been set.
REAIMS = [
    (0.130, 67.5),    # raising: 1.9 mm/s, well inside a coast
    (0.119, 109.7),   # opening: 1.1 mm/s
    (0.067, 123.1),   # the end of the first leg of the reach: 0.5 mm/s
    (0.169, 0.6),     # 0.6 s after the second leg: 280 mm/s, and fatal
]


def test_budget_starts_at_the_allowance():
    assert reaim_budget(ALLOWANCE, RATE, 0.0) == pytest.approx(ALLOWANCE)


def test_budget_grows_with_time():
    assert reaim_budget(ALLOWANCE, RATE, 30.0) == pytest.approx(0.42)


def test_negative_elapsed_does_not_shrink_the_budget():
    assert reaim_budget(ALLOWANCE, RATE, -5.0) == pytest.approx(ALLOWANCE)


@pytest.mark.parametrize("speed", COAST)
def test_the_rate_covers_every_coast_speed_measured(speed):
    """The bound must not refuse the robot having really moved.

    Each window is checked at its own speed over a ten second gap, which is longer than
    the servo ever leaves between sightings and short enough that the fixed allowance
    is not doing the work.
    """
    assert speed * 10.0 < reaim_budget(ALLOWANCE, RATE, 10.0)


@pytest.mark.parametrize("moved,since", REAIMS[:3])
def test_corrections_consistent_with_the_coast_are_believed(moved, since):
    """Three of the four were the base sliding, and refusing them would be an error.

    The first version of this bound capped the budget at 120 mm and threw all three
    away, on the belief that a held base sits still.
    """
    assert moved <= reaim_budget(ALLOWANCE, RATE, since)


def test_the_correction_that_broke_the_grasp_is_refused():
    """169 mm in 0.6 s is 280 mm a second, thirty-five times the measured coast."""
    moved, since = REAIMS[3]
    assert moved > reaim_budget(ALLOWANCE, RATE, since)


def test_a_correction_the_size_of_perception_error_is_still_allowed():
    """Perception places the book to 15-35 mm in x, which is the sensor being itself."""
    assert 0.035 < reaim_budget(ALLOWANCE, RATE, 0.0)


def test_arm_max_reach_is_an_attainable_upper_bound():
    """The constant has to be the arm's, not a guess, in both directions."""
    chain = ArmChain.from_urdf()
    rng = np.random.default_rng(7)
    limits = chain.limits
    best = 0.0
    for _ in range(3000):
        values = [0.0] + [rng.uniform(lo, hi) for lo, hi in limits[1:]]
        origins = chain.joint_origins(values)
        best = max(best, float(np.linalg.norm(origins[-1] - origins[1])))
    assert best <= ARM_MAX_REACH + 1e-6
    assert best > 0.75 * ARM_MAX_REACH


def test_the_target_that_failed_was_inside_the_arms_reach():
    """The 445 mm collapse was not the arm being over-extended.

    Commanded as one posture and given time to settle, the arm holds this target to
    3 mm (tools/sagcheck.py), and holds 0.97 m to 1 mm -- while the static torque
    estimate read 112 and 135 per cent of rated. Neither reach nor torque explains the
    failure, which is why the bound above is on the sighting instead.
    """
    chain = ArmChain.from_urdf()
    shoulder = chain.joint_origins([0.35] + [0.0] * 7)[1]
    need = math.dist([0.935, 0.2, 1.346], [float(v) for v in shoulder])
    assert need < ARM_MAX_REACH
    assert need / ARM_MAX_REACH == pytest.approx(0.85, abs=0.02)
