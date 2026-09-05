"""Tests for sizing a turn against the rate its steering bearing actually arrives at.

The centring controller had a fixed gain that saturated its 0.45 rad/s limit for any
error over about 93 px. The bearing that steers it comes from perception, which
processes a frame roughly every two SIMULATED seconds -- watched in the approach's own
log, where the reported bearing changed at 468255, 468257 and 468259 and then repeated.
Turning at 0.45 rad/s for two seconds is 0.9 rad, and the head camera sees about 1.0 rad
in total, so the base swung nearly its whole field of view between one look and the
next. The marker left the frame, no bearing arrived at all, and the state stood there
until its twelve second out-of-view timeout fired. Three times in one run.

A fixed gain cannot be right here, because the period is not fixed: the real-time factor
in this project has been measured between 0.013 and 0.60, a span of forty-five, and the
frame rate follows it.
"""

import math

import pytest

from avaa_solution.approach_node import turn_for

CAMERA_FOV = 1.0        # radians the head camera sees, near enough
SLOW = 2.0              # the measured bearing period, in simulated seconds


def test_a_turn_never_sweeps_more_than_the_error_it_is_correcting():
    """The whole point. At any error and any period, one period is not an overshoot."""
    for error in (0.05, 0.2, 0.35, 0.5, 1.0):
        for period in (0.3, 1.0, 2.0, 5.0):
            swept = abs(turn_for(error, period)) * period
            assert swept <= error + 1e-9


def test_the_old_behaviour_would_have_swept_most_of_the_field_of_view():
    """0.45 rad/s for a two second gap is 0.9 rad, against a 1.0 rad camera."""
    assert 0.45 * SLOW > 0.8 * CAMERA_FOV


def test_the_new_behaviour_does_not():
    error = math.radians(20.0)
    assert abs(turn_for(error, SLOW)) * SLOW < 0.25 * CAMERA_FOV


def test_a_faster_camera_is_allowed_a_faster_turn():
    error = math.radians(20.0)
    assert abs(turn_for(error, 0.3)) > abs(turn_for(error, 2.0))


def test_the_cap_still_applies_however_fast_the_camera_is():
    assert abs(turn_for(1.5, 0.01)) <= 0.45
    assert abs(turn_for(1.5, 0.01, fastest=0.2)) <= 0.2


def test_the_direction_is_kept():
    assert turn_for(0.3, 1.0) > 0
    assert turn_for(-0.3, 1.0) < 0


def test_a_zero_error_asks_for_no_turn():
    assert turn_for(0.0, 1.0) == pytest.approx(0.0)


def test_an_implausibly_short_period_cannot_ask_for_an_unbounded_rate():
    """A period floor keeps a burst of fast frames from re-creating the old problem."""
    assert abs(turn_for(0.3, 0.0)) == abs(turn_for(0.3, 0.25))
