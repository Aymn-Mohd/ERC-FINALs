"""Tests for the gate that decides whether a book sighting is the book we set out for.

The gate used to be held in odom, and odom is the one frame here it must not use.
Measured during a run held to 17 mm of true error, odom accumulated 813 mm of travel
that never happened, because holding this base still means driving the wheels against a
slide and odom faithfully integrates every one of those turns. The consequence was a
target that walked away from the book: one run rejected 24 consecutive correct sightings
as "2.42 m from the anchored target", reversed 2.98 m away from the shelf, lost the book
entirely and fell back to searching.

Continuity in base_link does the same job. The thing the gate exists to refuse is a
bearing that has jumped to a different book of the same colour, and those are 0.95 m
apart between columns and 0.33 m between rows -- distances in the robot's own frame now,
which is exactly the frame odom was not.
"""

import pytest

from avaa_solution.approach_node import sighting_gate

ALLOWANCE = 0.08
SPEED = 0.35
COLUMN_SPACING = 0.95
ROW_SPACING = 0.33


def test_a_sighting_arriving_immediately_gets_the_allowance():
    assert sighting_gate(ALLOWANCE, SPEED, 0.0) == pytest.approx(ALLOWANCE)


def test_the_budget_covers_the_base_driving_between_sightings():
    """Perception publishes at about 5 Hz and the base drives at 0.22 m/s."""
    assert sighting_gate(ALLOWANCE, SPEED, 0.2) > 0.22 * 0.2


def test_the_budget_stops_growing_after_a_long_silence():
    """A sighting after a long gap has nothing recent to be continuous with.

    Without the cap the gate would widen until it admitted the whole shelf, which is
    the failure the odom version had by a different route.
    """
    assert sighting_gate(ALLOWANCE, SPEED, 30.0) == sighting_gate(
        ALLOWANCE, SPEED, 1.5)
    assert sighting_gate(ALLOWANCE, SPEED, 1.5) < COLUMN_SPACING


def test_a_jump_to_the_next_column_is_refused_at_any_gap():
    """0.95 m is the distance that broke a run when it was admitted."""
    for gap in (0.0, 0.2, 0.5, 1.0, 1.5, 10.0):
        assert COLUMN_SPACING > sighting_gate(ALLOWANCE, SPEED, gap)


def test_a_jump_to_the_row_above_is_refused_at_a_normal_sighting_gap():
    for gap in (0.0, 0.2, 0.4):
        assert ROW_SPACING > sighting_gate(ALLOWANCE, SPEED, gap)


def test_the_corrections_the_odom_gate_refused_are_now_inside_it():
    """It rejected 24 sightings at 2.42 m because the FRAME had moved, not the book.

    In base_link that disagreement cannot arise from odom at all, so the case to check
    is the one it was really seeing: consecutive sightings of one book while driving.
    """
    assert 0.22 * 0.2 + 0.035 < sighting_gate(ALLOWANCE, SPEED, 0.2)


def test_a_negative_gap_does_not_shrink_the_budget():
    assert sighting_gate(ALLOWANCE, SPEED, -3.0) == pytest.approx(ALLOWANCE)
