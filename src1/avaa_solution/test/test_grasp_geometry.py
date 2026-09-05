"""Tests for the row-to-height mapping and the gripper command values.

The row direction is the risky part: the rules number the stocked rows 1-4 without saying
which end row 1 is. Getting it backwards costs the identification points and sends the arm
to the wrong shelf, so both directions are pinned down here.
"""

import pytest

from avaa_solution.grasp_node import (
    ARRIVAL_TOL_LATERAL,
    BOOK_THICKNESS,
    DEFAULT_ROW_HEIGHTS,
    GRIPPER_CLAMP,
    GRIPPER_OPEN,
    GRIPPER_OPEN_MIN,
    SERVO_RELEASE_GAP,
    TUCK_POSE,
    finger_for_gap,
    pad_gap,
    row_to_height,
)

HEIGHTS = [1.391, 1.061, 0.731, 0.401]   # top shelf first


def test_defaults_are_ordered_top_shelf_first():
    assert DEFAULT_ROW_HEIGHTS == sorted(DEFAULT_ROW_HEIGHTS, reverse=True)


def test_rows_are_evenly_spaced():
    gaps = [a - b for a, b in zip(DEFAULT_ROW_HEIGHTS, DEFAULT_ROW_HEIGHTS[1:])]
    for gap in gaps:
        assert gap == pytest.approx(0.33, abs=0.005)


@pytest.mark.parametrize("row,expected", [(1, 1.391), (2, 1.061), (3, 0.731), (4, 0.401)])
def test_top_down_numbering(row, expected):
    assert row_to_height(row, HEIGHTS, top_down=True) == pytest.approx(expected)


@pytest.mark.parametrize("row,expected", [(1, 0.401), (2, 0.731), (3, 1.061), (4, 1.391)])
def test_bottom_up_numbering(row, expected):
    assert row_to_height(row, HEIGHTS, top_down=False) == pytest.approx(expected)


def test_the_two_directions_are_mirror_images():
    top = [row_to_height(r, HEIGHTS, True) for r in range(1, 5)]
    bottom = [row_to_height(r, HEIGHTS, False) for r in range(1, 5)]
    assert top == list(reversed(bottom))


@pytest.mark.parametrize("row", [0, 5, -1, 99])
def test_out_of_range_rows_return_none(row):
    # Must surface rather than quietly aiming at the nearest shelf.
    assert row_to_height(row, HEIGHTS) is None


def test_empty_height_table_returns_none():
    assert row_to_height(1, []) is None


def test_gripper_open_clears_a_book_and_clamp_closes_past_it():
    # Measured span curve: span ~= 0.028 + 0.82 * joint. A book is 0.030 m thick.
    def span(joint):
        return 0.028 + 0.82 * joint

    assert span(GRIPPER_OPEN) > 0.030 * 1.8, "open span should give clear approach room"
    assert span(GRIPPER_CLAMP) < 0.030, "clamp must close past the book thickness"


def test_clamp_closes_past_a_book():
    """The jaws have to end up narrower than the book, or they grip nothing.

    This is the one that cost the most. At a commanded 0.000 the joint settles at 0.0026
    with the fingertips 30.4 mm apart, measured from TF, and a book is 30.0 mm thick: the
    jaws closed around it with 0.2 mm to spare on each side. A grasp that arrived 8 mm
    from plan, centred on the spine and 20 mm inside the front face, still lifted nothing.
    """
    def span(joint):
        # Fitted to three TF measurements: 60.5 mm at 0.040, 44.5 at 0.020, 30.4 at 0.0026.
        return 0.0285 + 0.80 * joint

    assert span(GRIPPER_OPEN) > 0.030 * 1.8, "open span should give clear approach room"
    assert span(GRIPPER_CLAMP) < 0.030, "the clamp must close past the book, not around it"
    assert span(GRIPPER_CLAMP) > 0.020, "and not so far that it is asking the impossible"


def test_the_tuck_is_not_the_old_self_colliding_one():
    """A regression guard on a pose that was wrong for the whole project.

    [-0.5, -2.4, 0, -2.4, 0, 0, 0] puts arm_left_2 through arm_left_5 against
    torso_base_link and torso_lift_link. Gazebo does not check self-collision so nothing
    complained, and MoveIt then would not plan from it at all. The replacement is checked
    against /check_state_validity by tools/find_tuck.py, which needs a running simulator
    and so cannot be asserted here; what can be asserted is that the old one is gone.
    """
    assert TUCK_POSE != [-0.5, -2.4, 0.0, -2.4, 0.0, 0.0, 0.0]
    assert len(TUCK_POSE) == 7


# --------------------------------------------------------------------- the pads
#
# The span model above measures the fingertip LINK ORIGINS, which sit about 22 mm
# outboard of the surfaces that meet the book. Every clearance in this controller used
# to be read off it, and every one of them was 22 mm too generous. These pin the pad
# model and the two defaults that depend on it.


@pytest.mark.parametrize("finger,gap_mm", [
    (0.052, 46.7), (0.0377, 35.7), (0.030, 29.5), (0.020, 21.4), (0.0012, 5.9),
])
def test_pad_gap_matches_the_measured_geometry(finger, gap_mm):
    """From the fingertip mesh through the mimic linkage, in the grasping frame.

    Fitted over 0.001 to 0.052, which is every value a grasp commands. Outside it the
    line runs wide -- 61.5 against a measured 59.7 at 0.070 -- so the model must not be
    used to justify opening further than GRIPPER_OPEN without re-fitting.
    """
    assert pad_gap(finger) * 1000 == pytest.approx(gap_mm, abs=0.4)


def test_the_pad_model_is_not_trusted_beyond_the_range_it_was_fitted_over():
    assert GRIPPER_OPEN <= 0.052, "the pad model is only good to 0.052"
    assert pad_gap(0.070) * 1000 == pytest.approx(61.5, abs=0.4)


def test_finger_for_gap_is_the_inverse():
    for gap in (0.010, 0.030, 0.047, 0.060):
        assert pad_gap(finger_for_gap(gap)) == pytest.approx(gap, abs=1e-9)


def test_the_open_jaws_clear_the_book_on_both_sides():
    clearance = (pad_gap(GRIPPER_OPEN) - BOOK_THICKNESS) / 2.0
    assert clearance > 0.005, "the pads must pass either side of the spine"


def test_the_clamp_closes_past_the_book():
    assert pad_gap(GRIPPER_CLAMP) < BOOK_THICKNESS


def test_arrival_tolerance_is_inside_the_clearance_the_pads_actually_have():
    """The regression that let the servo clamp from where a pad was inside the book.

    The tolerance was 12 mm because the opening was read off the span -- 69.7 mm, so
    "nearly 20 mm a side". The pads are 46.8 mm apart, which is 8.4 mm a side, and a
    clamp begun 12 mm off centre meets the front corner of the book and shoves it.
    """
    clearance = (pad_gap(GRIPPER_OPEN) - BOOK_THICKNESS) / 2.0
    assert ARRIVAL_TOL_LATERAL < clearance, (
        "an arrival judged good at %.1f mm off centre has a pad %.1f mm inside the book"
        % (ARRIVAL_TOL_LATERAL * 1000, (ARRIVAL_TOL_LATERAL - clearance) * 1000))
    assert ARRIVAL_TOL_LATERAL < clearance - 0.002, "leave room for the drift"


def test_the_arm_stops_tracking_before_the_pads_touch_the_book():
    """The gate that never opened.

    Expressed as a span of 0.040 this was a pad gap of 17 mm. The fingers stall on a
    30 mm book long before that, so the condition stayed true for the whole close and
    the arm went on correcting sideways against a book that tips at a third of a
    newton. It has to release at a gap the fingers actually pass through, and while
    there is still air either side of the book.
    """
    assert SERVO_RELEASE_GAP >= BOOK_THICKNESS, (
        "the fingers stall on the book at %.0f mm and never reach %.0f mm"
        % (BOOK_THICKNESS * 1000, SERVO_RELEASE_GAP * 1000))
    assert SERVO_RELEASE_GAP <= pad_gap(GRIPPER_OPEN), "it must not release immediately"
    assert SERVO_RELEASE_GAP - BOOK_THICKNESS < 0.010, "and not so early it stops"


def test_an_empty_clamp_is_told_apart_from_a_held_book():
    """The check that ends a run rather than delivering an empty gripper.

    Held, the fingers stall on the book at a gap of about 30 mm. Empty, they reach the
    command, a gap of 5. The threshold is 10 mm inside the book's own thickness, which
    neither can reach from the wrong side.
    """
    empty = pad_gap(GRIPPER_CLAMP)
    held = BOOK_THICKNESS
    threshold = BOOK_THICKNESS - 0.010
    assert empty < threshold < held


def test_the_reopen_threshold_covers_the_arrival_tolerance():
    """The jaws must clear the book from the worst position we call on target."""
    assert pad_gap(GRIPPER_OPEN_MIN) >= BOOK_THICKNESS + 2 * ARRIVAL_TOL_LATERAL
    assert GRIPPER_OPEN > GRIPPER_OPEN_MIN, (
        "the open command has to leave margin for the finger being back-driven")
