"""Tests for the row-to-height mapping and the gripper command values.

The row direction is the risky part: the rules number the stocked rows 1-4 without saying
which end row 1 is. Getting it backwards costs the identification points and sends the arm
to the wrong shelf, so both directions are pinned down here.
"""

import pytest

from avaa_solution.grasp_node import (
    BOOK_DEPTH,
    DEFAULT_ROW_HEIGHTS,
    DEFAULT_GRASP_DEPTH,
    GRASP_FRAME_TO_PAD_CENTER,
    GRIPPER_CLAMP,
    GRIPPER_OPEN,
    PREGRASP_TRIALS,
    REACH_STEPS,
    RIGHT_TUCK_POSE,
    TORSO_MAX,
    TORSO_MIN,
    TORSO_SEARCH_LEVELS,
    TUCK_POSE,
    TUCK_TORSO,
    row_to_height,
)

HEIGHTS = [1.501, 1.171, 0.841, 0.511]   # book centres, top shelf first


def test_defaults_are_ordered_top_shelf_first():
    assert DEFAULT_ROW_HEIGHTS == sorted(DEFAULT_ROW_HEIGHTS, reverse=True)


def test_rows_are_evenly_spaced():
    gaps = [a - b for a, b in zip(DEFAULT_ROW_HEIGHTS, DEFAULT_ROW_HEIGHTS[1:])]
    for gap in gaps:
        assert gap == pytest.approx(0.33, abs=0.005)


@pytest.mark.parametrize("row,expected", [(1, 1.501), (2, 1.171), (3, 0.841), (4, 0.511)])
def test_top_down_numbering(row, expected):
    assert row_to_height(row, HEIGHTS, top_down=True) == pytest.approx(expected)


@pytest.mark.parametrize("row,expected", [(1, 0.511), (2, 0.841), (3, 1.171), (4, 1.501)])
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
    complained, and MoveIt then would not plan from it at all. We now use the src1
    compact spherical-wrist home.
    """
    assert TUCK_POSE != [-0.5, -2.4, 0.0, -2.4, 0.0, 0.0, 0.0]
    assert len(TUCK_POSE) == 7
    assert TUCK_POSE == [0.36, -1.83, 0.47, -2.35, 0.0, -1.2, 0.0]


def test_approach_uses_the_same_collision_free_tuck():
    from avaa_solution.approach_node import (
        RIGHT_TUCK_POSE as APPROACH_RIGHT,
        TUCK_POSE as APPROACH_TUCK,
        TUCK_TORSO as APPROACH_TORSO,
    )
    assert APPROACH_TUCK == TUCK_POSE
    assert APPROACH_RIGHT == RIGHT_TUCK_POSE
    assert APPROACH_TORSO == TUCK_TORSO


def test_right_tuck_mirrors_left_joint_signs():
    """src1 right tuck is the signed mirror of the left (joints 1 and 3 flipped)."""
    mirrored = [
        -TUCK_POSE[0], TUCK_POSE[1], -TUCK_POSE[2],
        TUCK_POSE[3], TUCK_POSE[4], TUCK_POSE[5], TUCK_POSE[6],
    ]
    assert RIGHT_TUCK_POSE == mirrored
    assert len(RIGHT_TUCK_POSE) == 7


def test_tuck_fold_is_single_point():
    from avaa_solution.grasp_node import TUCK_TIME_SEC
    assert TUCK_TIME_SEC == 5.0
    assert RIGHT_TUCK_POSE == [-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0]


def test_grasp_search_uses_full_resolution_and_trial_budget():
    assert REACH_STEPS == 8
    assert PREGRASP_TRIALS == 24
    assert TORSO_MIN in TORSO_SEARCH_LEVELS
    assert (TORSO_MIN + TORSO_MAX) / 2.0 in TORSO_SEARCH_LEVELS
    assert TORSO_MAX in TORSO_SEARCH_LEVELS


def test_grasp_frame_places_pad_center_at_book_depth_center():
    assert DEFAULT_GRASP_DEPTH - GRASP_FRAME_TO_PAD_CENTER == pytest.approx(
        BOOK_DEPTH / 2.0)
