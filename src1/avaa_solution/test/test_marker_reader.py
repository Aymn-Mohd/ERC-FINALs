"""Unit tests for the column-marker digit reader.

Scenes are built from the simulator's own marker textures pasted onto a grey canvas, so
no simulator is needed but the artwork under test is the real artwork.
"""

import os

import cv2
import numpy as np
import pytest

from avaa_solution.vision import marker_reader as mr


@pytest.fixture(scope="module")
def textures():
    directory = mr._texture_dir()
    out = {}
    for digit in mr.DIGITS:
        img = cv2.imread(os.path.join(directory, f"{digit}.png"), cv2.IMREAD_COLOR)
        assert img is not None, f"missing texture for {digit}"
        out[digit] = img
    return out


def scene(digits, textures, height=26, gap=40, canvas=(200, 640)):
    """Grey canvas with the given digits pasted left to right at a realistic size."""
    img = np.full((canvas[0], canvas[1], 3), 210, np.uint8)
    x = 20
    for digit in digits:
        tex = textures[digit]
        scale = height / float(tex.shape[0])
        small = cv2.resize(tex, (max(1, int(tex.shape[1] * scale)), height),
                           interpolation=cv2.INTER_AREA)
        img[30:30 + height, x:x + small.shape[1]] = small
        x += small.shape[1] + gap
    return img


def test_templates_load_for_every_digit():
    templates = mr.load_templates()
    assert set(templates) == set(mr.DIGITS)
    for tpl in templates.values():
        assert tpl.shape == (mr.CANVAS, mr.CANVAS)
        assert tpl.max() == 255


@pytest.mark.parametrize("digit", mr.DIGITS)
def test_each_digit_is_read_correctly(digit, textures):
    img = scene([digit], textures)
    markers = mr.read_markers(img)
    assert len(markers) == 1
    assert markers[0].digit == digit
    assert markers[0].confident


def test_full_row_of_five_is_read_left_to_right(textures):
    order = [3, 1, 5, 2, 4]
    markers = sorted(mr.read_markers(scene(order, textures)), key=lambda m: m.cx)
    assert [m.digit for m in markers] == order


def test_blank_scene_yields_nothing():
    assert mr.read_markers(np.full((200, 640, 3), 210, np.uint8)) == []


def test_tiny_dark_blobs_are_not_read_as_digits(textures):
    # A 0.30 m plate at 8 px implies about 12 m, past the arena diagonal. An 8 px blob on
    # a blank wall was once read as a confident "4", which stopped the search facing away
    # from the shelves and took the whole run with it.
    img = scene([4], textures, height=8)
    assert mr.read_markers(img) == []


def test_a_marker_at_the_minimum_size_is_still_read(textures):
    # The floor must reject noise without discarding genuinely distant markers.
    img = scene([4], textures, height=mr.MIN_MARKER_HEIGHT_PX + 2)
    markers = mr.read_markers(img)
    assert len(markers) == 1
    assert markers[0].digit == 4


def test_saturated_books_are_not_mistaken_for_markers():
    # Books are strongly saturated; markers are dark and achromatic.
    img = np.full((200, 640, 3), 210, np.uint8)
    cv2.rectangle(img, (100, 60), (108, 82), (0, 0, 255), -1)
    cv2.rectangle(img, (200, 60), (208, 82), (255, 0, 0), -1)
    assert mr.read_markers(img) == []


def test_assign_distinct_never_repeats_a_digit():
    # Two candidates that both look most like a 3; only one may be assigned 3.
    rows = [
        {1: 0.10, 2: 0.20, 3: 0.90, 4: 0.10, 5: 0.55},
        {1: 0.10, 2: 0.20, 3: 0.80, 4: 0.10, 5: 0.60},
    ]
    chosen = mr.assign_distinct(rows)
    assert len(set(chosen)) == len(chosen)
    assert 3 in chosen


def test_assign_distinct_maximises_total_not_each_row():
    # Greedy per-row would take 3 for the first row and leave the second worse off.
    rows = [
        {1: 0.00, 2: 0.00, 3: 0.60, 4: 0.00, 5: 0.55},
        {1: 0.00, 2: 0.00, 3: 0.95, 4: 0.00, 5: 0.10},
    ]
    assert mr.assign_distinct(rows) == [5, 3]


def test_assign_distinct_handles_empty_and_oversized_input():
    assert mr.assign_distinct([]) == []
    rows = [{d: 0.5 for d in mr.DIGITS} for _ in range(7)]
    chosen = mr.assign_distinct(rows)
    assert len(chosen) == 7
    assigned = [c for c in chosen if c >= 0]
    assert len(assigned) == len(set(assigned)) == 5


def test_column_of_digit_finds_the_right_index(textures):
    order = [3, 1, 5, 2, 4]
    markers = mr.read_markers(scene(order, textures))
    for expected_index, digit in enumerate(order):
        assert mr.column_of_digit(markers, digit) == expected_index


def test_column_of_digit_returns_none_when_absent(textures):
    markers = mr.read_markers(scene([1, 2], textures))
    assert mr.column_of_digit(markers, 5) is None
