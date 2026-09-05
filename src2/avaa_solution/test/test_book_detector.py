"""Unit tests for the book detector.

Synthetic frames only -- no simulator needed, so these run in CI and catch regressions
in the shape gates that would otherwise only show up during a scored trial.
"""

import cv2
import numpy as np
import pytest

from avaa_solution.vision import book_detector as bd

BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 180, 0),
    "blue": (255, 0, 0),
}


def blank(w=640, h=360):
    # Mid-grey, like the shelves and walls: unsaturated, so it must not be detected.
    return np.full((h, w, 3), 200, np.uint8)


def put_book(img, x, y, colour, w=8, h=22):
    cv2.rectangle(img, (x, y), (x + w, y + h), BGR[colour], -1)


def test_blank_frame_finds_nothing():
    assert bd.detect_books(blank()) == []


def test_single_book_detected_with_colour():
    img = blank()
    put_book(img, 100, 100, "blue")
    books = bd.detect_books(img)
    assert len(books) == 1
    assert books[0].colour == "blue"


@pytest.mark.parametrize("colour", bd.COLOURS)
def test_every_colour_is_detected(colour):
    img = blank()
    put_book(img, 200, 120, colour)
    books = bd.detect_books(img)
    assert [b.colour for b in books] == [colour]


def test_large_red_blob_is_rejected_as_the_bin():
    # The collection bin is the same red as the red books. It measured 136x66 in the
    # simulation; anything that size must not be mistaken for a book.
    img = blank()
    cv2.rectangle(img, (400, 200), (536, 266), BGR["red"], -1)
    assert bd.detect_books(img) == []


def test_wide_green_patch_is_rejected_as_the_start_zone():
    # The start zone is the same green as the green books, but far wider than tall.
    img = blank()
    cv2.rectangle(img, (200, 300), (270, 336), BGR["green"], -1)
    assert bd.detect_books(img) == []


def test_columns_are_grouped_and_ordered_left_to_right():
    img = blank()
    for row, colour in enumerate(bd.COLOURS):
        put_book(img, 40, 100 + row * 30, colour)     # left column
        put_book(img, 300, 100 + row * 30, colour)    # right column
    columns = bd.group_into_columns(bd.detect_books(img))
    assert len(columns) == 2
    assert all(len(c) == bd.ROWS_PER_COLUMN for c in columns)
    assert columns[0][0].cx < columns[1][0].cx


def test_row_numbering_runs_top_down():
    img = blank()
    order = ["red", "green", "blue", "yellow"]  # top to bottom
    for row, colour in enumerate(order):
        put_book(img, 100, 60 + row * 40, colour)
    column = bd.group_into_columns(bd.detect_books(img))[0]
    for expected_row, colour in enumerate(order, start=1):
        assert bd.row_of(column, colour) == expected_row


def test_row_is_none_when_a_book_is_missing():
    # Three visible books are ambiguous: a missing top book shifts every row by one.
    # Reporting nothing is correct; reporting a guess costs points.
    img = blank()
    for row, colour in enumerate(["red", "green", "blue"]):
        put_book(img, 100, 60 + row * 40, colour)
    column = bd.group_into_columns(bd.detect_books(img))[0]
    assert bd.row_of(column, "green") is None


def test_row_is_none_for_a_colour_not_present():
    img = blank()
    for row, colour in enumerate(bd.COLOURS):
        put_book(img, 100, 60 + row * 40, colour)
    column = bd.group_into_columns(bd.detect_books(img))[0]
    assert bd.row_of(column, "magenta") is None


def test_find_book_returns_the_right_one():
    img = blank()
    for row, colour in enumerate(bd.COLOURS):
        put_book(img, 100, 60 + row * 40, colour)
        put_book(img, 400, 60 + row * 40, colour)
    columns = bd.group_into_columns(bd.detect_books(img))
    book = bd.find_book(columns, 1, "yellow")
    assert book is not None and book.colour == "yellow"
    assert book.cx > 300
    assert bd.find_book(columns, 5, "yellow") is None


def test_annotate_does_not_modify_the_input():
    img = blank()
    put_book(img, 100, 100, "red")
    books = bd.detect_books(img)
    before = img.copy()
    bd.annotate(img, books, highlight=books[0], caption="test")
    assert np.array_equal(img, before)
