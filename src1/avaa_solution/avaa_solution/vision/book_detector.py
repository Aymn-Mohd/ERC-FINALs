"""Book detection by colour and shape.

Pure OpenCV with no ROS dependency, so it can be exercised on saved frames without a
running simulator. The ROS wrapper lives in ``perception_node.py``.

The approach and the numbers here are measured against the simulation, not assumed --
see PERCEPTION.md. Two findings drive the design:

1. The four book colours form four isolated hue clusters with wide gaps between them,
   so HSV thresholding is sufficient and a learned classifier would be wasted effort.
   That matters for Phase 2, where the same code runs on the robot's i5 with no GPU.

2. Colour alone is *not* enough. The collection bin is the same red as the red books and
   the start zone the same green as the green books, and both are in view whenever the
   robot faces the shelves. Shape separates them.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

COLOURS = ("red", "blue", "green", "yellow")

# Measured hue clusters (OpenCV H is 0-179), widened for shading and viewing angle.
# Red wraps around zero so it needs two ranges.
HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
COLOUR_RANGES: Dict[str, List[HsvRange]] = {
    "red": [((0, 90, 60), (12, 255, 255)), ((168, 90, 60), (179, 255, 255))],
    "yellow": [((20, 90, 60), (42, 255, 255))],
    # Green books measure S~122, markedly less saturated than the other three at S=255,
    # so this range needs a lower saturation floor.
    "green": [((48, 60, 60), (85, 255, 255))],
    "blue": [((105, 90, 60), (135, 255, 255))],
}

DRAW_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 200, 255),
    "green": (0, 200, 0),
    "blue": (255, 0, 0),
}

# Shape gates. Books are 25 x 16 x 3 cm shelved spine-out, so they always present an
# upright face. Measured: 6-8 px wide, 14-26 px tall, aspect 1.75-4.33, fill 0.78-0.83.
# The bin measured 136x66 (area 6763) and start-zone patches measured aspect 0.54/0.56.
MIN_AREA = 30
MAX_AREA = 3000
MIN_ASPECT = 1.2
MIN_FILL = 0.55

ROWS_PER_COLUMN = 4

# Bin gates, the mirror image of the book gates above.
#
# The bin is the same red as a red book -- the module docstring says so, and it is the
# reason the shape gates exist at all -- so the only thing separating them is size and
# proportion. A book presents an upright face 6-8 px wide by 14-26 px tall; the bin
# measured 136 x 66, which is wider than it is tall and two hundred times the area.
# There is no overlap to argue about, so the gates sit an order of magnitude apart on
# both counts rather than being tuned to a boundary.
BIN_MIN_AREA = 1500
BIN_MAX_ASPECT = 0.9
BIN_MIN_FILL = 0.45


@dataclass(frozen=True)
class Book:
    colour: str
    x: int
    y: int
    w: int
    h: int
    area: float
    aspect: float
    fill: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def _mask_for(hsv: np.ndarray, colour: str) -> np.ndarray:
    mask = None
    for lo, hi in COLOUR_RANGES[colour]:
        part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def detect_books(bgr: np.ndarray) -> List[Book]:
    """Return every book-shaped coloured blob, left to right."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    found: List[Book] = []

    for colour in COLOURS:
        contours, _ = cv2.findContours(
            _mask_for(hsv, colour), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
            area = cv2.contourArea(contour)
            aspect = h / float(w)
            fill = area / float(w * h)
            if not (MIN_AREA <= area <= MAX_AREA):
                continue
            if aspect < MIN_ASPECT or fill < MIN_FILL:
                continue
            found.append(Book(colour, x, y, w, h, area, aspect, fill))

    found.sort(key=lambda b: b.cx)
    return found


def detect_bin(bgr: np.ndarray) -> Optional[Book]:
    """Return the collection bin, or None if it is not in view.

    The bin is returned as a ``Book`` with colour "red" so that everything downstream --
    the depth locator, the annotator, the tests -- takes it without a second code path.
    It is not a book and nothing treats it as one: only ``detect_bin`` produces it, and
    only the delivery controller asks.

    The largest qualifying blob wins rather than the first. A red book on the shelf can
    never pass the area gate, but the start zone and a red book seen close up are both
    red things of some size, and picking the biggest is the difference between aiming at
    the bin and aiming at whatever happened to be found first.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    contours, _ = cv2.findContours(
        _mask_for(hsv, "red"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[Book] = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        area = cv2.contourArea(contour)
        aspect = h / float(w)
        fill = area / float(w * h)
        if area < BIN_MIN_AREA or aspect > BIN_MAX_ASPECT or fill < BIN_MIN_FILL:
            continue
        if best is None or area > best.area:
            best = Book("red", x, y, w, h, area, aspect, fill)
    return best


def group_into_columns(books: Sequence[Book]) -> List[List[Book]]:
    """Cluster books into shelf columns, left to right; each column top to bottom.

    Books in one column share an x position to within a few book widths, while the gap
    between columns is an order of magnitude larger (measured: ~24 px spread within a
    column against an ~89 px gap between them). Splitting a 1-D sort on a gap threshold
    scaled to the median book width handles the perspective change across the frame.
    """
    if not books:
        return []

    ordered = sorted(books, key=lambda b: b.cx)
    median_w = float(np.median([b.w for b in ordered]))
    gap_threshold = max(40.0, 6.0 * median_w)

    columns: List[List[Book]] = [[ordered[0]]]
    for book in ordered[1:]:
        if book.cx - columns[-1][-1].cx > gap_threshold:
            columns.append([book])
        else:
            columns[-1].append(book)

    for column in columns:
        column.sort(key=lambda b: b.cy)
    return columns


def group_by_anchors(books: Sequence[Book], anchor_xs: Sequence[float],
                     max_dx: Optional[float] = None) -> List[List[Book]]:
    """Assign each book to the nearest anchor x, one list per anchor.

    Preferred over :func:`group_into_columns` whenever the column markers are visible.
    Gap-based clustering has to guess where one column ends and the next begins, and that
    guess is scale-dependent: at an oblique viewing angle two adjacent columns can sit
    closer together in the image than the books within a single column do at close range,
    which silently merges them into one group of eight.

    The markers remove the guess. Each marker sits above exactly one column, so the anchor
    positions *are* the column positions, and the returned index lines up with the marker
    order.
    """
    columns: List[List[Book]] = [[] for _ in anchor_xs]
    if not anchor_xs:
        return columns

    if max_dx is None:
        if len(anchor_xs) >= 2:
            spacings = [abs(b - a) for a, b in zip(sorted(anchor_xs), sorted(anchor_xs)[1:])]
            max_dx = 0.5 * float(np.median(spacings))
        else:
            # A single anchor gives no spacing to measure. An unbounded radius would sweep
            # every book in the frame into that one column -- observed as
            # "column 0 shows 12 of 4 books" once the robot drove close enough that only
            # one marker remained in view. The caller should pass a scale-derived max_dx
            # (see column_max_dx); this is only a last resort.
            max_dx = float("inf")

    for book in books:
        index = min(range(len(anchor_xs)), key=lambda i: abs(book.cx - anchor_xs[i]))
        if abs(book.cx - anchor_xs[index]) <= max_dx:
            columns[index].append(book)

    for column in columns:
        column.sort(key=lambda b: b.cy)
    return columns


def row_of(column: Sequence[Book], colour: str) -> Optional[int]:
    """Row (1-based, counting from the top) of the given colour within one column.

    Returns None when the row cannot be established with confidence. Rows are numbered
    top-down: measured shelf heights run 1.577 m for the top stocked row down to 0.587 m
    for the bottom, 0.33 m apart (PERCEPTION.md section 4).

    Requires all four books in the column to be visible. With only three the mapping from
    sorted position to row number is ambiguous -- a missing top book shifts every row by
    one -- and reporting a wrong row costs points and sends the arm to the wrong shelf.
    Returning None lets the caller move for a better view instead.
    """
    if len(column) != ROWS_PER_COLUMN:
        return None
    for index, book in enumerate(sorted(column, key=lambda b: b.cy)):
        if book.colour == colour:
            return index + 1
    return None


def find_book(columns: Sequence[Sequence[Book]], column_index: int,
              colour: str) -> Optional[Book]:
    """Return the book of ``colour`` in the given 0-based column, or None."""
    if not 0 <= column_index < len(columns):
        return None
    for book in columns[column_index]:
        if book.colour == colour:
            return book
    return None


def annotate(bgr: np.ndarray, books: Sequence[Book],
             highlight: Optional[Book] = None,
             caption: Optional[str] = None) -> np.ndarray:
    """Draw detections. ``highlight`` gets a thick box; everything else a thin one."""
    vis = bgr.copy()
    for book in books:
        cv2.rectangle(vis, (book.x, book.y), (book.x + book.w, book.y + book.h),
                      DRAW_BGR[book.colour], 1)
    if highlight is not None:
        pad = 4
        cv2.rectangle(
            vis,
            (highlight.x - pad, highlight.y - pad),
            (highlight.x + highlight.w + pad, highlight.y + highlight.h + pad),
            (255, 255, 255), 2,
        )
    if caption:
        cv2.putText(vis, caption, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, caption, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis
