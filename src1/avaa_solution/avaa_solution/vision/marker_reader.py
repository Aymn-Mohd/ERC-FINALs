"""Overhead column-marker digit recognition.

Pure OpenCV, no ROS import, so it runs against saved frames without a simulator.

The markers are 30 x 30 cm plates at Z = 2.26 m carrying a printed digit 1-5, one per
shelf column, and the digit-to-column assignment is reshuffled on every simulation load
(``random.shuffle`` in ``erc_bringup/launch/simulation.launch.py``). So the digit must be
read from the camera; position tells you nothing.

Templates are the **exact PNG textures the simulator renders** -- they ship in
``erc_description/models/number_marker/textures/``. Matching against the real artwork
rather than a synthesized font removes a whole class of error, and needs no training data,
which matters because there is none: the competition provides no dataset.

Classification is binary-mask IoU after letterboxed normalisation. Letterboxing (rather
than stretching to a square) preserves the width-to-height ratio inside the mask, which is
what separates a "1" from the rest -- its texture is 88 px wide against 110-117 for the
others.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DIGITS = (1, 2, 3, 4, 5)

# Normalised canvas. 32 px is comfortably above the smallest marker seen in practice
# (11 x 14 px at ~3 m), so upsampling a distant marker does not invent detail.
CANVAS = 32
INK_HEIGHT = 28

# A marker is dark ink on a light plate, and unsaturated -- which is what separates it
# from the books, all four of which are strongly saturated.
MAX_VALUE = 120
MAX_SATURATION = 60

MIN_INK_AREA = 15
MIN_ASPECT = 0.6
MAX_ASPECT = 4.0

# Smallest credible marker, in pixels of digit height.
#
# The plate is 0.30 m and the camera has fx = 337, so a digit filling the plate spans
# roughly 101/d pixels at d metres. The arena is 10 x 10 m, so nothing real can be much
# under 10 px; below that the classifier is matching noise. This was not merely
# theoretical: an 8 px blob on a blank wall was read as a confident "4", which stopped the
# search dead facing away from the shelves and took the rest of the run down with it.
MIN_MARKER_HEIGHT_PX = 12

# Below this IoU the match is not trustworthy. Calibrated against measured readings: on a
# frame where all four visible digits were read correctly, IoU ranged 0.43-0.75, the worst
# being the nearest plate (viewed most obliquely, so most perspective-skewed). A threshold
# of 0.45 would have rejected a correct read, so it sits below that.
#
# There is deliberately no margin threshold. Margin is still reported, but the
# distinct-digit constraint in assign_distinct() is a far stronger guard than any
# runner-up gap: a correctly-read "5" measured a margin of only 0.042 because 5 and 3 look
# alike at 15 px, and rejecting it would have lost a correct answer.
# Tried at 0.30 on 2026-09-05 and put back, because it changed nothing.
#
# The reasoning below still holds and is kept: the scores a run produces really are lower
# than the frame this was calibrated on. But loosening it did not make the target marker
# identifiable -- still 0 of 50 frames, eight windows running -- which means the plate is
# not being scored as a 3 and only just missing. It is not being scored as a 3 at all.
# A guard that costs false positives and buys nothing goes back where it was.
#
# Perception now reports every digit it reads and its best score. Over eight windows of
# a run, sweeping for the target, it read 1, 2, 4 and 5 -- digit 4 twelve times in one
# window at scores up to 0.56 -- and read the digit it was looking for, 3, exactly once,
# at 0.31. The threshold was 0.35, so that one reading was thrown away and the approach
# swept a dozen full circles finding nothing.
#
# The 0.35 was calibrated on a frame where correct reads scored 0.43 to 0.75. The scores
# actually seen in a run are lower than that across the board -- correct-looking reads at
# 0.36, 0.38, 0.41, 0.42, 0.44, 0.46 -- because the calibration frame was closer and
# squarer than a robot hunting from four metres. A threshold set from one favourable
# frame was rejecting most of a run.
#
# The risk this guards against is real and is recorded above: an 8 px blob on a blank
# wall once read as a confident "4" and stopped a search dead. But that was a SIZE
# failure and MIN_MARKER_HEIGHT_PX now catches it independently, and assign_distinct()
# still forbids two plates claiming the same digit. This threshold is not the only guard
# and should not be set as though it were.
MIN_SCORE = 0.35

_TEMPLATE_CACHE: Optional[Dict[int, np.ndarray]] = None


@dataclass(frozen=True)
class Marker:
    digit: int
    score: float
    margin: float
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def confident(self) -> bool:
        return self.score >= MIN_SCORE


def _texture_dir() -> str:
    """Locate the shipped marker textures."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("erc_description")
        path = os.path.join(share, "models", "number_marker", "textures")
        if os.path.isdir(path):
            return path
    except Exception:  # noqa: BLE001 - fall through to the source-tree locations
        pass

    for path in (
        "/opt/erc_ws/src/erc_description/models/number_marker/textures",
        os.path.join(os.path.dirname(__file__), "templates"),
    ):
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "marker textures not found; expected erc_description/models/number_marker/textures"
    )


def _ink_mask(bgr_or_gray: np.ndarray) -> np.ndarray:
    """Binary mask of the dark ink, 255 where ink."""
    if bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_or_gray
    # Otsu adapts to the exposure difference between a crisp texture file and a
    # distant, dimly-lit render of the same plate.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _normalise(mask: np.ndarray) -> Optional[np.ndarray]:
    """Crop to the ink, scale to a fixed height, and letterbox into a square canvas."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        return None

    ink = mask[y:y + h, x:x + w]
    scale = INK_HEIGHT / float(h)
    new_w = max(1, min(CANVAS, int(round(w * scale))))
    ink = cv2.resize(ink, (new_w, INK_HEIGHT), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((CANVAS, CANVAS), np.uint8)
    y0 = (CANVAS - INK_HEIGHT) // 2
    x0 = (CANVAS - new_w) // 2
    canvas[y0:y0 + INK_HEIGHT, x0:x0 + new_w] = ink
    return canvas


def load_templates(directory: Optional[str] = None) -> Dict[int, np.ndarray]:
    """Normalised ink masks for digits 1-5, cached."""
    global _TEMPLATE_CACHE
    if directory is None and _TEMPLATE_CACHE is not None:
        return _TEMPLATE_CACHE

    directory = directory or _texture_dir()
    templates: Dict[int, np.ndarray] = {}
    for digit in DIGITS:
        path = os.path.join(directory, f"{digit}.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"missing marker texture: {path}")
        norm = _normalise(_ink_mask(img))
        if norm is None:
            raise ValueError(f"no ink found in template {path}")
        templates[digit] = norm

    if directory is None:
        _TEMPLATE_CACHE = templates
    else:
        _TEMPLATE_CACHE = _TEMPLATE_CACHE or templates
    return templates


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(cv2.bitwise_and(a, b))
    union = np.count_nonzero(cv2.bitwise_or(a, b))
    return inter / float(union) if union else 0.0


def score_all(crop: np.ndarray,
              templates: Optional[Dict[int, np.ndarray]] = None
              ) -> Optional[Dict[int, float]]:
    """Score one cropped marker by IoU against every digit template."""
    templates = templates or load_templates()
    norm = _normalise(_ink_mask(crop))
    if norm is None:
        return None
    return {digit: _iou(norm, tpl) for digit, tpl in templates.items()}


def classify(crop: np.ndarray,
             templates: Optional[Dict[int, np.ndarray]] = None
             ) -> Tuple[Optional[int], float, float]:
    """Return (digit, score, margin) for one cropped marker, judged on its own.

    ``margin`` is the gap to the runner-up: a high score with a low margin means the
    reading is ambiguous, which matters far more than raw score for digits that look
    alike at low resolution.

    Prefer :func:`read_markers`, which additionally applies the constraint that the five
    markers carry distinct digits.
    """
    scores = score_all(crop, templates)
    if scores is None:
        return None, 0.0, 0.0
    ranked = sorted(scores.items(), key=lambda t: t[1], reverse=True)
    best_digit, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return best_digit, best_score, best_score - runner_up


def assign_distinct(score_rows: List[Dict[int, float]]) -> List[int]:
    """Pick digits maximising total score, with no digit used twice.

    The shelf carries markers 1-5, each exactly once
    (``random.shuffle`` over ``range(1, 6)`` in the simulation launch file). Reading each
    marker independently throws that away and lets two markers claim the same digit.

    Enforcing it globally repairs exactly the failures seen in practice: the nearest plate
    is viewed most obliquely and its perspective-squashed digit scores poorly on its own,
    but there is only one digit left for it once the clearer markers are resolved.

    At most 5 candidates and 5 digits, so the 120 permutations are enumerated exactly --
    no need for scipy's Hungarian solver, and no extra dependency to declare.
    """
    from itertools import permutations

    if not score_rows:
        return []

    digits = list(DIGITS)
    k = min(len(score_rows), len(digits))
    best_total = float("-inf")
    best_choice: List[int] = []

    for combo in permutations(digits, k):
        total = sum(score_rows[i].get(combo[i], 0.0) for i in range(k))
        if total > best_total:
            best_total = total
            best_choice = list(combo)

    # More candidates than digits: pad so indices still line up.
    while len(best_choice) < len(score_rows):
        best_choice.append(-1)
    return best_choice


def find_marker_candidates(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes of dark, unsaturated, digit-shaped blobs."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark = (((hsv[:, :, 2] < MAX_VALUE) & (hsv[:, :, 1] < MAX_SATURATION))
            .astype(np.uint8) * 255)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < MIN_MARKER_HEIGHT_PX:
            continue
        if cv2.contourArea(contour) < MIN_INK_AREA:
            continue
        if not (MIN_ASPECT <= h / float(w) <= MAX_ASPECT):
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


def read_markers(bgr: np.ndarray, pad: int = 2,
                 distinct: bool = True) -> List[Marker]:
    """Find and read every column marker in the frame, left to right.

    With ``distinct`` (the default) the digits are assigned globally so no digit is used
    twice. Set it False to judge each marker independently, which is only useful for
    measuring how much the constraint is contributing.
    """
    templates = load_templates()
    height, width = bgr.shape[:2]

    boxes: List[Tuple[int, int, int, int]] = []
    rows: List[Dict[int, float]] = []
    for x, y, w, h in find_marker_candidates(bgr):
        crop = bgr[max(0, y - pad):min(height, y + h + pad),
                   max(0, x - pad):min(width, x + w + pad)]
        if crop.size == 0:
            continue
        scores = score_all(crop, templates)
        if scores is None:
            continue
        boxes.append((x, y, w, h))
        rows.append(scores)

    if not boxes:
        return []

    if distinct:
        chosen = assign_distinct(rows)
    else:
        chosen = [max(r.items(), key=lambda t: t[1])[0] for r in rows]

    markers: List[Marker] = []
    for (x, y, w, h), scores, digit in zip(boxes, rows, chosen):
        if digit < 0:
            continue
        score = scores.get(digit, 0.0)
        others = [s for d, s in scores.items() if d != digit]
        # Negative margin means the global constraint overrode this marker's own best
        # guess -- worth surfacing rather than hiding, since a run of them suggests the
        # candidate set is wrong (a spurious blob, or a marker missed entirely).
        margin = score - (max(others) if others else 0.0)
        markers.append(Marker(digit, score, margin, x, y, w, h))
    return markers


def column_of_digit(markers: List[Marker], digit: int) -> Optional[int]:
    """0-based left-to-right index of the marker showing ``digit``.

    Returns None when the digit is absent or the reading is not confident. The caller
    should move for a better view rather than act on a guess -- a wrong column sends the
    robot to the wrong shelf and forfeits the navigation and grasp points too.
    """
    hits = [i for i, m in enumerate(sorted(markers, key=lambda m: m.cx))
            if m.digit == digit and m.confident]
    return hits[0] if len(hits) == 1 else None
