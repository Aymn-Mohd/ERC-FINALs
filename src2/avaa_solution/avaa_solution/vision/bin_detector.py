"""Collection-bin detection by colour and shape.

Pure OpenCV with no ROS dependency, matching ``book_detector.py``'s split between vision
logic and the ROS wrapper. Adapted from the ``erc_perception`` package's
``color_vision.py`` (the ``is_looking_at_red_bin`` / ``find_red_bin_centre`` pair), which
had no equivalent here -- ``avaa_solution`` can identify and grasp a book but has nothing
yet for the "place in bin" step.

The bin is the same red as red books (see ``book_detector.py``'s note on this), so this
gates on both colour and aspect ratio: the bin's bounding box is wide and short
(``h/w <= MAX_ASPECT_H_OVER_W``), where a book standing on a shelf is narrow and tall.
The aspect gate is the real discriminator; area only rejects noise.

MIN_AREA started at 5000 (erc_perception's original value, tuned for a bin filling much
of the frame at close range) and was too strict for the bin_searching state, which rotates
in place from several metres away: across more than two full rotations the blob never
reached that area and the bin was never found. Lowered here to match find_bin_centre_x's
own threshold -- both functions describe the same blob and had no reason to disagree.
Still not independently measured against the sim at range; if search keeps missing the
bin, this is the first number to revisit, with real frames rather than another guess.
"""

from typing import Optional

import cv2
import numpy as np

RED_LOWER_1 = (0, 90, 60)
RED_UPPER_1 = (10, 255, 255)
RED_LOWER_2 = (170, 90, 60)
RED_UPPER_2 = (180, 255, 255)

MIN_AREA = 200
MAX_ASPECT_H_OVER_W = 0.85


def _red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(RED_LOWER_1), np.array(RED_UPPER_1))
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(RED_LOWER_2), np.array(RED_UPPER_2)))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def is_looking_at_bin(
    frame: np.ndarray,
    min_area: float = MIN_AREA,
    max_aspect_h_over_w: float = MAX_ASPECT_H_OVER_W,
) -> bool:
    """Report whether the largest red blob in frame is wide-and-short like the bin."""
    mask = _red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return False
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 0:
        return False
    return (h / float(w)) <= max_aspect_h_over_w


def find_bin_centre_x(frame: np.ndarray, min_area: float = MIN_AREA) -> Optional[float]:
    """Pixel x-coordinate of the largest red blob's centre, for centring the base on it."""
    mask = _red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return x + w / 2.0
