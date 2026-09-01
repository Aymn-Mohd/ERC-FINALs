"""Collection-bin detection by colour and shape.

Pure OpenCV with no ROS dependency, matching ``book_detector.py``'s split between vision
logic and the ROS wrapper. Adapted from the ``erc_perception`` package's
``color_vision.py`` (the ``is_looking_at_red_bin`` / ``find_red_bin_centre`` pair), which
had no equivalent here -- ``avaa_solution`` can identify and grasp a book but has nothing
yet for the "place in bin" step.

The bin is the same red as red books (see ``book_detector.py``'s note on this), so this
gates on both colour and aspect ratio: the bin's bounding box is wide and short
(``h/w <= MAX_ASPECT_H_OVER_W``), where a book standing on a shelf is narrow and tall.
Not wired into a node yet -- that is follow-up work, tracked in STATE.md.
"""

from typing import Optional

import cv2
import numpy as np

RED_LOWER_1 = (0, 90, 60)
RED_UPPER_1 = (10, 255, 255)
RED_LOWER_2 = (170, 90, 60)
RED_UPPER_2 = (180, 255, 255)

MIN_AREA = 5000
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


def find_bin_centre_x(frame: np.ndarray, min_area: float = 200.0) -> Optional[float]:
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
