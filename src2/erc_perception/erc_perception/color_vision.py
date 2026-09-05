from typing import Dict, Optional, Tuple

import cv2
import numpy as np

COLOUR_RANGES = {
    'red': [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))],
    'blue': [((100, 100, 50), (130, 255, 255))],
    'green': [((40, 60, 50), (85, 255, 255))],
    'yellow': [((20, 100, 80), (35, 255, 255))],
}
MIN_BOOK_AREA = 150
MAX_BOOK_AREA = 15000
MIN_BOOK_ASPECT = 1.0
MAX_BOOK_ASPECT = 6.0

RED_BIN_LOWER_1 = (0, 90, 60)
RED_BIN_UPPER_1 = (10, 255, 255)
RED_BIN_LOWER_2 = (170, 90, 60)
RED_BIN_UPPER_2 = (180, 255, 255)
MIN_RED_BIN_AREA = 5000
MAX_RED_BIN_ASPECT_H_OVER_W = 0.85


def _mask_for(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def is_looking_at_red_bin(frame: np.ndarray, min_area: float = MIN_RED_BIN_AREA,
                           max_aspect_h_over_w: float = MAX_RED_BIN_ASPECT_H_OVER_W) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _mask_for(hsv, [(RED_BIN_LOWER_1, RED_BIN_UPPER_1), (RED_BIN_LOWER_2, RED_BIN_UPPER_2)])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return False
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 0:
        return False
    aspect = h / float(w)
    return aspect <= max_aspect_h_over_w


def detect_colour_blob(frame: np.ndarray, colour: str,
                        min_area: float = MIN_BOOK_AREA,
                        max_area: float = MAX_BOOK_AREA,
                        roi_x_frac: float = 1.0,
                        min_aspect_h_over_w: float = MIN_BOOK_ASPECT,
                        max_aspect_h_over_w: float = MAX_BOOK_ASPECT
                        ) -> Optional[Tuple[float, float, int, int, int, int]]:
    if colour not in COLOUR_RANGES:
        return None
    h_img, w_img = frame.shape[:2]
    x_lo, x_hi = 0, w_img
    if roi_x_frac < 1.0:
        margin = int(w_img * (1.0 - roi_x_frac) / 2.0)
        x_lo, x_hi = margin, w_img - margin

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _mask_for(hsv, COLOUR_RANGES[colour])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2.0
        if not (x_lo <= cx <= x_hi):
            continue
        if w <= 0:
            continue
        aspect = h / float(w)
        if not (min_aspect_h_over_w <= aspect <= max_aspect_h_over_w):
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)

    if best is None:
        return None
    _, x, y, w, h = best
    cx, cy = x + w / 2.0, y + h / 2.0
    return cx, cy, x, y, w, h


def detect_all_colours(frame: np.ndarray, min_area: float = MIN_BOOK_AREA,
                        max_area: float = MAX_BOOK_AREA,
                        roi_x_frac: float = 1.0,
                        min_aspect_h_over_w: float = MIN_BOOK_ASPECT,
                        max_aspect_h_over_w: float = MAX_BOOK_ASPECT
                        ) -> Dict[str, Tuple[float, float, int, int, int, int]]:
    found = {}
    for colour in COLOUR_RANGES:
        hit = detect_colour_blob(frame, colour, min_area, max_area, roi_x_frac,
                                  min_aspect_h_over_w, max_aspect_h_over_w)
        if hit is not None:
            found[colour] = hit
    return found


def find_red_bin_centre(frame: np.ndarray) -> Optional[float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _mask_for(hsv, [(RED_BIN_LOWER_1, RED_BIN_UPPER_1), (RED_BIN_LOWER_2, RED_BIN_UPPER_2)])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return x + w / 2.0
