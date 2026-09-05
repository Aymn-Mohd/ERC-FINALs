"""Turn a detected book's pixel box into a 3D point.

Pure numpy, no ROS, so it is testable without a simulator.

The RGB and depth streams share intrinsics exactly (fx = fy = 337.2096, cx = 320,
cy = 180, both 640x360), so a bounding box found in the colour image indexes the depth
image directly with no reprojection.

Depth arrives as 32FC1 in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, k) -> "Intrinsics":
        """From a CameraInfo k matrix (row-major 3x3)."""
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]))


def deproject(u: float, v: float, depth: float, intr: Intrinsics) -> np.ndarray:
    """Pixel plus depth to a point in the camera OPTICAL frame.

    Optical frame convention: +Z forward along the view axis, +X right, +Y down. This is
    not the robot's convention and the difference matters -- the transform to base_link
    must come from TF, not from an assumption about which axis points where.
    """
    z = float(depth)
    x = (float(u) - intr.cx) * z / intr.fx
    y = (float(v) - intr.cy) * z / intr.fy
    return np.array([x, y, z])


def sample_depth(depth_image: np.ndarray,
                 bbox: Tuple[int, int, int, int],
                 shrink: float = 0.4,
                 min_valid: int = 4,
                 max_range: float = 8.0) -> Optional[float]:
    """Measure a robust depth for one box, or None if there is too little signal.

    Takes the median over a shrunk central patch rather than the single centre pixel.
    Books are thin and stand against an open shelf, so a box edge often straddles the gap
    behind the book; a lone centre pixel that happens to land on the gap would place the
    book most of a metre too far away. Shrinking to the middle and taking a median makes
    that a minority vote instead of the answer.

    Zeros, negatives, NaN and infinity are all treated as missing, which is how the
    various depth pipelines signal "no return".
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    height, width = depth_image.shape[:2]

    def clip(x0, x1, y0, y1):
        return (max(0, x0), min(width, x1), max(0, y0), min(height, y1))

    inset_x = int(w * (1.0 - shrink) / 2.0)
    inset_y = int(h * (1.0 - shrink) / 2.0)
    x0, x1, y0, y1 = clip(x + inset_x, x + w - inset_x, y + inset_y, y + h - inset_y)

    # Fall back to the full box when the shrunk patch is empty. That happens two ways: a
    # box only a few pixels across shrinks to nothing, and a box straddling the image edge
    # can have its centre off-image entirely while part of the full box is still visible.
    # Clipping has to happen before the check, or the second case is missed.
    if x1 <= x0 or y1 <= y0:
        x0, x1, y0, y1 = clip(x, x + w, y, y + h)
    if x1 <= x0 or y1 <= y0:
        return None

    patch = np.asarray(depth_image[y0:y1, x0:x1], dtype=float).ravel()
    valid = patch[np.isfinite(patch) & (patch > 0.0) & (patch < max_range)]
    if valid.size < min_valid:
        return None
    return float(np.median(valid))


def locate(bbox: Tuple[int, int, int, int],
           depth_image: np.ndarray,
           intr: Intrinsics,
           **kwargs) -> Optional[np.ndarray]:
    """Centre of a bounding box as a 3D point in the camera optical frame."""
    depth = sample_depth(depth_image, bbox, **kwargs)
    if depth is None:
        return None
    x, y, w, h = bbox
    return deproject(x + w / 2.0, y + h / 2.0, depth, intr)


def transform_point(point: np.ndarray, rotation, translation) -> np.ndarray:
    """Apply a ROS transform (quaternion + translation) to a point.

    ``rotation`` needs .x .y .z .w and ``translation`` .x .y .z, matching
    geometry_msgs/Transform, so a TF lookup can be passed straight in.
    """
    qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    r = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])
    return r @ np.asarray(point, dtype=float) + np.array(
        [translation.x, translation.y, translation.z], dtype=float
    )
