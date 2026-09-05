"""Fit the shelf's front face from the depth image.

The instrument the shelf is actually visible to.

The front laser cannot see it: it sits 209 mm off the floor, and at that height the shelf
is an open compartment, so the beam passes through and lands on the far wall. Measured at
0.74 m from the shelf front, 163 returns in the forward cone and not one inside two
metres. The depth camera has the opposite property -- it looks at the unit at book height,
where it is solid, and perception already trusts it enough to place books to 15-35 mm.

What this returns:

    distance   perpendicular distance from base_link to the fitted plane
    yaw        the base's yaw error against it: 0 when square on, positive yawed CCW

USE THE YAW. Treat the distance as diagnostic only.

Measured against Gazebo over six samples from one pose, with about 2300 points in the
fit each time, the yaw came out -34.5, -34.5, -35.3, -34.7, -34.2, -34.5 degrees against
a true -35.9: better than a degree and a half, and steady. That is the number the
approach has been getting badly wrong -- one run reached the standoff 38.9 degrees off
square, at which angle the camera looks along the shelf instead of at it.

The distance is a different story. At book height most of the band is open shelf, so the
biggest plane in view is the BACK panel about 0.35 m behind the books, and that is what
the fit locks onto: it reported 1.28 m where the front was 0.97 m away. The same open-
shelf problem that makes the shelf invisible to the laser makes its front face a minority
surface to the camera.

It does not matter for the yaw, and that is the point worth keeping: the back panel is
PARALLEL to the front, so a fit to either measures the same heading. Distance still comes
from the book's own depth, which perception already places to 15-35 mm and which measures
the thing the arm is actually reaching for.

Pure numpy on the depth image, no ROS and no point-cloud pipeline, so it works with the
simulation's point cloud disabled (which is how it is run, because the cloud costs most
of the frame rate).

Method, and why this one. The face is a plane at roughly constant depth across the image,
and the things that are not the face -- the openings, the books standing proud of it, the
floor and the far wall seen through it -- are all either much further away or a small part
of the picture. So: take a horizontal band of the image at the height of the row being
worked on, convert it to points in base_link, and fit a line in the horizontal plane by
consensus. A line, not a plane: the base can only translate and rotate on the floor, so
the vertical extent of the face carries no information the controller can act on, and
collapsing it removes the shelf's own boards from the problem.
"""

import math
from typing import Optional, Tuple

import numpy as np

# Fraction of the sampled points that must agree before a surface is believed to be the
# face. Deliberately low: the openings mean a good fraction of any band is looking at
# something else entirely, and requiring a majority is how the laser fit ended up
# preferring the far wall.
MIN_CONSENSUS = 0.25
MIN_POINTS = 40

# How far a point may sit from the fitted line and still count as on the face. The books
# stand about 30 mm proud of it and should NOT be rejected -- they are what the robot is
# aiming at -- so this is wide enough to include them and far short of the 0.35 m to the
# back panel.
INLIER_TOLERANCE = 0.06

# Plausible range to the shelf. Anything outside is the far wall or a mis-fit, and
# returning None is better than returning it.
MIN_RANGE = 0.30
MAX_RANGE = 4.00


def _fit_line(ys: np.ndarray, xs: np.ndarray, tolerance: float,
              iterations: int = 80) -> Optional[np.ndarray]:
    """Boolean mask of the largest set of points lying on one line, x = m*y + c."""
    count = len(xs)
    if count < 4:
        return None
    rng = np.random.default_rng(17)
    best = None
    for _ in range(iterations):
        a, b = rng.choice(count, size=2, replace=False)
        if abs(ys[a] - ys[b]) < 1e-6:
            continue
        slope = (xs[a] - xs[b]) / (ys[a] - ys[b])
        intercept = xs[a] - slope * ys[a]
        distance = (np.abs(xs - (slope * ys + intercept))
                    / math.sqrt(1.0 + slope * slope))
        inliers = distance < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    return best


def face_from_depth(depth: np.ndarray, intrinsics, rotation: np.ndarray,
                    translation: np.ndarray, row_height: float,
                    band: float = 0.20, stride: int = 4
                    ) -> Optional[Tuple[float, float, int]]:
    """Fit the shelf face. Returns (distance, yaw, points used) or None.

    ``rotation`` and ``translation`` place the depth optical frame in base_link.
    ``row_height`` is the height in base_link of the row being worked on, and ``band``
    how much of the shelf above and below it to include -- a slice rather than the whole
    image, so that the floor, the ceiling and the shelf's own boards stay out of the fit.
    """
    if depth is None or intrinsics is None:
        return None
    height, width = depth.shape[:2]

    # Every valid pixel, subsampled. The stride is for speed: a quarter of the columns
    # is still hundreds of points on a face this size.
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    zs = depth[0:height:stride, 0:width:stride].astype(np.float64)
    good = np.isfinite(zs) & (zs > 0.05) & (zs < MAX_RANGE + 1.0)
    if int(good.sum()) < MIN_POINTS:
        return None
    us, vs, zs = us[good], vs[good], zs[good]

    # Deproject, then into base_link.
    x_cam = (us - intrinsics.cx) * zs / intrinsics.fx
    y_cam = (vs - intrinsics.cy) * zs / intrinsics.fy
    points = np.vstack((x_cam, y_cam, zs))
    base = rotation @ points + translation.reshape(3, 1)

    # Keep the slice around the row of interest.
    keep = np.abs(base[2] - row_height) <= band
    if int(keep.sum()) < MIN_POINTS:
        return None
    xs, ys = base[0][keep], base[1][keep]

    floor = max(MIN_POINTS, int(MIN_CONSENSUS * len(xs)))
    remaining = np.ones(len(xs), dtype=bool)
    best = None

    # Peel surfaces off and keep the NEAREST credible one, for the same reason the laser
    # version did: through the openings there is a wall behind the shelf, and it is
    # parallel to it, and it is not what the robot should dock against.
    for _ in range(3):
        if int(remaining.sum()) < floor:
            break
        sub_y, sub_x = ys[remaining], xs[remaining]
        inliers = _fit_line(sub_y, sub_x, INLIER_TOLERANCE)
        if inliers is None or int(inliers.sum()) < floor:
            break
        face_y, face_x = sub_y[inliers], sub_x[inliers]
        slope, intercept = np.polyfit(face_y, face_x, 1)
        spread = float(np.std(face_x - (slope * face_y + intercept)))
        if spread <= INLIER_TOLERANCE:
            distance = abs(float(intercept)) / math.sqrt(1.0 + slope * slope)
            if MIN_RANGE <= distance <= MAX_RANGE:
                if best is None or distance < best[0]:
                    best = (distance, float(math.atan(slope)), int(inliers.sum()))
        index = np.where(remaining)[0]
        remaining[index[inliers]] = False

    return best
