"""The shelf fit must find the surface most returns agree on, not average them all.

Two real scenes broke a plain least-squares line through the forward cone.

Far from the shelf the cone reaches past the end of it to the wall 2.5 m beyond: 226
returns spanning x from 0.74 to 3.62 m, which pulled the fitted angle from -34 to -53
degrees and the residual to 0.26 m. The guard refused, and the approach then handed an
unsquared base to a grasp that reaches along base x. That run finished 35.8 degrees off
square with the target 15 mm from where it believed it was, and came away empty.

Close in, the opposite shape: a flat face at x = 0.85 across the whole cone with a shelf
divider sticking out to x = 0.55 in front of it. Those 46 points out of 207 held the
residual at 0.10 m while the robot was already square to within half a degree, so the
approach retreated and eventually gave up on a pose that was fine.

These drive the real fitting helper, on synthetic scans, so they need no simulator.
"""

import math

import numpy as np

import pytest

from avaa_solution.approach_node import _largest_collinear_set

FACE_TOLERANCE = 0.05
FACE_CONSENSUS = 0.6
MAX_RESIDUAL = 0.05


def fit(points):
    """Reproduce _shelf_angle: consensus surface, refit, then the credibility guard."""
    if len(points) < 12:
        return None
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    inliers = _largest_collinear_set(xs, ys, tolerance=FACE_TOLERANCE)
    if inliers is None:
        return None
    if int(inliers.sum()) < max(12, int(FACE_CONSENSUS * len(xs))):
        return None
    xs, ys = xs[inliers], ys[inliers]

    slope, intercept = np.polyfit(ys, xs, 1)
    if float(np.std(xs - (slope * ys + intercept))) > MAX_RESIDUAL:
        return None
    return float(math.atan(slope))


def face(yaw, distance=0.9, n=120, half_angle=0.45):
    """Build a flat wall of constant world x, seen by a robot yawed by yaw.

    In the robot frame such a wall is the line x = distance / cos(yaw) + tan(yaw) * y,
    which is exactly where the fit reads the yaw from. A ray at bearing b meets it at
    range distance / cos(b + yaw).
    """
    points = []
    for i in range(n):
        bearing = -half_angle + 2 * half_angle * i / (n - 1)
        denominator = math.cos(bearing + yaw)
        if denominator <= 1e-3:
            continue   # the wall is edge-on or behind at this bearing
        r = distance / denominator
        points.append((r * math.cos(bearing), r * math.sin(bearing)))
    return points


@pytest.mark.parametrize("yaw_deg", [0.0, -1.5, -10.0, -35.8, 12.0, 25.0])
def test_clean_face_recovers_the_yaw(yaw_deg):
    angle = fit(face(math.radians(yaw_deg)))
    assert angle is not None, "refused a clean face at %.1f deg" % yaw_deg
    assert abs(math.degrees(angle) - yaw_deg) < 3.0


def test_far_wall_beyond_the_shelf_is_ignored():
    """The far scene: a shelf face plus returns from a wall 2.5 m further out."""
    points = face(math.radians(-35.8))
    points += [(3.6 + 0.02 * i, -0.9 + 0.03 * i) for i in range(40)]
    angle = fit(points)
    assert angle is not None, "the far wall still defeats the fit"
    assert abs(math.degrees(angle) + 35.8) < 5.0


def test_divider_in_front_of_the_face_is_ignored():
    """The close scene, to scale: 207 returns of which 46 are a divider 0.3 m proud."""
    points = face(math.radians(-1.5), distance=0.85, n=161)
    points += [(0.55 + 0.004 * (i % 6), -0.11 + 0.001 * i) for i in range(46)]
    angle = fit(points)
    assert angle is not None, "the divider still defeats the fit"
    assert abs(math.degrees(angle) + 1.5) < 3.0


def test_a_protruding_book_does_not_drag_the_answer():
    points = face(math.radians(-8.0))
    points += [(0.55, 0.02 * i) for i in range(6)]   # one book pulled forward
    angle = fit(points)
    assert angle is not None
    assert abs(math.degrees(angle) + 8.0) < 4.0


def test_two_surfaces_of_similar_size_are_refused():
    """A corner is not something to square against; neither surface is the shelf."""
    points = [(0.9, -0.6 + 0.01 * i) for i in range(60)]              # facing us
    points += [(0.9 + 0.012 * i, 0.0 + 0.01 * i) for i in range(60)]  # at about 50 deg
    assert fit(points) is None


def test_too_few_returns_is_refused():
    assert fit([(0.9, 0.01 * i) for i in range(8)]) is None


def test_the_fit_is_repeatable():
    """Identical input must give an identical answer, or the controller cannot be read."""
    points = face(math.radians(-12.0))
    points += [(0.5, 0.01 * i) for i in range(20)]
    answers = {fit(points) for _ in range(5)}
    assert len(answers) == 1
