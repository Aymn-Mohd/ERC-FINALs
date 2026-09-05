"""Unit tests for turning a pixel box plus depth into a 3D point."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from avaa_solution.vision.depth_locator import (
    Intrinsics,
    deproject,
    locate,
    sample_depth,
    transform_point,
)

# The simulator's actual values.
INTR = Intrinsics(fx=337.2096, fy=337.2096, cx=320.0, cy=180.0)


def test_intrinsics_from_camera_info_k():
    k = [337.2096, 0.0, 320.0, 0.0, 337.2096, 180.0, 0.0, 0.0, 1.0]
    intr = Intrinsics.from_k(k)
    assert intr.fx == pytest.approx(337.2096)
    assert intr.cy == pytest.approx(180.0)


def test_principal_point_projects_straight_ahead():
    point = deproject(INTR.cx, INTR.cy, 2.0, INTR)
    assert np.allclose(point, [0.0, 0.0, 2.0])


def test_right_of_centre_gives_positive_x():
    # Optical frame: +x is to the right, +y is down.
    point = deproject(INTR.cx + 100, INTR.cy, 2.0, INTR)
    assert point[0] > 0
    assert point[2] == pytest.approx(2.0)


def test_below_centre_gives_positive_y():
    point = deproject(INTR.cx, INTR.cy + 100, 2.0, INTR)
    assert point[1] > 0


def test_offset_scales_with_depth():
    near = deproject(INTR.cx + 100, INTR.cy, 1.0, INTR)
    far = deproject(INTR.cx + 100, INTR.cy, 2.0, INTR)
    assert far[0] == pytest.approx(2 * near[0])


def test_sample_depth_takes_the_median_of_the_patch():
    img = np.full((100, 100), 2.0, dtype=np.float32)
    assert sample_depth(img, (40, 40, 20, 20)) == pytest.approx(2.0)


def test_sample_depth_ignores_edge_outliers():
    # Book at 1.5 m with the shelf gap behind it showing at the box edges. Shrinking to
    # the centre must keep the book's distance, not average in the background.
    img = np.full((100, 100), 3.0, dtype=np.float32)
    img[45:55, 45:55] = 1.5
    got = sample_depth(img, (44, 44, 12, 12), shrink=0.5)
    assert got == pytest.approx(1.5)


def test_sample_depth_rejects_zeros_and_nans():
    img = np.zeros((50, 50), dtype=np.float32)
    assert sample_depth(img, (10, 10, 20, 20)) is None
    img[:] = np.nan
    assert sample_depth(img, (10, 10, 20, 20)) is None


def test_sample_depth_rejects_out_of_range():
    img = np.full((50, 50), 50.0, dtype=np.float32)
    assert sample_depth(img, (10, 10, 20, 20), max_range=8.0) is None


def test_sample_depth_handles_a_tiny_box():
    # A distant book can be only a few pixels across; shrinking must not empty the patch.
    img = np.full((50, 50), 2.5, dtype=np.float32)
    assert sample_depth(img, (20, 20, 3, 5)) == pytest.approx(2.5)


def test_sample_depth_clips_a_box_off_the_image_edge():
    img = np.full((50, 50), 2.0, dtype=np.float32)
    assert sample_depth(img, (45, 45, 20, 20)) == pytest.approx(2.0)
    assert sample_depth(img, (100, 100, 10, 10)) is None


def test_sample_depth_rejects_a_degenerate_box():
    img = np.full((50, 50), 2.0, dtype=np.float32)
    assert sample_depth(img, (10, 10, 0, 10)) is None


def test_locate_uses_the_box_centre():
    img = np.full((360, 640), 1.8, dtype=np.float32)
    point = locate((316, 176, 8, 8), img, INTR)
    assert point is not None
    assert np.allclose(point, [0.0, 0.0, 1.8], atol=1e-3)


def test_locate_returns_none_without_depth():
    assert locate((10, 10, 8, 8), np.zeros((360, 640), np.float32), INTR) is None


def test_transform_identity_leaves_the_point_alone():
    rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    translation = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    assert np.allclose(transform_point([1, 2, 3], rotation, translation), [1, 2, 3])


def test_transform_applies_translation():
    rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    translation = SimpleNamespace(x=1.0, y=-2.0, z=0.5)
    assert np.allclose(transform_point([1, 2, 3], rotation, translation), [2, 0, 3.5])


def test_transform_applies_yaw():
    # 90 degrees about z takes +x onto +y.
    half = math.pi / 4
    rotation = SimpleNamespace(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
    translation = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    assert np.allclose(
        transform_point([1, 0, 0], rotation, translation), [0, 1, 0], atol=1e-9
    )
