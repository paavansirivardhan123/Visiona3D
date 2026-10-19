"""Test T1: Pinhole camera projection and backprojection round trip."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Intrinsics
from geometry.camera import backproject, project, ray_directions, range_to_depth, depth_to_range


def test_t1_project_backproject_roundtrip():
    """T1: Max error < 1e-6 for projection/backprojection round trip."""
    K = Intrinsics.from_fov(width=1280, height=720, hfov_deg=80.0)

    # 1. Test grid of pixels with varying depth map
    rng = np.random.default_rng(42)
    depth = rng.uniform(0.5, 10.0, size=(720, 1280))

    points_c = backproject(depth, K)
    assert points_c.shape == (720, 1280, 3)

    # Project back to pixels
    pixels = project(points_c, K)
    v_grid, u_grid = np.indices((720, 1280), dtype=np.float64)
    expected_pixels = np.stack([u_grid, v_grid], axis=-1)

    max_pixel_error = np.max(np.abs(pixels - expected_pixels))
    assert max_pixel_error < 1e-6, f"Pixel reprojection error {max_pixel_error} >= 1e-6"

    # 2. Test backprojecting the reprojected 3D points
    depth_recovered = points_c[..., 2]
    max_depth_error = np.max(np.abs(depth_recovered - depth))
    assert max_depth_error < 1e-6, f"Depth recovery error {max_depth_error} >= 1e-6"

    # 3. Test arbitrary 3D points in C
    points_rand = rng.uniform(-5.0, 5.0, size=(1000, 3))
    points_rand[:, 2] = rng.uniform(0.5, 20.0, size=1000)  # Positive z depth
    pixels_rand = project(points_rand, K)

    # Reconstruct 3D points using pixel coords and known z
    fx, fy, cx, cy = K.fx, K.fy, K.cx, K.cy
    u = pixels_rand[:, 0]
    v = pixels_rand[:, 1]
    z = points_rand[:, 2]
    x_rec = (u - cx) * z / fx
    y_rec = (v - cy) * z / fy
    points_rec = np.stack([x_rec, y_rec, z], axis=-1)

    max_3d_error = np.max(np.abs(points_rec - points_rand))
    assert max_3d_error < 1e-6, f"3D point recovery error {max_3d_error} >= 1e-6"


def test_ray_directions():
    """Verify ray directions are unit vectors pointing forward."""
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=70.0)
    rays = ray_directions(K, H=480, W=640)
    assert rays.shape == (480, 640, 3)

    norms = np.linalg.norm(rays, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert np.all(rays[..., 2] > 0.0)


# --- Conversion tests (D2) ---

def test_range_depth_roundtrip():
    """D2: range_to_depth and depth_to_range round-trip within 1e-9."""
    K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

    # Test at principal point
    ranges = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    us = np.full_like(ranges, 320.0)
    vs = np.full_like(ranges, 240.0)

    depths = range_to_depth(ranges, us, vs, K)
    ranges_back = depth_to_range(depths, us, vs, K)
    assert np.allclose(ranges, ranges_back, rtol=1e-9), f"Round-trip error: {np.max(np.abs(ranges - ranges_back))}"

    # Test off-axis
    us_off = np.array([220.0, 320.0, 420.0, 520.0])  # left, center, right, far right
    vs_off = np.full_like(us_off, 240.0)
    ranges_off = np.array([3.0, 3.0, 3.0, 3.0])
    depths_off = range_to_depth(ranges_off, us_off, vs_off, K)
    ranges_back_off = depth_to_range(depths_off, us_off, vs_off, K)
    assert np.allclose(ranges_off, ranges_back_off, rtol=1e-9), f"Off-axis round-trip error: {np.max(np.abs(ranges_off - ranges_back_off))}"


def test_range_depth_equal_at_principal_point():
    """D2: range == depth at principal point."""
    K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

    for r in [0.5, 1.0, 2.5, 5.0, 10.0, 50.0]:
        d = range_to_depth(r, 320.0, 240.0, K)
        assert np.isclose(d, r, rtol=1e-12), f"At principal point, range {r} != depth {d}"


def test_range_depth_off_axis_known_value():
    """D2: Known off-axis pixel matches hand-computed value."""
    K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)

    # u=420 (100px right), v=240 (center row)
    # dx = (420-320)/500 = 0.2, dy = 0, dz = 1
    # |r| = sqrt(0.2^2 + 0^2 + 1^2) = sqrt(1.04)
    # depth = range * (1 / sqrt(1.04))
    range_val = 5.0
    expected_depth = range_val / np.sqrt(1.04)
    computed_depth = range_to_depth(range_val, 420.0, 240.0, K)
    assert np.isclose(computed_depth, expected_depth, rtol=1e-9), (
        f"Off-axis depth {computed_depth} != expected {expected_depth}"
    )
