"""Test T2: Ground frame transformations and floor/wall coordinate recovery."""

from __future__ import annotations

import math
import numpy as np
import pytest

from geometry.frames import camera_to_ground, ground_to_camera


@pytest.mark.parametrize("pitch_deg", [0.0, 10.0, 20.0])
def test_t2_frames_floor_and_wall(pitch_deg: float):
    """T2: Floor points give Z within 1e-6 of 0; wall point at Y=3 gives Y within 1e-6."""
    h = 1.4
    theta_rad = math.radians(pitch_deg)
    n = np.array([0.0, math.cos(theta_rad), math.sin(theta_rad)], dtype=np.float64)

    rng = np.random.default_rng(123)

    # 1. Floor points: points in C satisfying n.p = h
    # Since n = [0, cos(theta), sin(theta)], ny * y + nz * z = h => y = (h - nz * z) / ny
    x_c = rng.uniform(-4.0, 4.0, size=500)
    z_c = rng.uniform(0.5, 15.0, size=500)
    y_c = (h - n[2] * z_c) / n[1]
    floor_pts_c = np.stack([x_c, y_c, z_c], axis=-1)

    # Verify points lie on floor plane in C
    residuals = np.abs(np.dot(floor_pts_c, n) - h)
    assert np.max(residuals) < 1e-12

    # Transform to ground frame G
    floor_pts_g = camera_to_ground(floor_pts_c, n, h)
    z_g = floor_pts_g[:, 2]
    max_floor_z_err = np.max(np.abs(z_g))
    assert max_floor_z_err < 1e-6, f"Floor Z error {max_floor_z_err} >= 1e-6 at pitch {pitch_deg}"

    # 2. Wall points at Y = 3 in G
    x_wall_g = rng.uniform(-3.0, 3.0, size=500)
    y_wall_g = np.full(500, 3.0, dtype=np.float64)
    z_wall_g = rng.uniform(0.0, 2.5, size=500)  # height above ground
    wall_pts_g = np.stack([x_wall_g, y_wall_g, z_wall_g], axis=-1)

    # Transform to C then back to G
    wall_pts_c = ground_to_camera(wall_pts_g, n, h)
    wall_pts_rec_g = camera_to_ground(wall_pts_c, n, h)

    max_wall_y_err = np.max(np.abs(wall_pts_rec_g[:, 1] - 3.0))
    assert max_wall_y_err < 1e-6, f"Wall Y error {max_wall_y_err} >= 1e-6 at pitch {pitch_deg}"

    max_wall_full_err = np.max(np.abs(wall_pts_rec_g - wall_pts_g))
    assert max_wall_full_err < 1e-6, f"Full wall coordinate error {max_wall_full_err} >= 1e-6"
