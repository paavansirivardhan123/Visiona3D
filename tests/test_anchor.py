"""Test T8: Object anchor extraction, distance, and 3D Ground-frame positioning."""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.config import Intrinsics, MountConfig
from geometry.camera import backproject
from geometry.frames import ground_frame_basis, ground_to_camera
from geometry.ground_plane import PlaneFit, fit_ground_plane
from geometry.lift import AnchorStatus, object_anchor
from tests.synthetic import render_scene


def test_t8_object_anchor_recovery():
    """T8: Box at known G position gives distance within 2% and position within 5 cm; empty mask gives error."""
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)

    # Box in G: symmetric in X, front face at Y=2.95, back face at Y=3.05, height Z in [0.45, 0.55]
    box_bounds = (-0.15, 0.15, 2.95, 3.05, 0.45, 0.55)
    depth, floor_mask, _, box_masks, _ = render_scene(
        K, H=480, W=640, h=h_true, pitch_deg=pitch_true, boxes=[box_bounds]
    )

    box_mask = box_masks[0]
    assert np.sum(box_mask) > 100, "Box must be visible in rendered frame"

    # Plane fit on floor
    rng = np.random.default_rng(42)
    floor_pts_c = backproject(depth, K)[floor_mask]
    plane = fit_ground_plane(floor_pts_c, rng=rng)

    # True front face center point in G
    p_front_g = np.array([0.0, 2.95, 0.5], dtype=np.float64)
    theta_rad = math.radians(pitch_true)
    n_true = np.array([0.0, math.cos(theta_rad), math.sin(theta_rad)], dtype=np.float64)
    p_front_c = ground_to_camera(p_front_g, n_true, h_true)
    true_front_depth = float(p_front_c[2])

    # Compute anchor
    anchor = object_anchor(depth, box_mask, K, mount, plane=plane, q=0.10)
    assert anchor.status == AnchorStatus.OK

    # 1. Distance error within 2% of front face depth
    dist_err_rel = abs(anchor.distance_m - true_front_depth) / true_front_depth
    assert dist_err_rel < 0.02, (
        f"Anchor distance {anchor.distance_m:.3f} m vs true front {true_front_depth:.3f} m "
        f"({dist_err_rel * 100:.2f}% >= 2%)"
    )

    # 2. Position error within 5 cm (0.05 m) of front face center
    pos_err_euclidean = float(np.linalg.norm(anchor.position_g - p_front_g))
    assert pos_err_euclidean < 0.05, (
        f"Anchor position error {pos_err_euclidean * 100:.2f} cm >= 5 cm "
        f"(estimated {anchor.position_g}, true {p_front_g})"
    )

    # 3. Empty mask gives error status
    empty_mask = np.zeros_like(box_mask)
    anchor_empty = object_anchor(depth, empty_mask, K, mount, plane=plane)
    assert anchor_empty.status == AnchorStatus.EMPTY_MASK

    # 4. Tiny mask gives error status (fewer than min_pixels valid pixels)
    tiny_mask = np.zeros_like(box_mask)
    box_v, box_u = np.where(box_mask)
    tiny_mask[box_v[:3], box_u[:3]] = True  # exactly 3 valid pixels
    anchor_tiny = object_anchor(depth, tiny_mask, K, mount, plane=plane, min_pixels=10)
    assert anchor_tiny.status == AnchorStatus.TINY_MASK


def test_t8_anchor_percentile_vs_mean():
    """T8 additional verification: Obstacle with depth extent verifies 10th percentile distance (<2%)."""
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)

    # Elongated obstacle extending from Y=2.5 to Y=3.5 in G
    box_bounds = (-0.2, 0.2, 2.5, 3.5, 0.4, 0.6)
    depth, floor_mask, _, box_masks, _ = render_scene(
        K, H=480, W=640, h=h_true, pitch_deg=pitch_true, boxes=[box_bounds]
    )
    box_mask = box_masks[0]

    floor_pts_c = backproject(depth, K)[floor_mask]
    plane = fit_ground_plane(floor_pts_c, rng=np.random.default_rng(42))

    p_front_g = np.array([0.0, 2.5, 0.5], dtype=np.float64)
    theta_rad = math.radians(pitch_true)
    n_true = np.array([0.0, math.cos(theta_rad), math.sin(theta_rad)], dtype=np.float64)
    p_front_c = ground_to_camera(p_front_g, n_true, h_true)
    true_front_depth = float(p_front_c[2])

    anchor = object_anchor(depth, box_mask, K, mount, plane=plane, q=0.10)
    dist_err_rel = abs(anchor.distance_m - true_front_depth) / true_front_depth

    # 10th percentile gives ~0.59% error; mean would give ~7.39% error (failing this assertion)
    assert dist_err_rel < 0.02, (
        f"Obstacle anchor distance error {dist_err_rel * 100:.2f}% >= 2.0% "
        f"(estimated {anchor.distance_m:.3f} m, true {true_front_depth:.3f} m)"
    )
