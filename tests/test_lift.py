"""Tests T5 and T6: Full frame lifting, scale recovery, and failure handling."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Intrinsics, MountConfig, RansacConfig, ScaleConfig
from geometry.ground_plane import FitStatus
from geometry.lift import lift_frame
from geometry.scale import ScaleEstimator
from tests.synthetic import render_scene, scale_depth


@pytest.mark.parametrize("k", [0.5, 0.8, 1.3, 2.0])
def test_t5_scale_recovery(k: float):
    """T5: Multiply true depth by k in {0.5, 0.8, 1.3, 2.0}; recovered wall distance within 2% of 4m."""
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    wall_dist_true = 4.0

    depth_true, floor_mask, wall_mask, _, _ = render_scene(
        K, H=480, W=640, h=h_true, pitch_deg=pitch_true, wall_distance=wall_dist_true
    )

    uncalibrated_depth = scale_depth(depth_true, k)

    mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)
    cfg_scale = ScaleConfig(alpha=0.2, max_step_change=0.15, outlier_threshold=0.25, outlier_patience=5)
    estimator = ScaleEstimator(mount, cfg_scale)

    rng = np.random.default_rng(42)

    # Process stream of frames to allow EMA smoothing / persistent shift reset to converge
    result = None
    for _ in range(25):
        result = lift_frame(
            uncalibrated_depth,
            floor_mask,
            {"wall": wall_mask},
            K,
            mount,
            estimator,
            rng=rng,
        )

    assert result is not None
    assert result.plane_fit.status == FitStatus.OK

    # Extract wall points in ground frame G
    wall_pts_g = result.points_g["wall"]
    assert len(wall_pts_g) > 100

    # Measured distance along Y in G
    measured_wall_y = float(np.median(wall_pts_g[:, 1]))
    error_rel = abs(measured_wall_y - wall_dist_true) / wall_dist_true

    assert error_rel < 0.02, (
        f"For k={k}: recovered wall distance {measured_wall_y:.3f} m "
        f"is {error_rel * 100:.2f}% from true 4.0 m (>= 2%)"
    )


def test_t6_failure_handling():
    """T6: Tiny floor mask returns TOO_FEW_POINTS; wall mask returns LOW_INLIER_RATIO or IMPLAUSIBLE_GEOMETRY."""
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0

    depth, floor_mask, wall_mask, _, _ = render_scene(
        K, H=480, W=640, h=h_true, pitch_deg=pitch_true, wall_distance=3.0
    )

    mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)
    estimator = ScaleEstimator(mount)
    initial_s = estimator.s
    rng = np.random.default_rng(42)

    # 1. Tiny floor mask (< 500 points)
    tiny_mask = np.zeros_like(floor_mask)
    tiny_mask[240:250, 320:330] = True  # 100 pixels
    assert np.sum(tiny_mask) < 500

    res_tiny = lift_frame(
        depth,
        tiny_mask,
        {},
        K,
        mount,
        estimator,
        rng=rng,
    )

    assert res_tiny.plane_fit.status == FitStatus.TOO_FEW_POINTS
    assert res_tiny.scale_result.status == FitStatus.TOO_FEW_POINTS.value
    assert res_tiny.scale_result.s == initial_s  # Kept previous scale

    # 2. Wall mask as floor (vertical plane, violates floor geometry)
    res_wall = lift_frame(
        depth,
        wall_mask,
        {},
        K,
        mount,
        estimator,
        rng=rng,
    )

    assert res_wall.plane_fit.status in (
        FitStatus.LOW_INLIER_RATIO,
        FitStatus.IMPLAUSIBLE_GEOMETRY,
    )
    # Neither raised an exception
    assert np.isfinite(res_wall.scale_result.s)


# --- Integration test D12 ---

def test_d12_integration_scale_corrected_depth_metrics():
    """D12: For synthetic frames with predicted depth = k * true depth (k in {0.6, 1.0, 1.7}),
    run lift_frame and compute harness depth metrics on scale-corrected depth.
    Assert AbsRel after correction is below 0.02 and strictly lower than before for k != 1.
    """
    from evaluation.metrics_depth import compute_depth_metrics

    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    wall_dist_true = 4.0

    depth_true, floor_mask, wall_mask, _, _ = render_scene(
        K, H=480, W=640, h=h_true, pitch_deg=pitch_true, wall_distance=wall_dist_true
    )

    # Ground truth depth for evaluation (valid pixels only)
    valid_mask = floor_mask | wall_mask
    gt_depth = depth_true[valid_mask]

    for k in [0.6, 1.0, 1.7]:
        uncalibrated_depth = scale_depth(depth_true, k)

        mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)
        # Use aggressive scale config for faster convergence
        cfg_scale = ScaleConfig(alpha=1.0, max_step_change=1.0, outlier_threshold=1.0, outlier_patience=1, initial_scale=1.0)
        estimator = ScaleEstimator(mount, cfg_scale)
        rng = np.random.default_rng(123)

        # Run lift to get scale-corrected depth - need enough frames for convergence
        result = None
        for _ in range(10):
            result = lift_frame(
                uncalibrated_depth,
                floor_mask,
                {"wall": wall_mask},
                K,
                mount,
                estimator,
                rng=rng,
            )

        # Compute metrics on uncalibrated depth (before correction)
        pred_uncal = uncalibrated_depth[valid_mask]
        metrics_before = compute_depth_metrics(gt_depth, pred_uncal)

        # Compute metrics on scale-corrected depth (after correction)
        pred_corrected = result.scaled_depth[valid_mask]
        metrics_after = compute_depth_metrics(gt_depth, pred_corrected)

        absrel_before = metrics_before["AbsRel"]
        absrel_after = metrics_after["AbsRel"]

        print(f"k={k}: AbsRel before={absrel_before:.6f}, after={absrel_after:.6f}")

        # After correction, AbsRel should be < 0.02 (2%)
        assert absrel_after < 0.02, (
            f"k={k}: AbsRel after correction {absrel_after:.6f} >= 0.02"
        )

        # For k != 1, correction should strictly improve AbsRel
        if k != 1.0:
            assert absrel_after < absrel_before, (
                f"k={k}: AbsRel after ({absrel_after:.6f}) not strictly lower than before ({absrel_before:.6f})"
            )

        # For k == 1, AbsRel should be essentially zero (within numerical precision)
        if k == 1.0:
            assert absrel_after < 1e-6, f"k=1: AbsRel {absrel_after:.6f} not near zero"
