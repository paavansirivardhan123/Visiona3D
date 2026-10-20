"""Test T7: Temporal scale estimator EMA smoothing, outlier rejection, and clipping."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Intrinsics, MountConfig, ScaleConfig
from geometry.ground_plane import FitStatus
from geometry.lift import lift_frame
from geometry.scale import ScaleEstimator
from tests.synthetic import render_scene, scale_depth


def test_t7_ema_behaviour():
    """T7: Single 40% outlier ignored; persistent 40% change accepted after 5 frames; clipping holds."""
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    cfg = ScaleConfig(alpha=0.2, max_step_change=0.15, outlier_threshold=0.25, outlier_patience=5, initial_scale=1.0)
    estimator = ScaleEstimator(mount, cfg)

    # Baseline frame (h_hat = 1.4 => s_raw = 1.0)
    res = estimator.update(1.4, FitStatus.OK)
    assert res.s == pytest.approx(1.0)
    assert res.status == "OK"

    # Part 1: Single 40% outlier frame (s_raw = 1.4, h_hat = 1.0)
    # Deviation = (1.4 - 1.0) / 1.0 = 40% > 25% threshold
    res_outlier = estimator.update(1.0, FitStatus.OK)
    assert res_outlier.s == pytest.approx(1.0), "Single 40% outlier must be ignored"
    assert res_outlier.status == "OUTLIER_IGNORED"

    # Return to normal: counter resets
    res_normal = estimator.update(1.4, FitStatus.OK)
    assert res_normal.s == pytest.approx(1.0)

    # Part 2: Persistent 40% change fed for consecutive frames
    # Frames 1-4 must still be ignored
    for frame_idx in range(1, 5):
        res_step = estimator.update(1.0, FitStatus.OK)
        assert res_step.s == pytest.approx(1.0), f"Frame {frame_idx} of persistent outlier must still be ignored"
        assert res_step.status == "OUTLIER_IGNORED"

    # Frame 5: exactly 5 frames reached => persistent shift accepted and reset to 1.4
    res_frame5 = estimator.update(1.0, FitStatus.OK)
    assert res_frame5.s == pytest.approx(1.4), "Persistent 40% change must be accepted on frame 5"
    assert res_frame5.status == "OK"

    # Part 3: Per-frame clipping holds (+-15% limit)
    # Start fresh estimator at s = 1.0 with alpha=1.0 (step change) but within non-outlier range
    cfg_clip = ScaleConfig(alpha=1.0, max_step_change=0.15, outlier_threshold=0.25, initial_scale=1.0)
    est_clip = ScaleEstimator(mount, cfg_clip)

    # Prime the first-frame direct initialisation at s = 1.0; clipping applies afterwards.
    est_clip.update(1.4, FitStatus.OK)

    # Candidate with 20% change (s_raw = 1.20, h_hat = 1.4 / 1.20 = 1.1667)
    # 20% < 25% outlier threshold, but exceeds 15% max step change
    s_prev = est_clip.s
    res_clipped = est_clip.update(1.4 / 1.20, FitStatus.OK)
    change_ratio = (res_clipped.s - s_prev) / s_prev
    assert change_ratio <= 0.15 + 1e-9, f"Change ratio {change_ratio} exceeds max_step_change 0.15"
    assert res_clipped.s == pytest.approx(1.15)


def test_first_valid_frame_initialises_scale_directly():
    """First valid frame must yield metric-correct wall distance, not start at s = 1.

    Scene: h = 1.4 m, pitch = 15 deg, wall at Y = 4.0 m. Corrupted depth is k * z,
    so the plane fit sees h_hat = k * 1.4 and s_raw = 1.4 / (k * 1.4) = 1 / k.
    With direct first-frame initialisation, the lifted wall distance error is 0
    (numerically < 2%) on frame 1 for every k.
    """
    K = Intrinsics.from_fov(width=320, height=240, hfov_deg=75.0)
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    depth, floor_mask, wall_mask, _, _ = render_scene(K, 240, 320, 1.4, 15.0, 4.0, seed=0)

    for k in [0.5, 0.8, 1.3, 2.0]:
        estimator = ScaleEstimator(mount)
        res = lift_frame(scale_depth(depth, k), floor_mask, {"wall": wall_mask}, K, mount, estimator)
        assert res.scale_result.s == pytest.approx(1.0 / k, rel=1e-9)
        measured_wall_y = float(np.median(res.points_g["wall"][:, 1]))
        err = abs(measured_wall_y - 4.0) / 4.0
        assert err < 0.02, f"k={k}: first-frame wall distance error {err:.4%} exceeds 2%"


def test_non_ok_status_keeps_previous_s():
    """A failed plane fit must keep the previous scale and report the fit status."""
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    estimator = ScaleEstimator(mount)
    assert estimator.update(1.4, FitStatus.OK).s == pytest.approx(1.0)

    res = estimator.update(0.7, FitStatus.TOO_FEW_POINTS)  # would imply s_raw = 2.0
    assert res.s == pytest.approx(1.0)
    assert res.status == FitStatus.TOO_FEW_POINTS.value

    res_invalid = estimator.update(float("nan"), FitStatus.OK)
    assert res_invalid.s == pytest.approx(1.0)


def test_first_valid_after_only_non_ok_initialises_directly():
    """Non-OK frames before the first valid one must not count as initialisation."""
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    estimator = ScaleEstimator(mount)
    for status in (FitStatus.TOO_FEW_POINTS, FitStatus.LOW_INLIER_RATIO, FitStatus.IMPLAUSIBLE_GEOMETRY):
        res = estimator.update(0.7, status)
        assert res.s == pytest.approx(1.0)
        assert res.status == status.value

    res = estimator.update(0.7, FitStatus.OK)  # s_raw = 1.4 / 0.7 = 2.0
    assert res.status == "OK"
    assert res.s == pytest.approx(2.0)
