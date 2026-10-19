"""Test T7: Temporal scale estimator EMA smoothing, outlier rejection, and clipping."""

from __future__ import annotations

import pytest

from core.config import MountConfig, ScaleConfig
from geometry.ground_plane import FitStatus
from geometry.scale import ScaleEstimator


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

    # Candidate with 20% change (s_raw = 1.20, h_hat = 1.4 / 1.20 = 1.1667)
    # 20% < 25% outlier threshold, but exceeds 15% max step change
    s_prev = est_clip.s
    res_clipped = est_clip.update(1.4 / 1.20, FitStatus.OK)
    change_ratio = (res_clipped.s - s_prev) / s_prev
    assert change_ratio <= 0.15 + 1e-9, f"Change ratio {change_ratio} exceeds max_step_change 0.15"
    assert res_clipped.s == pytest.approx(1.15)
