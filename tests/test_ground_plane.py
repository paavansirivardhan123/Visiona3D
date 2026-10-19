"""Tests T3 and T4: Ground plane recovery under clean and noisy conditions."""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.config import Intrinsics, RansacConfig
from geometry.camera import backproject
from geometry.ground_plane import FitStatus, fit_ground_plane
from tests.synthetic import add_noise, render_scene


@pytest.mark.parametrize("h", [1.2, 1.6])
@pytest.mark.parametrize("pitch_deg", [0.0, 10.0, 20.0])
def test_t3_clean_plane_recovery(h: float, pitch_deg: float):
    """T3: h within 1% and normal angle error < 0.5 deg for h in {1.2, 1.6} and pitch in {0, 10, 20}."""
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    depth, floor_mask, _, _, gt = render_scene(K, H=480, W=640, h=h, pitch_deg=pitch_deg)

    points_c = backproject(depth, K)
    floor_pts = points_c[floor_mask]

    rng = np.random.default_rng(42)
    cfg = RansacConfig(max_iterations=500, max_points=5000)
    fit = fit_ground_plane(floor_pts, cfg=cfg, rng=rng)

    assert fit.status == FitStatus.OK

    # Height error within 1%
    rel_h_error = abs(fit.h_hat - h) / h
    assert rel_h_error < 0.01, f"Height error {rel_h_error * 100:.3f}% >= 1% (h={h}, pitch={pitch_deg})"

    # Normal angle error < 0.5 deg
    n_true = gt["n"]
    dot_prod = float(np.clip(np.dot(fit.n, n_true), -1.0, 1.0))
    angle_err_deg = math.degrees(math.acos(dot_prod))
    assert angle_err_deg < 0.5, f"Normal angle error {angle_err_deg:.3f} deg >= 0.5 deg (h={h}, pitch={pitch_deg})"


def test_t4_noisy_recovery():
    """T4: 2% relative noise + 10% outliers gives h within 3% over 20 seeds."""
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0

    clean_depth, floor_mask, _, _, _ = render_scene(
        K, H=480, W=640, h=h_true, pitch_deg=pitch_true
    )

    errors = []
    cfg = RansacConfig(max_iterations=600, max_points=5000)

    for seed in range(20):
        rng = np.random.default_rng(seed)
        noisy_depth = add_noise(
            clean_depth,
            rel_sigma=0.02,
            outlier_frac=0.10,
            outlier_scale=(0.3, 3.0),
            rng=rng,
        )

        noisy_pts_c = backproject(noisy_depth, K)
        floor_pts = noisy_pts_c[floor_mask]

        fit = fit_ground_plane(floor_pts, cfg=cfg, rng=rng)
        assert fit.status == FitStatus.OK, f"Fit failed on seed {seed} with status {fit.status}"

        rel_err = abs(fit.h_hat - h_true) / h_true
        errors.append(rel_err)
        assert rel_err < 0.03, f"Seed {seed} error {rel_err * 100:.3f}% >= 3.0%"

    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    print(f"\nT4 noisy recovery over 20 seeds: mean={mean_err * 100:.3f}%, max={max_err * 100:.3f}%")
    assert max_err < 0.03
