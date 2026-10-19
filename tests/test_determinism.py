"""Test T10: Determinism and bit-identical reproducibility."""

from __future__ import annotations

import time
import numpy as np
import pytest

from core.config import Intrinsics, MountConfig, RansacConfig, ScaleConfig
from geometry.lift import lift_frame
from geometry.scale import ScaleEstimator
from tests.synthetic import add_noise, render_scene


def test_t10_determinism():
    """T10: Same seed gives bit-identical results twice; runs in well under 60 s."""
    start_time = time.perf_counter()

    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)

    clean_depth, floor_mask, wall_mask, _, _ = render_scene(
        K, H=480, W=640, h=h_true, pitch_deg=pitch_true, wall_distance=4.0
    )

    # Run 1
    rng1 = np.random.default_rng(98765)
    noisy_depth1 = add_noise(clean_depth, rel_sigma=0.02, outlier_frac=0.10, rng=rng1)
    est1 = ScaleEstimator(mount, ScaleConfig())
    res1 = lift_frame(
        noisy_depth1,
        floor_mask,
        {"wall": wall_mask},
        K,
        mount,
        est1,
        rng=rng1,
    )

    # Run 2 with identical seed
    rng2 = np.random.default_rng(98765)
    noisy_depth2 = add_noise(clean_depth, rel_sigma=0.02, outlier_frac=0.10, rng=rng2)
    est2 = ScaleEstimator(mount, ScaleConfig())
    res2 = lift_frame(
        noisy_depth2,
        floor_mask,
        {"wall": wall_mask},
        K,
        mount,
        est2,
        rng=rng2,
    )

    elapsed = time.perf_counter() - start_time
    assert elapsed < 60.0, f"Execution took {elapsed:.2f} s >= 60 s"

    # Bit-identical verification
    assert np.array_equal(noisy_depth1, noisy_depth2), "Noisy depth maps must be bit-identical"
    assert np.array_equal(res1.plane_fit.n, res2.plane_fit.n), "Estimated normal must be bit-identical"
    assert res1.plane_fit.h_hat == res2.plane_fit.h_hat, "Estimated height must be bit-identical"
    assert res1.plane_fit.inlier_ratio == res2.plane_fit.inlier_ratio
    assert res1.plane_fit.rms_residual == res2.plane_fit.rms_residual
    assert res1.scale_result.s == res2.scale_result.s, "Scale factor must be bit-identical"
    assert np.array_equal(res1.scaled_depth, res2.scaled_depth), "Scaled depth must be bit-identical"
    assert np.array_equal(res1.points_c["floor"], res2.points_c["floor"])
    assert np.array_equal(res1.points_g["floor"], res2.points_g["floor"])
    assert np.array_equal(res1.points_g["wall"], res2.points_g["wall"])
