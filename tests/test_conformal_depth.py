"""C1-C6, C11: conformal depth calibration tests.

All expected values are hand-computed or analytically justified.  The
simulation sizes are chosen to keep the file comfortably under the 40 s budget
for the new calibration tests.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from calibration.conformal_depth import (
    ConformalDepth,
    SCORE_OVERESTIMATION,
    SCORE_TWO_SIDED,
    conformal_quantile,
    depth_bin_groups,
)


def _lognormal_predictions(
    rng: np.random.Generator,
    n: int,
    sigma: float,
    depth_low: float = 0.5,
    depth_high: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    gt = rng.uniform(depth_low, depth_high, size=n)
    pred = gt * np.exp(rng.normal(0.0, sigma, size=n))
    return pred, gt


# ---------------------------------------------------------------------------
# C1: quantile
# ---------------------------------------------------------------------------
def test_c1_conformal_quantile_hand_computed():
    assert conformal_quantile(np.arange(1, 11), 0.10) == 10.0
    assert conformal_quantile(np.arange(1, 11), 0.20) == 9.0
    assert math.isinf(conformal_quantile(np.arange(1, 6), 0.10))
    assert math.isinf(conformal_quantile([], 0.10))
    # NaN is not a score; it must be ignored rather than poisoning the sort.
    assert conformal_quantile([1.0, np.nan, 2.0, 3.0], 0.5) == 2.0
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 0.0)


# ---------------------------------------------------------------------------
# C2: synthetic marginal coverage
# ---------------------------------------------------------------------------
def test_c2_marginal_coverage_300_splits():
    rng = np.random.default_rng(20261022)
    coverages = []
    for _ in range(300):
        pred_cal, gt_cal = _lognormal_predictions(rng, 200, sigma=0.15)
        model = ConformalDepth(alpha=0.10).fit(pred_cal, gt_cal)
        pred_test, gt_test = _lognormal_predictions(rng, 2000, sigma=0.15)
        lower, upper = model.intervals(pred_test)
        coverages.append(float(np.mean((gt_test >= lower) & (gt_test <= upper))))

    mean_coverage = float(np.mean(coverages))
    assert 0.89 <= mean_coverage <= 0.92, mean_coverage
    assert 0.0 <= min(coverages) <= max(coverages) <= 1.0


# ---------------------------------------------------------------------------
# C3: one-sided lower bound
# ---------------------------------------------------------------------------
def test_c3_one_sided_lower_bound_coverage_and_direction():
    rng = np.random.default_rng(303)
    coverages = []
    for _ in range(200):
        pred_cal, gt_cal = _lognormal_predictions(rng, 200, sigma=0.12)
        model = ConformalDepth(alpha=0.10, score=SCORE_OVERESTIMATION).fit(
            pred_cal, gt_cal
        )
        pred_test, gt_test = _lognormal_predictions(rng, 2000, sigma=0.12)
        lower = model.lower_bounds(pred_test)
        coverages.append(float(np.mean(gt_test >= lower)))

    mean_coverage = float(np.mean(coverages))
    assert 0.87 <= mean_coverage <= 0.93, mean_coverage

    # With a positive tail score the bound must sit strictly below a
    # prediction that overestimates the truth.
    over = np.array([2.0, 3.0, 4.0])
    truth = np.array([1.0, 2.0, 3.0])
    model = ConformalDepth(alpha=0.10, score=SCORE_OVERESTIMATION).fit(over, truth)
    lower = model.lower_bounds(over)
    assert np.all(lower < over)
    assert np.all(lower >= 0.0)

    # Hand-computed small-sample result: with n=3 and alpha=0.1 the corrected
    # order is k=4 > n, so the quantile is +inf and the safe lower bound is 0.
    assert np.allclose(lower, 0.0)


# ---------------------------------------------------------------------------
# C4: Mondrian calibration
# ---------------------------------------------------------------------------
def test_c4_mondrian_by_predicted_depth_bins_and_pooled_gap():
    rng = np.random.default_rng(404)
    n_cal, n_test = 4000, 4000
    sigmas = np.array([0.03, 0.08, 0.18, 0.28])

    def sample(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gt = rng.uniform(0.5, 8.0, size=n)
        true_bins = depth_bin_groups(gt)
        pred = gt * np.exp(rng.normal(0.0, sigmas[true_bins], size=n))
        groups = depth_bin_groups(pred)
        return pred, gt, groups

    pred_cal, gt_cal, groups_cal = sample(n_cal)
    pred_test, gt_test, groups_test = sample(n_test)

    mondrian = ConformalDepth(alpha=0.10, min_group_size=50).fit(
        pred_cal, gt_cal, groups=groups_cal
    )
    pooled = ConformalDepth(alpha=0.10).fit(pred_cal, gt_cal)
    m_lo, m_hi = mondrian.intervals(pred_test, groups=groups_test)
    p_lo, p_hi = pooled.intervals(pred_test)

    names = ["0-2m", "2-4m", "4-6m", "6+m"]
    mondrian_coverages = {}
    pooled_coverages = {}
    for index, name in enumerate(names):
        mask = groups_test == index
        assert np.sum(mask) >= 50
        mondrian_coverages[name] = float(
            np.mean((gt_test[mask] >= m_lo[mask]) & (gt_test[mask] <= m_hi[mask]))
        )
        pooled_coverages[name] = float(
            np.mean((gt_test[mask] >= p_lo[mask]) & (gt_test[mask] <= p_hi[mask]))
        )

    for value in mondrian_coverages.values():
        assert 0.86 <= value <= 0.93, mondrian_coverages
    gaps = {
        name: abs(pooled_coverages[name] - mondrian_coverages[name])
        for name in names
    }
    assert max(gaps.values()) > 0.05, (gaps, mondrian_coverages, pooled_coverages)


# ---------------------------------------------------------------------------
# C5: small groups, invalid predictions, no ground-truth grouping
# ---------------------------------------------------------------------------
def test_c5_small_group_pooled_fallback_and_invalid_predictions():
    rng = np.random.default_rng(505)
    n_common = 100
    pred_common = rng.uniform(1.0, 5.0, size=n_common)
    gt_common = pred_common * np.exp(rng.normal(0.0, 0.10, size=n_common))
    pred_rare = rng.uniform(1.0, 5.0, size=5)
    gt_rare = pred_rare * np.exp(rng.normal(0.0, 0.10, size=5))

    pred = np.concatenate([pred_common, pred_rare])
    gt = np.concatenate([gt_common, gt_rare])
    groups = ["common"] * n_common + ["rare"] * 5
    model = ConformalDepth(alpha=0.10, min_group_size=30).fit(pred, gt, groups=groups)

    assert "rare" in model.fallback_groups_
    assert model.fallback_groups_["rare"]["n"] == 5
    assert "rare" not in model.group_quantiles_
    result = model.predict([2.0, 3.0], groups=["rare", "unknown"])
    assert np.all(result.used_pooled)
    assert np.allclose(result.quantile, model.pooled_quantile_)

    invalid_pred = [np.nan, 0.0, -1.0, 2.0]
    invalid_gt = [1.0, 1.0, 1.0, 2.0]
    invalid_model = ConformalDepth(alpha=0.10).fit(invalid_pred, invalid_gt)
    assert invalid_model.n_invalid_ == 3
    out = invalid_model.predict(invalid_pred)
    assert np.isnan(out.lower[:3]).all()
    assert np.isnan(out.upper[:3]).all()
    assert not out.valid[:3].any()
    assert out.valid[3]

    restored = ConformalDepth.from_dict(model.to_dict())
    assert restored.pooled_quantile_ == model.pooled_quantile_
    assert restored.group_quantiles_ == model.group_quantiles_
    assert restored.fallback_groups_ == model.fallback_groups_


def test_c5_no_implicit_ground_truth_grouping():
    """Groups must be supplied from inference-time information, never GT."""
    rng = np.random.default_rng(506)
    pred = rng.uniform(1.0, 6.0, size=200)
    gt = pred * np.exp(rng.normal(0.0, 0.08, size=200))
    model = ConformalDepth(alpha=0.10, min_group_size=10).fit(pred, gt, groups=None)
    assert model.group_quantiles_ == {}
    assert model.group_counts_ == {}
    assert model.fallback_groups_ == {}
    assert np.all(model.predict(pred).used_pooled)


# ---------------------------------------------------------------------------
# C6: distribution shift and route-level spread
# ---------------------------------------------------------------------------
def test_c6_distribution_shift_coverage_degradation():
    rng = np.random.default_rng(606)
    pred_cal, gt_cal = _lognormal_predictions(rng, 2000, sigma=0.03)
    model = ConformalDepth(alpha=0.10).fit(pred_cal, gt_cal)

    pred_same, gt_same = _lognormal_predictions(rng, 5000, sigma=0.03)
    lo_same, hi_same = model.intervals(pred_same)
    same_coverage = float(np.mean((gt_same >= lo_same) & (gt_same <= hi_same)))

    pred_shift, gt_shift = _lognormal_predictions(rng, 5000, sigma=0.08)
    lo_shift, hi_shift = model.intervals(pred_shift)
    shift_coverage = float(np.mean((gt_shift >= lo_shift) & (gt_shift <= hi_shift)))

    assert 0.88 <= same_coverage <= 0.92, same_coverage
    assert shift_coverage < 0.75, shift_coverage
    assert same_coverage - shift_coverage > 0.15


def test_c6_route_level_coverage_spread_across_ten_routes():
    rng = np.random.default_rng(607)
    pred_cal, gt_cal = _lognormal_predictions(rng, 2000, sigma=0.03)
    model = ConformalDepth(alpha=0.10).fit(pred_cal, gt_cal)

    route_sigmas = np.linspace(0.03, 0.10, 10)
    coverages = []
    for sigma in route_sigmas:
        pred_route, gt_route = _lognormal_predictions(rng, 500, sigma=float(sigma))
        lower, upper = model.intervals(pred_route)
        coverages.append(float(np.mean((gt_route >= lower) & (gt_route <= upper))))

    spread = max(coverages) - min(coverages)
    assert len(coverages) == 10
    assert spread > 0.20, coverages
    assert min(coverages) < 0.80
    assert max(coverages) > 0.85


# ---------------------------------------------------------------------------
# C11: determinism and speed
# ---------------------------------------------------------------------------
def test_c11_determinism_and_speed():
    start = time.perf_counter()
    rng = np.random.default_rng(1111)
    pred = rng.uniform(0.5, 10.0, size=5000)
    gt = pred * np.exp(rng.normal(0.0, 0.10, size=5000))
    groups = depth_bin_groups(pred)

    first = ConformalDepth(alpha=0.10, min_group_size=100).fit(pred, gt, groups)
    second = ConformalDepth(alpha=0.10, min_group_size=100).fit(pred, gt, groups)
    assert first.to_dict() == second.to_dict()
    assert np.array_equal(first.intervals(pred, groups)[0], second.intervals(pred, groups)[0])
    assert np.array_equal(first.intervals(pred, groups)[1], second.intervals(pred, groups)[1])
    assert first.score == SCORE_TWO_SIDED
    assert time.perf_counter() - start < 10.0
