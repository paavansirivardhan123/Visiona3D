"""Test D6: Depth metrics with hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics_depth import compute_depth_metrics, aggregate_depth_metrics


def test_d6_absrel_exact():
    """D6: GT [1,2,4,8], pred 1.1x GT gives AbsRel exactly 0.1."""
    gt = np.array([1.0, 2.0, 4.0, 8.0])
    pred = gt * 1.1  # 10% overestimation

    metrics = compute_depth_metrics(gt, pred)

    # AbsRel = mean(|gt - pred| / gt) = mean(0.1) = 0.1
    assert metrics["AbsRel"] == pytest.approx(0.1, abs=1e-10)

    # Bias = mean(pred - gt) = mean(0.1 * gt) = 0.1 * mean(gt) = 0.1 * 3.75 = 0.375
    # But bias is signed mean error: mean(pred - gt) = mean(0.1*gt) = 0.375
    # Bias percent = mean((pred-gt)/gt) * 100 = 10%
    bias = metrics["Bias"]
    expected_bias = np.mean(pred - gt)
    assert bias == pytest.approx(expected_bias)

    # Delta125 = all ratios = 1.1 < 1.25 -> 1.0
    assert metrics["Delta125"] == 1.0


def test_d6_per_bin_edges():
    """D6: Exact edges 2.0, 4.0, 6.0 fall in right-hand bin."""
    # Bins: [0,2), [2,4), [4,6), [6,inf) — left-inclusive, right-exclusive
    # So 2.0 goes to bin [2,4), 4.0 to [4,6), 6.0 to [6,inf)
    gt = np.array([1.9, 2.0, 3.9, 4.0, 5.9, 6.0, 7.0])
    pred = gt * 1.0  # Perfect prediction

    metrics = compute_depth_metrics(gt, pred)

    per_bin = metrics["per_bin"]

    # 1.9 in [0,2)
    assert per_bin["0-2m"]["count"] == 1
    # 2.0, 3.9 in [2,4)
    assert per_bin["2-4m"]["count"] == 2
    # 4.0, 5.9 in [4,6)
    assert per_bin["4-6m"]["count"] == 2
    # 6.0, 7.0 in [6,inf)
    assert per_bin["6+m"]["count"] == 2

    # All AbsRel should be 0
    for bin_name in ["0-2m", "2-4m", "4-6m", "6+m"]:
        assert per_bin[bin_name]["AbsRel"] == pytest.approx(0.0, abs=1e-10)


def test_d6_mae_rmse():
    """D6: MAE and RMSE match hand computation."""
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 2.5, 2.5, 4.5])  # Errors: +0.5, +0.5, -0.5, +0.5

    metrics = compute_depth_metrics(gt, pred)

    # MAE = (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
    assert metrics["MAE"] == pytest.approx(0.5)

    # RMSE = sqrt((0.25 + 0.25 + 0.25 + 0.25) / 4) = sqrt(0.25) = 0.5
    assert metrics["RMSE"] == pytest.approx(0.5)


def test_d6_aggregate():
    """D6: Aggregate across frames."""
    # Frame 1: gt=[1,2], pred=[1.1,2.1] -> rel = [0.1, 0.05] -> mean = 0.075
    # Frame 2: gt=[3,4], pred=[3.3,4.4] -> rel = [0.1, 0.1] -> mean = 0.1
    # Aggregate mean = (0.075 + 0.1) / 2 = 0.0875
    gt1 = np.array([1.0, 2.0])
    pred1 = np.array([1.1, 2.1])

    gt2 = np.array([3.0, 4.0])
    pred2 = np.array([3.3, 4.4])

    m1 = compute_depth_metrics(gt1, pred1)
    m2 = compute_depth_metrics(gt2, pred2)

    agg = aggregate_depth_metrics([m1, m2])

    # Aggregate AbsRel = mean of per-frame AbsRel
    assert agg["AbsRel"] == pytest.approx(0.0875)
    assert agg["per_bin"]["0-2m"]["count"] >= 0  # Just check it runs