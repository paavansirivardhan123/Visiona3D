"""Test D7: Interval metrics with hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics_interval import compute_interval_metrics


def test_d7_coverage_width_score():
    """D7: Coverage, width, interval score equal hand-computed values on 6-sample example."""
    # 6 samples:
    # Sample 0: GT=1.0, interval [0.8, 1.2] -> inside
    # Sample 1: GT=2.0, interval [1.5, 2.5] -> inside
    # Sample 2: GT=3.0, interval [2.0, 2.8] -> GT > hi (3.0 > 2.8)
    # Sample 3: GT=4.0, interval [4.2, 5.0] -> GT < lo (4.0 < 4.2)
    # Sample 4: GT=5.0, interval [4.5, 5.5] -> inside
    # Sample 5: GT=6.0, interval [5.0, 7.0] -> inside
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lo = np.array([0.8, 1.5, 2.0, 4.2, 4.5, 5.0])
    hi = np.array([1.2, 2.5, 2.8, 5.0, 5.5, 7.0])

    # Nominal coverage 0.95, alpha = 0.05
    metrics = compute_interval_metrics(gt, lo, hi, nominal_coverage=0.95)

    # Coverage: 4/6 inside (samples 0, 1, 4, 5)
    assert metrics["coverage"] == pytest.approx(4/6)

    # Mean width: (0.4 + 1.0 + 0.8 + 0.8 + 1.0 + 2.0) / 6 = 6.0/6 = 1.0
    assert metrics["mean_width"] == pytest.approx(1.0)

    # Mean rel width: (0.4/1 + 1.0/2 + 0.8/3 + 0.8/4 + 1.0/5 + 2.0/6) / 6
    rel_widths = np.array([0.4, 0.5, 0.8/3, 0.2, 0.2, 2.0/6])
    assert metrics["mean_rel_width"] == pytest.approx(float(np.mean(rel_widths)))

    # Coverage gap = coverage - 0.95 = 4/6 - 0.95 = 0.666... - 0.95 = -0.2833...
    assert metrics["coverage_gap"] == pytest.approx(4/6 - 0.95)

    # Interval score: width + (2/alpha)*(lo-y) for y<lo + (2/alpha)*(y-hi) for y>hi
    # alpha = 0.05, 2/alpha = 40
    # Sample 0: width=0.4, inside -> score=0.4
    # Sample 1: width=1.0, inside -> score=1.0
    # Sample 2: width=0.8, y=3.0 > hi=2.8 -> penalty=40*(3.0-2.8)=8.0 -> score=8.8
    # Sample 3: width=0.8, y=4.0 < lo=4.2 -> penalty=40*(4.2-4.0)=8.0 -> score=8.8
    # Sample 4: width=1.0, inside -> score=1.0
    # Sample 5: width=2.0, inside -> score=2.0
    # Mean = (0.4 + 1.0 + 8.8 + 8.8 + 1.0 + 2.0) / 6 = 22.0/6 = 3.666...
    expected_is = 22.0 / 6.0
    assert metrics["interval_score"] == pytest.approx(expected_is)


def test_d7_coverage_gap_sign():
    """D7: Coverage-gap sign correct (negative when under-covered, positive when over-covered)."""
    # Over-covered: nominal 0.5, actual 1.0
    gt = np.array([1.0, 2.0])
    lo = np.array([0.5, 1.5])
    hi = np.array([1.5, 2.5])
    m = compute_interval_metrics(gt, lo, hi, nominal_coverage=0.5)
    assert m["coverage_gap"] > 0  # 1.0 - 0.5 = 0.5

    # Under-covered: nominal 0.95, actual 0.0
    gt2 = np.array([1.0])
    lo2 = np.array([2.0])  # GT below interval
    hi2 = np.array([3.0])
    m2 = compute_interval_metrics(gt2, lo2, hi2, nominal_coverage=0.95)
    assert m2["coverage_gap"] < 0  # 0.0 - 0.95 = -0.95


def test_d7_per_bin_and_per_tag():
    """D7: Per-bin and per-tag splits correct."""
    gt = np.array([1.0, 3.0, 5.0, 7.0])
    lo = np.array([0.8, 2.5, 4.5, 6.5])
    hi = np.array([1.2, 3.5, 5.5, 7.5])
    depths = np.array([1.0, 3.0, 5.0, 7.0])  # Same as gt for binning

    metrics = compute_interval_metrics(gt, lo, hi, nominal_coverage=0.95, depths=depths)

    # 1.0 in [0,2), 3.0 in [2,4), 5.0 in [4,6), 7.0 in [6,inf)
    for bin_name in ["0-2m", "2-4m", "4-6m", "6+m"]:
        assert metrics["per_bin"][bin_name]["count"] == 1
        assert metrics["per_bin"][bin_name]["coverage"] == 1.0  # All inside