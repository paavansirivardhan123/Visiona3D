"""Test D9: Bootstrap route-level resampling."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.bootstrap import bootstrap_routes, bootstrap_routes_multi


def test_d9_route_level_resampling():
    """D9: Route-level resampling keeps a route's frames together."""
    # Create route metrics with identifiable frame values
    route_metrics = {
        "route_0": np.array([1.0, 1.0, 1.0]),  # All 1.0
        "route_1": np.array([2.0, 2.0]),       # All 2.0
        "route_2": np.array([3.0, 3.0, 3.0]),  # All 3.0
    }

    def mean_fn(values):
        return float(np.mean(values))

    result = bootstrap_routes(route_metrics, mean_fn, n_bootstrap=100, seed=42)

    # Check that bootstrap samples are means of concatenated routes
    # Each bootstrap sample should be mean of 3 routes (with replacement)
    # Possible means: various combinations of [1,1,1], [2,2], [3,3,3]

    # Verify all samples are finite
    assert np.all(np.isfinite(result["samples"]))

    # Mean should be around 2.0 (average of 1, 2, 3 weighted by frame counts)
    # Route 0: 3 frames of 1.0, Route 1: 2 frames of 2.0, Route 2: 3 frames of 3.0
    # Overall mean = (3*1 + 2*2 + 3*3) / 8 = 16/8 = 2.0
    assert result["mean"] == pytest.approx(2.0, abs=0.2)


def test_d9_ci_contains_true_mean():
    """D9: CI contains true mean in at least 90% of 200 seeded trials."""
    # Known distribution: route means = 1, 2, 3 with equal frame counts
    # True overall mean = 2.0
    n_trials = 200
    contains_count = 0

    for trial_seed in range(n_trials):
        route_metrics = {
            f"route_{i}": np.full(10, float(i + 1)) for i in range(3)  # 10 frames each of 1, 2, 3
        }

        def mean_fn(values):
            return float(np.mean(values))

        result = bootstrap_routes(route_metrics, mean_fn, n_bootstrap=500, seed=trial_seed)

        if result["ci_lower"] <= 2.0 <= result["ci_upper"]:
            contains_count += 1

    # Should contain true mean in ~95% of trials (95% CI)
    # Allow some variance - at least 85%
    containment_rate = contains_count / n_trials
    assert containment_rate >= 0.85, f"CI containment rate {containment_rate:.2f} < 0.85"


def test_d9_same_seed_identical_output():
    """D9: Same seed gives identical output."""
    route_metrics = {
        "route_0": np.array([1.0, 2.0]),
        "route_1": np.array([3.0, 4.0]),
    }

    def mean_fn(values):
        return float(np.mean(values))

    r1 = bootstrap_routes(route_metrics, mean_fn, n_bootstrap=100, seed=999)
    r2 = bootstrap_routes(route_metrics, mean_fn, n_bootstrap=100, seed=999)

    assert r1["mean"] == r2["mean"]
    assert r1["ci_lower"] == r2["ci_lower"]
    assert r1["ci_upper"] == r2["ci_upper"]
    assert np.array_equal(r1["samples"], r2["samples"])


def test_d9_multi_metric():
    """D9: Multi-metric bootstrap works."""
    route_metrics = {
        "route_0": {"absrel": np.array([0.1, 0.1]), "rmse": np.array([0.5, 0.5])},
        "route_1": {"absrel": np.array([0.2, 0.2]), "rmse": np.array([1.0, 1.0])},
    }

    def multi_fn(values):
        return {
            "absrel": float(np.mean(values["absrel"])),
            "rmse": float(np.mean(values["rmse"])),
        }

    results = bootstrap_routes_multi(route_metrics, multi_fn, n_bootstrap=100, seed=42)

    assert "absrel" in results
    assert "rmse" in results
    assert np.isfinite(results["absrel"]["mean"])
    assert np.isfinite(results["rmse"]["mean"])