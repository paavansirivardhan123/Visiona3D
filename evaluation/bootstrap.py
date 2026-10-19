"""Bootstrap confidence intervals for Visiona3D.

Provides seeded cluster bootstrap that resamples whole routes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional
import numpy as np


def bootstrap_routes(
    route_metrics: dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Cluster bootstrap resampling routes.

    Args:
        route_metrics: Dict mapping route_id -> array of per-frame metric values.
        metric_fn: Function that takes concatenated per-frame values and returns a scalar metric.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.
        confidence: Confidence level for percentile CI (e.g., 0.95).

    Returns:
        Dict with keys:
        - "mean": Bootstrap mean of the metric
        - "ci_lower": Lower percentile CI bound
        - "ci_upper": Upper percentile CI bound
        - "samples": Array of bootstrap sample values
    """
    route_ids = list(route_metrics.keys())
    route_arrays = [route_metrics[rid] for rid in route_ids]
    n_routes = len(route_ids)

    if n_routes == 0:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "samples": np.array([]),
        }

    rng = np.random.default_rng(seed)
    bootstrap_samples = []

    for _ in range(n_bootstrap):
        # Resample routes with replacement
        sampled_indices = rng.choice(n_routes, size=n_routes, replace=True)
        # Concatenate all frames from sampled routes
        sampled_values = np.concatenate([route_arrays[i] for i in sampled_indices])
        # Compute metric on resampled data
        metric_val = metric_fn(sampled_values)
        bootstrap_samples.append(metric_val)

    bootstrap_samples = np.array(bootstrap_samples)
    bootstrap_samples = bootstrap_samples[np.isfinite(bootstrap_samples)]

    if len(bootstrap_samples) == 0:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "samples": np.array([]),
        }

    alpha = 1.0 - confidence
    lower_pct = 100 * alpha / 2.0
    upper_pct = 100 * (1.0 - alpha / 2.0)

    return {
        "mean": float(np.mean(bootstrap_samples)),
        "ci_lower": float(np.percentile(bootstrap_samples, lower_pct)),
        "ci_upper": float(np.percentile(bootstrap_samples, upper_pct)),
        "samples": bootstrap_samples,
    }


def bootstrap_routes_multi(
    route_metrics: dict[str, dict[str, np.ndarray]],
    metric_fn: Callable[[dict[str, np.ndarray]], dict[str, float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Cluster bootstrap for multiple metrics simultaneously.

    Args:
        route_metrics: Dict route_id -> dict metric_name -> per-frame values array.
        metric_fn: Function that takes concatenated per-frame dict and returns dict of metrics.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.
        confidence: Confidence level.

    Returns:
        Dict metric_name -> bootstrap result dict.
    """
    route_ids = list(route_metrics.keys())
    n_routes = len(route_ids)

    if n_routes == 0:
        return {}

    rng = np.random.default_rng(seed)

    # Collect all metric names from first route
    metric_names = list(route_metrics[route_ids[0]].keys())
    bootstrap_results: dict[str, list[float]] = {m: [] for m in metric_names}

    for _ in range(n_bootstrap):
        sampled_indices = rng.choice(n_routes, size=n_routes, replace=True)
        # Concatenate per metric
        concatenated = {}
        for m in metric_names:
            concatenated[m] = np.concatenate([
                route_metrics[route_ids[i]][m] for i in sampled_indices
            ])
        # Compute all metrics
        metrics = metric_fn(concatenated)
        for m, val in metrics.items():
            if np.isfinite(val):
                bootstrap_results[m].append(val)

    # Compute CIs for each metric
    alpha = 1.0 - confidence
    lower_pct = 100 * alpha / 2.0
    upper_pct = 100 * (1.0 - alpha / 2.0)

    output = {}
    for m in metric_names:
        samples = np.array(bootstrap_results[m])
        if len(samples) > 0:
            output[m] = {
                "mean": float(np.mean(samples)),
                "ci_lower": float(np.percentile(samples, lower_pct)),
                "ci_upper": float(np.percentile(samples, upper_pct)),
                "samples": samples,
            }
        else:
            output[m] = {
                "mean": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "samples": np.array([]),
            }

    return output