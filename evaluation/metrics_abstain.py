"""Abstention/risk-coverage metrics for Visiona3D.

Computes risk-coverage curve, AURC (Area Under Risk-Coverage curve),
and abstention rate at a threshold.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np


def compute_abstention_metrics(
    errors: np.ndarray,
    confidence: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute abstention / risk-coverage metrics.

    Args:
        errors: Per-sample errors (e.g., absolute depth error), shape (N,).
        confidence: Per-sample confidence scores (higher = more confident), shape (N,).
        thresholds: Optional confidence thresholds to evaluate abstention rate.
                    If None, uses 100 thresholds from min to max confidence.

    Returns:
        Dict with keys:
        - "aurc": Area Under Risk-Coverage curve (lower is better)
        - "risk_coverage_curve": list of (coverage, risk) points
        - "abstention_rates": dict threshold -> abstention rate (if thresholds provided)
        - "random_aurc_mean": mean AURC over random confidence permutations (for comparison)
        - "random_aurc_std": std of random AURC
    """
    err = np.asarray(errors, dtype=np.float64).ravel()
    conf = np.asarray(confidence, dtype=np.float64).ravel()

    valid = np.isfinite(err) & np.isfinite(conf)
    err = err[valid]
    conf = conf[valid]

    if len(err) == 0:
        return {
            "aurc": float("nan"),
            "risk_coverage_curve": [],
            "abstention_rates": {},
            "random_aurc_mean": float("nan"),
            "random_aurc_std": float("nan"),
        }

    # Sort by confidence descending (high confidence = keep)
    order = np.argsort(conf)[::-1]
    err_sorted = err[order]
    n = len(err_sorted)

    # Risk-coverage curve: risk = mean error of kept samples, coverage = fraction kept
    coverages = np.arange(1, n + 1) / n
    cumulative_err = np.cumsum(err_sorted)
    risks = cumulative_err / np.arange(1, n + 1)

    # AURC = area under risk-coverage curve (trapezoidal rule)
    aurc = float(np.trapz(risks, coverages))

    # Risk-coverage curve points (subsample for storage)
    stride = max(1, n // 100)
    curve = [(float(coverages[i]), float(risks[i])) for i in range(0, n, stride)]

    # Random baseline AURC (mean over 200 permutations)
    rng = np.random.default_rng(42)
    random_aurcs = []
    for _ in range(200):
        rand_order = rng.permutation(n)
        rand_err = err[rand_order]
        rand_cumsum = np.cumsum(rand_err)
        rand_risks = rand_cumsum / np.arange(1, n + 1)
        random_aurcs.append(float(np.trapz(rand_risks, coverages)))

    result = {
        "aurc": aurc,
        "risk_coverage_curve": curve,
        "random_aurc_mean": float(np.mean(random_aurcs)),
        "random_aurc_std": float(np.std(random_aurcs)),
    }

    # Abstention rates at specific thresholds
    if thresholds is not None:
        abst_rates = {}
        for thresh in thresholds:
            kept = conf >= thresh
            abst_rate = 1.0 - float(np.mean(kept))
            abst_rates[float(thresh)] = abst_rate
        result["abstention_rates"] = abst_rates

    return result


def compute_abstention_metrics_per_frame(
    frame_errors: list[np.ndarray],
    frame_confidences: list[np.ndarray],
) -> dict[str, Any]:
    """Aggregate abstention metrics over multiple frames."""
    all_errors = np.concatenate([e.ravel() for e in frame_errors])
    all_conf = np.concatenate([c.ravel() for c in frame_confidences])
    return compute_abstention_metrics(all_errors, all_conf)