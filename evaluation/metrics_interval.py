"""Interval/uncertainty metrics for Visiona3D.

Computes coverage, mean width, mean relative width, coverage gap,
and Gneiting-Raftery interval score.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.config import RANGE_BIN_EDGES


def compute_interval_metrics(
    gt: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    nominal_coverage: float = 0.95,
    valid_mask: Optional[np.ndarray] = None,
    depths: Optional[np.ndarray] = None,
    tags: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Compute prediction interval metrics.

    Args:
        gt: Ground truth values (depth) of shape (N,) or (H, W).
        pred_lo: Lower bound of prediction interval, same shape as gt.
        pred_hi: Upper bound of prediction interval, same shape as gt.
        nominal_coverage: Target coverage level (e.g., 0.95).
        valid_mask: Optional boolean mask of valid GT.
        depths: Optional depth values for per-bin metrics (same shape as gt).
        tags: Optional dict with tag keys for per-tag metrics.

    Returns:
        Dict with keys:
        - "coverage": fraction of GT inside [lo, hi] (inclusive bounds)
        - "mean_width": mean(hi - lo)
        - "mean_rel_width": mean((hi - lo) / gt)
        - "coverage_gap": coverage - nominal_coverage
        - "interval_score": Gneiting-Raftery score
        - "per_bin": dict bin_name -> metrics (if depths provided)
        - "per_tag": dict tag_key -> metrics (if tags provided)
    """
    gt_arr = np.asarray(gt, dtype=np.float64).ravel()
    lo = np.asarray(pred_lo, dtype=np.float64).ravel()
    hi = np.asarray(pred_hi, dtype=np.float64).ravel()

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).ravel()
        gt_arr = gt_arr[valid]
        lo = lo[valid]
        hi = hi[valid]
        if depths is not None:
            depths = np.asarray(depths, dtype=np.float64).ravel()[valid]
    else:
        valid = np.isfinite(gt_arr) & np.isfinite(lo) & np.isfinite(hi) & (gt_arr > 0) & (lo > 0) & (hi > 0)
        gt_arr = gt_arr[valid]
        lo = lo[valid]
        hi = hi[valid]
        if depths is not None:
            depths = np.asarray(depths, dtype=np.float64).ravel()[valid]

    if len(gt_arr) == 0:
        return _empty_interval_metrics()

    # Inclusive bounds: gt in [lo, hi]
    inside = (gt_arr >= lo) & (gt_arr <= hi)
    coverage = float(np.mean(inside))

    width = hi - lo
    mean_width = float(np.mean(width))
    mean_rel_width = float(np.mean(width / gt_arr))

    coverage_gap = coverage - nominal_coverage

    # Gneiting-Raftery interval score
    # IS = width + (2/alpha)*(lo - y)*1[y < lo] + (2/alpha)*(y - hi)*1[y > hi]
    # where alpha = 1 - nominal_coverage
    alpha = 1.0 - nominal_coverage
    penalty_lo = (2.0 / alpha) * np.maximum(lo - gt_arr, 0.0)
    penalty_hi = (2.0 / alpha) * np.maximum(gt_arr - hi, 0.0)
    interval_score = float(np.mean(width + penalty_lo + penalty_hi))

    result = {
        "coverage": coverage,
        "mean_width": mean_width,
        "mean_rel_width": mean_rel_width,
        "coverage_gap": coverage_gap,
        "interval_score": interval_score,
    }

    # Per range bin metrics
    if depths is not None and len(depths) == len(gt_arr):
        per_bin = {}
        bin_names = ["0-2m", "2-4m", "4-6m", "6+m"]
        for i, bin_name in enumerate(bin_names):
            if i < 3:
                bin_mask = (depths >= RANGE_BIN_EDGES[i]) & (depths < RANGE_BIN_EDGES[i + 1])
            else:
                bin_mask = depths >= RANGE_BIN_EDGES[3]

            gt_bin = gt_arr[bin_mask]
            lo_bin = lo[bin_mask]
            hi_bin = hi[bin_mask]

            if len(gt_bin) > 0:
                inside_bin = (gt_bin >= lo_bin) & (gt_bin <= hi_bin)
                cov_bin = float(np.mean(inside_bin))
                width_bin = hi_bin - lo_bin
                m_width_bin = float(np.mean(width_bin))
                m_rel_width_bin = float(np.mean(width_bin / gt_bin))
                gap_bin = cov_bin - nominal_coverage

                alpha_bin = alpha
                penalty_lo_bin = (2.0 / alpha_bin) * np.maximum(lo_bin - gt_bin, 0.0)
                penalty_hi_bin = (2.0 / alpha_bin) * np.maximum(gt_bin - hi_bin, 0.0)
                is_bin = float(np.mean(width_bin + penalty_lo_bin + penalty_hi_bin))

                per_bin[bin_name] = {
                    "coverage": cov_bin,
                    "mean_width": m_width_bin,
                    "mean_rel_width": m_rel_width_bin,
                    "coverage_gap": gap_bin,
                    "interval_score": is_bin,
                    "count": int(len(gt_bin)),
                }
            else:
                per_bin[bin_name] = {
                    "coverage": float("nan"),
                    "mean_width": float("nan"),
                    "mean_rel_width": float("nan"),
                    "coverage_gap": float("nan"),
                    "interval_score": float("nan"),
                    "count": 0,
                }
        result["per_bin"] = per_bin

    # Per tag metrics (placeholder for frame-level aggregation)
    if tags is not None:
        result["per_tag"] = {}

    return result


def _empty_interval_metrics() -> dict[str, Any]:
    empty = {
        "coverage": float("nan"),
        "mean_width": float("nan"),
        "mean_rel_width": float("nan"),
        "coverage_gap": float("nan"),
        "interval_score": float("nan"),
    }
    per_bin = {}
    for bin_name in ["0-2m", "2-4m", "4-6m", "6+m"]:
        per_bin[bin_name] = {**empty, "count": 0}
    return {**empty, "per_bin": per_bin, "per_tag": {}}


def aggregate_interval_metrics(
    frame_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-frame interval metrics across a dataset."""
    if not frame_metrics:
        return _empty_interval_metrics()

    keys = ["coverage", "mean_width", "mean_rel_width", "coverage_gap", "interval_score"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in frame_metrics if not np.isnan(m[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")

    bin_names = ["0-2m", "2-4m", "4-6m", "6+m"]
    per_bin = {}
    for bin_name in bin_names:
        bin_vals = {k: [] for k in keys}
        counts = []
        for m in frame_metrics:
            if bin_name in m.get("per_bin", {}):
                bin_m = m["per_bin"][bin_name]
                if "count" in bin_m and bin_m["count"] > 0:
                    counts.append(bin_m["count"])
                    for k in keys:
                        if not np.isnan(bin_m[k]):
                            bin_vals[k].append(bin_m[k])
        per_bin[bin_name] = {}
        for k in keys:
            per_bin[bin_name][k] = float(np.mean(bin_vals[k])) if bin_vals[k] else float("nan")
        per_bin[bin_name]["count"] = int(np.sum(counts)) if counts else 0

    agg["per_bin"] = per_bin
    agg["per_tag"] = {}

    return agg