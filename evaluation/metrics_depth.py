"""Depth metrics for Visiona3D.

Computes AbsRel, RMSE, MAE, signed bias, delta<1.25 overall, per range bin, and per tag.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.config import RANGE_BIN_EDGES, range_bin_index


def compute_depth_metrics(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    tags: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Compute depth estimation metrics.

    Args:
        gt_depth: Ground truth depth (z in camera frame) of shape (N,) or (H, W).
        pred_depth: Predicted depth of same shape.
        valid_mask: Optional boolean mask of valid GT pixels.
        tags: Optional dict with tag keys (env, light, hazard, crowd) for per-tag metrics.

    Returns:
        Dict with keys:
        - "AbsRel", "RMSE", "MAE", "Bias", "Delta125": overall metrics
        - "per_bin": dict bin_name -> metrics dict
        - "per_tag": dict tag_key -> dict tag_value -> metrics dict (if tags provided)
    """
    gt = np.asarray(gt_depth, dtype=np.float64).ravel()
    pred = np.asarray(pred_depth, dtype=np.float64).ravel()

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).ravel()
        gt = gt[valid]
        pred = pred[valid]
    else:
        # Valid if GT is finite and positive
        valid = np.isfinite(gt) & (gt > 0.0) & np.isfinite(pred) & (pred > 0.0)
        gt = gt[valid]
        pred = pred[valid]

    if len(gt) == 0:
        return _empty_depth_metrics()

    # Overall metrics
    rel = np.abs(gt - pred) / gt
    absrel = float(np.mean(rel))
    rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
    mae = float(np.mean(np.abs(gt - pred)))
    bias = float(np.mean(pred - gt))  # signed bias: positive = overestimation
    delta125 = float(np.mean(np.maximum(gt / pred, pred / gt) < 1.25))

    result = {
        "AbsRel": absrel,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "Delta125": delta125,
    }

    # Per range bin metrics
    per_bin = {}
    bin_names = ["0-2m", "2-4m", "4-6m", "6+m"]
    for i, bin_name in enumerate(bin_names):
        if i < 3:
            bin_mask = (gt >= RANGE_BIN_EDGES[i]) & (gt < RANGE_BIN_EDGES[i + 1])
        else:
            bin_mask = gt >= RANGE_BIN_EDGES[3]

        gt_bin = gt[bin_mask]
        pred_bin = pred[bin_mask]

        if len(gt_bin) > 0:
            rel_bin = np.abs(gt_bin - pred_bin) / gt_bin
            per_bin[bin_name] = {
                "AbsRel": float(np.mean(rel_bin)),
                "RMSE": float(np.sqrt(np.mean((gt_bin - pred_bin) ** 2))),
                "MAE": float(np.mean(np.abs(gt_bin - pred_bin))),
                "Bias": float(np.mean(pred_bin - gt_bin)),
                "Delta125": float(np.mean(np.maximum(gt_bin / pred_bin, pred_bin / gt_bin) < 1.25)),
                "count": int(len(gt_bin)),
            }
        else:
            per_bin[bin_name] = {
                "AbsRel": float("nan"),
                "RMSE": float("nan"),
                "MAE": float("nan"),
                "Bias": float("nan"),
                "Delta125": float("nan"),
                "count": 0,
            }

    result["per_bin"] = per_bin

    # Per tag metrics
    if tags is not None:
        per_tag = {}
        for tag_key, tag_value in tags.items():
            # For simplicity, we compute per-tag only if we have frame-level tags
            # In practice, this would be called per-frame and aggregated
            pass
        result["per_tag"] = per_tag

    return result


def _empty_depth_metrics() -> dict[str, Any]:
    """Return empty metrics structure."""
    empty = {
        "AbsRel": float("nan"),
        "RMSE": float("nan"),
        "MAE": float("nan"),
        "Bias": float("nan"),
        "Delta125": float("nan"),
    }
    per_bin = {}
    for bin_name in ["0-2m", "2-4m", "4-6m", "6+m"]:
        per_bin[bin_name] = {**empty, "count": 0}
    return {**empty, "per_bin": per_bin, "per_tag": {}}


def aggregate_depth_metrics(
    frame_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-frame depth metrics across a dataset.

    Args:
        frame_metrics: List of metrics dicts from compute_depth_metrics.

    Returns:
        Aggregated metrics (mean of per-frame values).
    """
    if not frame_metrics:
        return _empty_depth_metrics()

    # Overall metrics: mean of per-frame values
    keys = ["AbsRel", "RMSE", "MAE", "Bias", "Delta125"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in frame_metrics if not np.isnan(m[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")

    # Per bin: mean of per-frame bin metrics
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
    agg["per_tag"] = {}  # Not implemented in simple aggregation

    return agg