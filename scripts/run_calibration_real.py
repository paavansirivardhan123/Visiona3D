"""Real/fake evaluation for calibrated monocular depth and certified floor.

Usage (real data)::

    python scripts/run_calibration_real.py \
        --dataset data/visiona \
        --intrinsics reports/camera_intrinsics.json \
        --out reports

Usage (deterministic synthetic backend; no models, camera, network)::

    python scripts/run_calibration_real.py --fake --routes 6 --frames-per-route 4 \
        --out reports/fake_calibration

The real backend predicts depth at each annotated ``DistancePoint`` using the
median of a 5x5 patch.  It then performs leave-one-route-out split-conformal
calibration.  Predictions are written to ``predictions.jsonl`` and all
aggregate results to ``calibration_report.json``.

Scope labels in the report are deliberate: ``calibration_set`` metrics are
in-sample and optimistic; ``held_out`` metrics are the out-of-fold numbers on
which the G1 miscalibration gate is evaluated.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.conformal_depth import ConformalDepth, DEFAULT_ALPHA
from calibration.conformal_floor import (
    DEFAULT_DELTA,
    DEFAULT_LAMBDA_GRID,
    floor_loss,
    select_floor_threshold,
)
from core.config import (
    Intrinsics,
    MASK_FLOOR,
    MASK_IGNORE,
    MASK_UNLABELED,
    MountConfig,
    RANGE_BIN_EDGES,
    range_bin_index,
)
from dataset.loader import load_frame
from dataset.schema import FrameRecord
from dataset.synth_dataset import build_synthetic_dataset
from evaluation.bootstrap import bootstrap_routes
from evaluation.metrics_depth import compute_depth_metrics
from evaluation.metrics_interval import compute_interval_metrics

MIN_ROUTES = 5
DEFAULT_PATCH_RADIUS = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def discover_routes(dataset_root: str | Path) -> list[Path]:
    """Return sorted route directories containing a manifest."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    routes = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "route_manifest.json").exists()
    )
    return routes


def load_annotated_frames(
    dataset_root: str | Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load every frame with a manifest and return frame dicts plus warnings."""
    root = Path(dataset_root)
    routes = discover_routes(root)
    if not routes:
        raise FileNotFoundError(f"no route_manifest.json found under {root}")

    frames: list[dict[str, Any]] = []
    warnings: list[str] = []
    for route_dir in routes:
        manifest = json.loads((route_dir / "route_manifest.json").read_text(encoding="utf-8"))
        route_tags = dict(manifest.get("tags", {}))
        for frame_data in manifest.get("frames", []):
            record = FrameRecord.from_dict(frame_data)
            loaded = load_frame(root, record)
            tags = record.tags.to_dict()
            # Route-level tags override only when the frame omitted them; the
            # schema requires all four, so this is a consistency check.
            if route_tags and any(tags.get(k) in (None, "") for k in route_tags):
                tags.update(route_tags)
                warnings.append(
                    f"{record.frame_id}: filled missing frame tags from route manifest"
                )
            frames.append(
                {
                    "record": record,
                    "depth": np.asarray(loaded["depth"], dtype=np.float64),
                    "mask": np.asarray(loaded["mask"]),
                    "tags": tags,
                    "mount": loaded["mount"],
                    "distance_points": loaded["distance_points"],
                    "rgb_path": Path(loaded["rgb_path"]),
                    "route_id": record.route_id,
                }
            )
    return frames, warnings


def load_intrinsics_json(path: str | Path) -> Intrinsics:
    """Load the calibration JSON produced by ``scripts/calibrate_camera.py``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ["fx", "fy", "cx", "cy", "image_size"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"intrinsics JSON missing keys: {missing}")
    width, height = payload["image_size"]
    K = Intrinsics(
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        width=int(width),
        height=int(height),
    )
    if min(K.fx, K.fy, K.cx, K.cy, K.width, K.height) <= 0:
        raise ValueError("intrinsics values must be positive")
    return K


def intrinsics_from_frame(record: FrameRecord) -> Intrinsics:
    payload = record.intrinsics
    return Intrinsics(
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
    )


# ---------------------------------------------------------------------------
# Point predictions
# ---------------------------------------------------------------------------
def median_patch(
    depth: np.ndarray,
    u: float,
    v: float,
    radius: int = DEFAULT_PATCH_RADIUS,
) -> float:
    """Median of finite values in a square patch around ``(u, v)``."""
    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"depth must be 2D, got {arr.shape}")
    height, width = arr.shape
    u_int = int(round(float(u)))
    v_int = int(round(float(v)))
    u0 = max(0, u_int - radius)
    u1 = min(width, u_int + radius + 1)
    v0 = max(0, v_int - radius)
    v1 = min(height, v_int + radius + 1)
    patch = arr[v0:v1, u0:u1]
    finite = patch[np.isfinite(patch) & (patch > 0.0)]
    return float(np.median(finite)) if finite.size else float("nan")


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _real_models(weights_dir: Path, variant: str):
    from perception.depth_backend import DepthAnythingBackend
    from perception.seg_backend import SegFormerBackend

    depth_dir = weights_dir / (
        "depth-anything-v2-metric-indoor-small"
        if variant == "indoor"
        else "depth-anything-v2-metric-outdoor-small"
    )
    depth = DepthAnythingBackend(variant=variant, weights_dir=depth_dir)
    seg = SegFormerBackend(weights_dir=weights_dir / "segformer-b0-ade")
    return depth, seg


def _fake_depth_for_frame(
    depth: np.ndarray,
    route_index: int,
    frame_index: int,
    seed: int,
    sigma: float = 0.06,
    bias: float = 0.02,
) -> np.ndarray:
    """Deterministic log-normal depth corruption used by ``--fake``."""
    rng = np.random.default_rng(seed + 1000 * route_index + frame_index)
    eps = rng.normal(0.0, sigma, size=depth.shape)
    return np.asarray(depth, dtype=np.float64) * (1.0 + bias) * np.exp(eps)


def _fake_floor_probs(
    mask: np.ndarray,
    route_index: int,
    frame_index: int,
    seed: int,
) -> np.ndarray:
    """Deterministic floor probabilities derived from the synthetic GT mask."""
    rng = np.random.default_rng(seed + 2000 * route_index + frame_index)
    labels = np.asarray(mask)
    p = np.full(labels.shape, 0.08, dtype=np.float64)
    p[labels == MASK_FLOOR] = 0.88
    # Some wall pixels are deliberately floor-like, which gives the floor
    # calibration a non-trivial false-positive trade-off.
    wall = labels == 2
    if np.any(wall):
        picks = rng.random(labels.shape) < 0.25
        p[wall & picks] = 0.62
    p = np.clip(p + rng.normal(0.0, 0.05, size=labels.shape), 0.0, 1.0)
    return p


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _valid_point_mask(
    gt: np.ndarray,
    pred: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(gt) & (gt > 0.0) & np.isfinite(pred) & (pred > 0.0)
    if lower is not None:
        valid &= np.isfinite(lower)
    if upper is not None:
        valid &= np.isfinite(upper)
    return valid


def _interval_score(
    gt: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    width = upper - lower
    return (
        width
        + (2.0 / alpha) * np.maximum(lower - gt, 0.0)
        + (2.0 / alpha) * np.maximum(gt - upper, 0.0)
    )


def _bin_masks(values: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for index, name in enumerate(["0-2m", "2-4m", "4-6m", "6+m"]):
        if index < 3:
            masks[name] = (values >= RANGE_BIN_EDGES[index]) & (
                values < RANGE_BIN_EDGES[index + 1]
            )
        else:
            masks[name] = values >= RANGE_BIN_EDGES[index]
    return masks


def evaluate_g1(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    min_count: int = 30,
    max_abs_relative_bias: float = 0.05,
    min_naive_coverage: float = 0.80,
) -> dict[str, Any]:
    """Apply the G1 raw-depth miscalibration gate.

    A range bin is miscalibrated when it contains at least ``min_count``
    points and either the absolute signed relative bias exceeds 5% or naive
    +/-10% interval coverage is below 80%.  Counts below ``min_count`` are
    reported but cannot trigger the gate.
    """
    gt_arr = np.asarray(gt, dtype=np.float64).ravel()
    pred_arr = np.asarray(pred, dtype=np.float64).ravel()
    valid = np.isfinite(gt_arr) & (gt_arr > 0.0) & np.isfinite(pred_arr) & (pred_arr > 0.0)
    gt_arr = gt_arr[valid]
    pred_arr = pred_arr[valid]

    bins: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    triggered = False
    for name, mask in _bin_masks(gt_arr).items():
        count = int(np.sum(mask))
        if count == 0:
            bins[name] = {
                "count": 0,
                "relative_bias": float("nan"),
                "naive_10pct_coverage": float("nan"),
                "miscalibrated": False,
            }
            continue
        g = gt_arr[mask]
        p = pred_arr[mask]
        rel_bias = float(np.mean((p - g) / g))
        naive = float(np.mean((g >= 0.9 * p) & (g <= 1.1 * p)))
        bin_miscal = False
        bin_reasons: list[str] = []
        if count >= min_count:
            if abs(rel_bias) > max_abs_relative_bias:
                bin_miscal = True
                bin_reasons.append(
                    f"|relative bias| {abs(rel_bias):.3f} > {max_abs_relative_bias:.3f}"
                )
            if naive < min_naive_coverage:
                bin_miscal = True
                bin_reasons.append(
                    f"naive +/-10% coverage {naive:.3f} < {min_naive_coverage:.3f}"
                )
        if bin_miscal:
            triggered = True
            reasons.append(f"{name}: " + "; ".join(bin_reasons))
        bins[name] = {
            "count": count,
            "relative_bias": rel_bias,
            "naive_10pct_coverage": naive,
            "miscalibrated": bin_miscal,
            "reasons": bin_reasons,
        }

    return {
        "miscalibrated": triggered,
        "reasons": reasons,
        "bins": bins,
        "min_count": int(min_count),
        "max_abs_relative_bias": float(max_abs_relative_bias),
        "min_naive_coverage": float(min_naive_coverage),
        "scope": "held_out (out-of-fold predictions)",
    }


def _interval_report(
    gt: np.ndarray,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tags: Sequence[Mapping[str, str]],
    alpha: float,
) -> dict[str, Any]:
    valid = _valid_point_mask(gt, pred, lower, upper)
    gt_v = gt[valid]
    pred_v = pred[valid]
    lo_v = lower[valid]
    hi_v = upper[valid]
    if gt_v.size == 0:
        return {"count": 0, "metrics": {}, "per_bin": {}, "per_tag": {}}

    metrics = compute_interval_metrics(
        gt_v,
        lo_v,
        hi_v,
        nominal_coverage=1.0 - alpha,
        valid_mask=np.ones(gt_v.shape, dtype=bool),
        depths=gt_v,
    )
    per_bin = metrics.pop("per_bin", {})
    per_tag: dict[str, dict[str, Any]] = {}
    tag_keys = sorted({key for mapping in tags for key in mapping})
    for key in tag_keys:
        per_tag[key] = {}
        values = sorted({str(mapping.get(key, "unknown")) for mapping in tags})
        for value in values:
            subset = np.array(
                [
                    str(mapping.get(key, "unknown")) == value
                    for mapping in tags
                ],
                dtype=bool,
            )[valid]
            if not np.any(subset):
                continue
            sub_metrics = compute_interval_metrics(
                gt_v[subset],
                lo_v[subset],
                hi_v[subset],
                nominal_coverage=1.0 - alpha,
                valid_mask=np.ones(int(np.sum(subset)), dtype=bool),
                depths=gt_v[subset],
            )
            sub_metrics.pop("per_bin", None)
            sub_metrics.pop("per_tag", None)
            sub_metrics["count"] = int(np.sum(subset))
            per_tag[key][value] = sub_metrics

    return {
        "count": int(gt_v.size),
        "metrics": metrics,
        "per_bin": per_bin,
        "per_tag": per_tag,
    }


def _baseline_report(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict[str, Any]:
    valid = _valid_point_mask(gt, pred)
    gt_v = gt[valid]
    pred_v = pred[valid]
    overall = compute_depth_metrics(
        gt_v, pred_v, valid_mask=np.ones(gt_v.shape, dtype=bool)
    )
    per_bin = overall.pop("per_bin")
    per_bin_naive: dict[str, Any] = {}
    for name, mask in _bin_masks(gt_v).items():
        count = int(np.sum(mask))
        if count == 0:
            per_bin_naive[name] = {"count": 0, "coverage": float("nan"), "relative_bias": float("nan")}
            continue
        g = gt_v[mask]
        p = pred_v[mask]
        per_bin_naive[name] = {
            "count": count,
            "coverage": float(np.mean((g >= 0.9 * p) & (g <= 1.1 * p))),
            "relative_bias": float(np.mean((p - g) / g)),
        }
    return {
        "count": int(gt_v.size),
        "depth_metrics": overall,
        "per_bin": per_bin,
        "naive_10pct": {
            "overall_coverage": float(
                np.mean((gt_v >= 0.9 * pred_v) & (gt_v <= 1.1 * pred_v))
            ),
            "per_bin": per_bin_naive,
        },
    }


# ---------------------------------------------------------------------------
# Conformal evaluation
# ---------------------------------------------------------------------------
def _point_arrays(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> np.ndarray:
    return np.asarray([record[key] for record in records], dtype=np.float64)


def leave_one_route_out(
    records: Sequence[Mapping[str, Any]],
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Fit pooled conformal intervals with one route held out per fold."""
    routes = sorted({str(record["route_id"]) for record in records})
    fold_rows: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []

    for test_route in routes:
        train = [r for r in records if str(r["route_id"]) != test_route]
        test = [r for r in records if str(r["route_id"]) == test_route]
        if not train or not test:
            continue
        model = ConformalDepth(alpha=alpha).fit(
            _point_arrays(train, "pred_m"), _point_arrays(train, "gt_m")
        )
        train_pred = model.predict(_point_arrays(train, "pred_m"))
        test_pred = model.predict(_point_arrays(test, "pred_m"))

        for record, lower, upper in zip(train, train_pred.lower, train_pred.upper):
            row = dict(record)
            row["lower_m"] = float(lower)
            row["upper_m"] = float(upper)
            calibration_records.append(row)
        for record, lower, upper in zip(test, test_pred.lower, test_pred.upper):
            row = dict(record)
            row["lower_m"] = float(lower)
            row["upper_m"] = float(upper)
            heldout_records.append(row)

        fold_rows.append(
            {
                "test_route": test_route,
                "calibration_points": len(train),
                "held_out_points": len(test),
                "quantile": float(model.pooled_quantile_),
            }
        )

    return fold_rows, calibration_records, heldout_records


def _bootstrap_summary(
    records: Sequence[Mapping[str, Any]],
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    routes = sorted({str(record["route_id"]) for record in records})
    route_metrics: dict[str, dict[str, np.ndarray]] = {}
    for route in routes:
        subset = [r for r in records if str(r["route_id"]) == route]
        gt = _point_arrays(subset, "gt_m")
        pred = _point_arrays(subset, "pred_m")
        lo = _point_arrays(subset, "lower_m")
        hi = _point_arrays(subset, "upper_m")
        valid = _valid_point_mask(gt, pred, lo, hi)
        gt, pred, lo, hi = gt[valid], pred[valid], lo[valid], hi[valid]
        if gt.size == 0:
            continue
        route_metrics[route] = {
            "coverage": ((gt >= lo) & (gt <= hi)).astype(np.float64),
            "abs_rel": np.abs(gt - pred) / gt,
            "width": hi - lo,
            "interval_score": _interval_score(gt, lo, hi, alpha),
            "naive_coverage": ((gt >= 0.9 * pred) & (gt <= 1.1 * pred)).astype(np.float64),
        }

    output: dict[str, Any] = {}
    for metric in ("coverage", "abs_rel", "width", "interval_score", "naive_coverage"):
        route_values = {
            route: values[metric]
            for route, values in route_metrics.items()
            if values[metric].size
        }
        result = bootstrap_routes(
            route_values,
            metric_fn=lambda values: float(np.mean(values)),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        output[metric] = {
            "mean": result["mean"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "n_routes": len(route_values),
        }

    if route_metrics:
        coverage_means = [float(np.mean(values["coverage"])) for values in route_metrics.values()]
        output["route_coverage_spread"] = {
            "min": float(np.min(coverage_means)),
            "max": float(np.max(coverage_means)),
            "std": float(np.std(coverage_means)),
            "n_routes": len(coverage_means),
        }
    return output


# ---------------------------------------------------------------------------
# Floor evaluation
# ---------------------------------------------------------------------------
def _floor_rows(
    frames: Sequence[Mapping[str, Any]],
    floor_probs: Sequence[np.ndarray],
    thresholds: Sequence[float],
    max_depth_m: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame, probs in zip(frames, floor_probs):
        gt = np.asarray(frame["mask"])
        sky = np.asarray(frame.get("sky_mask", np.zeros_like(gt, dtype=bool)), dtype=bool)
        depth = np.asarray(frame["depth"], dtype=np.float64)
        loss_row = np.full(len(thresholds), np.nan, dtype=np.float64)
        false_row = np.zeros(len(thresholds), dtype=np.float64)
        nonfloor_row = np.zeros(len(thresholds), dtype=np.float64)
        pred_row = np.zeros(len(thresholds), dtype=np.float64)
        floor_row = np.zeros(len(thresholds), dtype=np.float64)
        true_row = np.zeros(len(thresholds), dtype=np.float64)
        for index, threshold in enumerate(thresholds):
            result = floor_loss(
                probs,
                gt,
                threshold,
                sky_mask=sky,
                depth=depth,
                max_depth_m=max_depth_m,
            )
            loss_row[index] = result["loss"]
            false_row[index] = result["false_positive_count"]
            nonfloor_row[index] = result["nonfloor_count"]
            pred_row[index] = result["predicted_floor_count"]
            floor_row[index] = result["floor_count"]
            true_row[index] = result["predicted_floor_count"] - result["false_positive_count"]
        rows.append(
            {
                "route_id": str(frame["route_id"]),
                "frame_id": str(frame["record"].frame_id),
                "loss": loss_row,
                "false_positive": false_row,
                "nonfloor": nonfloor_row,
                "predicted_floor": pred_row,
                "floor": floor_row,
                "true_positive": true_row,
            }
        )
    return rows


def _floor_metrics_at(
    rows: Sequence[Mapping[str, Any]],
    threshold_index: int,
) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    false = float(sum(row["false_positive"][threshold_index] for row in rows))
    nonfloor = float(sum(row["nonfloor"][threshold_index] for row in rows))
    predicted = float(sum(row["predicted_floor"][threshold_index] for row in rows))
    floor = float(sum(row["floor"][threshold_index] for row in rows))
    true = float(sum(row["true_positive"][threshold_index] for row in rows))
    return {
        "n_images": len(rows),
        "false_positive_rate": false / nonfloor if nonfloor > 0 else float("nan"),
        "false_walkable_rate": false / predicted if predicted > 0 else float("nan"),
        "floor_recall": true / floor if floor > 0 else float("nan"),
        "false_positive_pixels": false,
        "nonfloor_pixels": nonfloor,
        "predicted_floor_pixels": predicted,
        "floor_pixels": floor,
    }


def evaluate_floor_loro(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    delta: float,
) -> dict[str, Any]:
    """Leave-one-route-out certified-floor evaluation from precomputed rows."""
    routes = sorted({str(row["route_id"]) for row in rows})
    fold_results: list[dict[str, Any]] = []
    for test_route in routes:
        train = [row for row in rows if str(row["route_id"]) != test_route]
        test = [row for row in rows if str(row["route_id"]) == test_route]
        train_matrix = np.vstack([row["loss"] for row in train]) if train else np.empty((0, len(thresholds)))
        usable = np.any(np.isfinite(train_matrix), axis=1)
        threshold, _, _ = select_floor_threshold(
            train_matrix[usable], delta, thresholds
        )
        if math.isfinite(threshold):
            threshold_index = int(np.argmin(np.abs(np.asarray(thresholds) - threshold)))
            test_metrics = _floor_metrics_at(test, threshold_index)
        else:
            test_metrics = {"n_images": len(test), "error": "no threshold certified"}
        fold_results.append(
            {
                "test_route": test_route,
                "calibration_images": int(np.sum(usable)),
                "threshold": threshold if math.isfinite(threshold) else None,
                **test_metrics,
            }
        )

    all_matrix = np.vstack([row["loss"] for row in rows]) if rows else np.empty((0, len(thresholds)))
    usable = np.any(np.isfinite(all_matrix), axis=1)
    threshold, mean_losses, risk_bounds = select_floor_threshold(
        all_matrix[usable], delta, thresholds
    )
    threshold_index = (
        int(np.argmin(np.abs(np.asarray(thresholds) - threshold)))
        if math.isfinite(threshold)
        else None
    )
    return {
        "available": bool(rows and np.any(usable)),
        "delta": float(delta),
        "threshold": threshold if math.isfinite(threshold) else None,
        "full_calibration": _floor_metrics_at(rows, threshold_index) if threshold_index is not None else {},
        "folds": fold_results,
        "mean_losses_sample": [float(x) for x in mean_losses[::10]],
        "risk_bounds_sample": [float(x) for x in risk_bounds[::10]],
        "note": "thresholds and delta are TO TUNE ON REAL DATA",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    dataset_root: str | Path,
    out_dir: str | Path,
    *,
    fake: bool,
    intrinsics_path: str | Path | None,
    weights_dir: str | Path,
    depth_variant: str,
    alpha: float,
    delta: float,
    seed: int,
    bootstrap_samples: int,
    floor_max_depth_m: float | None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run the evaluation and return the report payload."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames, warnings = load_annotated_frames(dataset_root)
    routes = sorted({str(frame["route_id"]) for frame in frames})
    if len(routes) < MIN_ROUTES:
        raise ValueError(
            f"Need at least {MIN_ROUTES} routes for leave-one-route-out evaluation; "
            f"found {len(routes)} ({routes}). Add routes or use --fake --routes {MIN_ROUTES}."
        )
    if max_frames is not None:
        frames = frames[: int(max_frames)]

    if intrinsics_path is not None:
        intrinsics = load_intrinsics_json(intrinsics_path)
    else:
        if not fake:
            raise ValueError("--intrinsics is required for real evaluation")
        intrinsics = intrinsics_from_frame(frames[0]["record"])

    for index, frame in enumerate(frames):
        frame["route_index"] = routes.index(str(frame["route_id"]))
        frame["frame_index"] = index
        frame["sky_mask"] = np.zeros(np.asarray(frame["mask"]).shape, dtype=bool)
        if tuple(np.asarray(frame["depth"]).shape) != (intrinsics.height, intrinsics.width):
            warnings.append(
                f"{frame['record'].frame_id}: depth shape {np.asarray(frame['depth']).shape} "
                f"does not match intrinsics {(intrinsics.height, intrinsics.width)}"
            )

    depth_backend = None
    seg_backend = None
    if not fake:
        try:
            depth_backend, seg_backend = _real_models(Path(weights_dir), depth_variant)
        except Exception as exc:  # noqa: BLE001 - clear failure with fetch guidance
            raise RuntimeError(
                f"Could not construct real model wrappers: {exc}. "
                "Fetch weights with python scripts/fetch_models.py --dry-run then --yes."
            ) from exc

    points: list[dict[str, Any]] = []
    floor_probs: list[np.ndarray] = []
    floor_frames: list[dict[str, Any]] = []
    model_errors: list[str] = []

    for frame in frames:
        record: FrameRecord = frame["record"]
        depth_pred: np.ndarray | None = None
        floor_prob: np.ndarray | None = None
        if fake:
            depth_pred = _fake_depth_for_frame(
                np.asarray(frame["depth"], dtype=np.float64),
                int(frame["route_index"]),
                int(frame["frame_index"]),
                seed,
            )
            floor_prob = _fake_floor_probs(
                np.asarray(frame["mask"]),
                int(frame["route_index"]),
                int(frame["frame_index"]),
                seed,
            )
        else:
            rgb_path = Path(frame["rgb_path"])
            if not rgb_path.exists():
                model_errors.append(f"{record.frame_id}: RGB missing at {rgb_path}")
                continue
            image = _load_rgb(rgb_path)
            if image.shape[:2] != (intrinsics.height, intrinsics.width):
                model_errors.append(
                    f"{record.frame_id}: RGB shape {image.shape[:2]} != intrinsics "
                    f"{(intrinsics.height, intrinsics.width)}"
                )
                continue
            try:
                depth_pred = np.asarray(depth_backend.predict(image).depth_m, dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                model_errors.append(f"{record.frame_id}: depth model failed: {exc}")
                continue
            if seg_backend is not None:
                try:
                    seg = seg_backend.predict(image)
                    floor_prob = np.asarray(seg.probs[..., 0], dtype=np.float64)
                    frame["sky_mask"] = np.asarray(seg.sky_mask, dtype=bool)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"{record.frame_id}: segmentation failed: {exc}")

        if depth_pred is None:
            continue
        if depth_pred.shape != np.asarray(frame["depth"]).shape:
            warnings.append(
                f"{record.frame_id}: model depth shape {depth_pred.shape} != dataset "
                f"{np.asarray(frame['depth']).shape}"
            )
            continue

        dp = frame["distance_points"]
        for u, v, gt in zip(dp["u"], dp["v"], dp["distance_m"]):
            pred = median_patch(depth_pred, float(u), float(v))
            points.append(
                {
                    "frame_id": record.frame_id,
                    "route_id": record.route_id,
                    "u": float(u),
                    "v": float(v),
                    "gt_m": float(gt),
                    "pred_m": pred,
                    "tags": dict(frame["tags"]),
                }
            )

        if floor_prob is not None:
            floor_probs.append(floor_prob)
            floor_frames.append(frame)

    if not points:
        raise RuntimeError("No annotated distance points produced a depth prediction")

    fold_rows, calibration_records, heldout_records = leave_one_route_out(points, alpha)
    heldout_gt = _point_arrays(heldout_records, "gt_m")
    heldout_pred = _point_arrays(heldout_records, "pred_m")
    heldout_lo = _point_arrays(heldout_records, "lower_m")
    heldout_hi = _point_arrays(heldout_records, "upper_m")
    heldout_tags = [record["tags"] for record in heldout_records]

    calib_gt = _point_arrays(calibration_records, "gt_m")
    calib_pred = _point_arrays(calibration_records, "pred_m")
    calib_lo = _point_arrays(calibration_records, "lower_m")
    calib_hi = _point_arrays(calibration_records, "upper_m")
    calib_tags = [record["tags"] for record in calibration_records]

    heldout_intervals = _interval_report(
        heldout_gt, heldout_pred, heldout_lo, heldout_hi, heldout_tags, alpha
    )
    calibration_intervals = _interval_report(
        calib_gt, calib_pred, calib_lo, calib_hi, calib_tags, alpha
    )
    heldout_baseline = _baseline_report(heldout_gt, heldout_pred)
    g1 = evaluate_g1(heldout_gt, heldout_pred)

    full_model = ConformalDepth(alpha=alpha).fit(
        _point_arrays(points, "pred_m"), _point_arrays(points, "gt_m")
    )
    (out / "conformal_depth.json").write_text(
        json.dumps(full_model.to_dict(), indent=2), encoding="utf-8"
    )

    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in heldout_records:
            handle.write(
                json.dumps(
                    {
                        "frame_id": record["frame_id"],
                        "route_id": record["route_id"],
                        "u": record["u"],
                        "v": record["v"],
                        "gt_m": record["gt_m"],
                        "pred_m": record["pred_m"],
                        "lower_m": record["lower_m"],
                        "upper_m": record["upper_m"],
                        "covered": bool(
                            record["gt_m"] >= record["lower_m"]
                            and record["gt_m"] <= record["upper_m"]
                        ),
                        "tags": record["tags"],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    bootstrap = _bootstrap_summary(
        heldout_records, alpha, n_bootstrap=bootstrap_samples, seed=seed
    )

    floor_report: dict[str, Any]
    if not floor_frames:
        floor_report = {
            "available": False,
            "reason": "no segmentation probabilities (missing model or labelled masks)",
        }
    else:
        floor_rows = _floor_rows(
            floor_frames, floor_probs, DEFAULT_LAMBDA_GRID, floor_max_depth_m
        )
        floor_report = evaluate_floor_loro(floor_rows, DEFAULT_LAMBDA_GRID, delta)
        floor_report["n_frames"] = len(floor_frames)

    report: dict[str, Any] = {
        "generated": datetime.datetime.now().isoformat(),
        "mode": "fake" if fake else "real",
        "dataset": str(Path(dataset_root)),
        "n_routes": len(routes),
        "routes": routes,
        "n_frames": len(frames),
        "n_points": len(points),
        "n_held_out_points": len(heldout_records),
        "intrinsics": {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "width": intrinsics.width,
            "height": intrinsics.height,
        },
        "conformal": {
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "score": "two_sided_relative",
            "folds": fold_rows,
            "full_model": full_model.to_dict(),
        },
        "calibration_set": {
            "scope": "in-sample (optimistic; not the G1 gate)",
            "intervals": calibration_intervals,
        },
        "held_out": {
            "scope": "leave-one-route-out out-of-fold",
            "intervals": heldout_intervals,
            "baseline": heldout_baseline,
        },
        "bootstrap": {
            "scope": "route-level cluster bootstrap on held-out points",
            "n_bootstrap": int(bootstrap_samples),
            "metrics": bootstrap,
        },
        "g1": g1,
        "floor": floor_report,
        "files": {
            "predictions_jsonl": str(out / "predictions.jsonl"),
            "report_json": str(out / "calibration_report.json"),
            "conformal_depth_json": str(out / "conformal_depth.json"),
        },
        "warnings": warnings,
        "model_errors": model_errors,
        "thresholds_status": "TO TUNE ON REAL DATA",
    }

    (out / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _build_fake_dataset(root: Path, routes: int, frames_per_route: int, seed: int) -> Path:
    build_synthetic_dataset(
        out_dir=root,
        n_routes=int(routes),
        frames_per_route=int(frames_per_route),
        seed=int(seed),
        image_size=(320, 240),
        hfov_deg=75.0,
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Conformal depth and certified-floor evaluation (real or fake)"
    )
    parser.add_argument("--dataset", default=None, help="dataset root with route manifests")
    parser.add_argument("--intrinsics", default=None, help="camera_intrinsics.json")
    parser.add_argument("--out", default="reports", help="output directory")
    parser.add_argument("--fake", action="store_true", help="deterministic synthetic backend")
    parser.add_argument("--routes", type=int, default=6, help="fake mode route count")
    parser.add_argument(
        "--frames-per-route", type=int, default=4, help="fake mode frames per route"
    )
    parser.add_argument("--weights-dir", default=str(REPO_ROOT / "weights"))
    parser.add_argument("--depth-variant", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--floor-max-depth-m",
        type=float,
        default=None,
        help="optional calibration restriction to GT depth <= R (TO TUNE)",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    if args.fake and args.routes < MIN_ROUTES:
        print(
            f"ERROR: at least {MIN_ROUTES} routes are required; got {args.routes}. "
            "This guard applies to fake and real evaluation.",
            file=sys.stderr,
        )
        return 2

    temp_context = None
    dataset_root = args.dataset
    if args.fake:
        if dataset_root is None:
            temp_context = tempfile.TemporaryDirectory(prefix="visiona_fake_calibration_")
            dataset_root = _build_fake_dataset(
                Path(temp_context.name), args.routes, args.frames_per_route, args.seed
            )
    elif dataset_root is None:
        print("ERROR: --dataset is required unless --fake is used", file=sys.stderr)
        return 2

    try:
        report = run(
            dataset_root,
            args.out,
            fake=bool(args.fake),
            intrinsics_path=args.intrinsics,
            weights_dir=args.weights_dir,
            depth_variant=args.depth_variant,
            alpha=float(args.alpha),
            delta=float(args.delta),
            seed=int(args.seed),
            bootstrap_samples=int(args.bootstrap),
            floor_max_depth_m=args.floor_max_depth_m,
            max_frames=args.max_frames,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    print(
        f"{report['mode']} evaluation: {report['n_routes']} routes, "
        f"{report['n_frames']} frames, {report['n_points']} points"
    )
    print(
        f"held-out interval coverage={report['held_out']['intervals']['metrics'].get('coverage', float('nan')):.3f}, "
        f"mean width={report['held_out']['intervals']['metrics'].get('mean_width', float('nan')):.3f} m"
    )
    print(f"G1 miscalibrated={report['g1']['miscalibrated']} reasons={report['g1']['reasons']}")
    print(f"Wrote {report['files']['predictions_jsonl']}")
    print(f"Wrote {report['files']['report_json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
