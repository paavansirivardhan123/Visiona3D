"""Safety-first tuning of SegFormer + promptable-mask floor refinement.

The tool splits labelled frames into calibration and held-out sets, runs the
segmenter and the promptable mask once per frame, and sweeps
``min_accept_iou`` x ``veto_prob`` using the production ``apply_floor_mask``
decision core. Thresholds are selected on calibration frames only; the
held-out split is reported afterwards and never influences the choice.

Selection policy (safety first):

1. A grid configuration is eligible only if, on calibration frames, its floor
   IoU is at least the SegFormer-only floor IoU and neither
   ``unsafe_free_rate`` nor ``wall_as_floor_rate`` is above SegFormer-only.
2. Eligible configurations are ordered by lowest unsafe-free rate, lowest
   wall-as-floor rate, highest floor IoU, highest floor recall, then the most
   conservative thresholds (higher acceptance IoU, higher veto probability).
   Availability (accepted-mask rate) is only a final tie-break.
3. If no configuration is eligible, or the winner shows no measured
   improvement over SegFormer-only, the tool selects SegFormer-only.

Splits are deterministic for a given frame list and seed. With ``--route-map``
all frames of a route stay on one side of the split.

Every numeric default here is a starting point, not a validated navigation
setting. TO TUNE ON REAL DATA.

Usage:
    python scripts/tune_fusion.py --fake --images path/to/frames \
        --out reports/fusion_tuning
    python scripts/tune_fusion.py --images path/to/frames \
        --labels-dir path/to/labels --route-map routes.json \
        --out reports/fusion_tuning

Fake mode renders analytic scenes and needs no weights. Real mode loads
local/cached weights only and exits with code 2 (printing what to fetch) when
they are absent. Labels are mandatory in real mode.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import zlib

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    Intrinsics,
    MASK_FLOOR,
    MASK_OBSTACLE,
    MASK_OTHER,
    MASK_WALL,
)
from perception.interfaces import SegOutput  # noqa: E402
from perception.seg_fusion import (  # noqa: E402
    DEFAULT_EROSION_PX,
    DEFAULT_MAX_PROMPTS,
    DEFAULT_MIN_ACCEPT_IOU,
    DEFAULT_MIN_FLOOR_PROB,
    DEFAULT_MIN_PROMPT_SPACING_PX,
    DEFAULT_VETO_PROB,
    PromptMaskOutput,
    apply_floor_mask,
    select_floor_prompts,
)
from scripts.eval_frames import (  # noqa: E402
    LABEL_SUFFIXES,
    _check_weights,
    _find_sidecar,
    _load_labels,
    label_metrics,
)
from scripts.real_models_smoke import IMAGE_SUFFIXES, load_image  # noqa: E402

# Every numeric default below is TO TUNE ON REAL DATA.
DEFAULT_OUT = "reports/fusion_tuning"
DEFAULT_IOU_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
DEFAULT_VETO_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
DEFAULT_CALIBRATION_FRACTION = 0.5
DEFAULT_MIN_FRAMES = 6
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_FAKE_WALL_AS_FLOOR = 0.25

METRIC_FIELDS = (
    "floor_iou",
    "wall_iou",
    "mean_iou",
    "unsafe_free_rate",
    "wall_as_floor_rate",
    "floor_recall",
    "floor_fraction",
)
CI_FIELDS = ("unsafe_free_rate", "wall_as_floor_rate", "floor_iou", "floor_recall")
VARIANT_FIELDS = ("segformer", "fusion_default", "fusion_selected")
CSV_FIELDS = (
    "frame",
    "split",
    "route",
    "variant",
    "min_accept_iou",
    "veto_prob",
    "floor_iou",
    "wall_iou",
    "mean_iou",
    "unsafe_free_rate",
    "wall_as_floor_rate",
    "floor_recall",
    "floor_fraction",
    "floor_pixels",
    "accepted",
    "reason",
    "prompt_count",
    "sam_iou",
    "latency_ms",
)
PROMPT_KWARGS = {
    "min_floor_prob": DEFAULT_MIN_FLOOR_PROB,
    "erosion_px": DEFAULT_EROSION_PX,
    "max_points": DEFAULT_MAX_PROMPTS,
    "min_spacing_px": DEFAULT_MIN_PROMPT_SPACING_PX,
}
EPS = 1e-9


@dataclass
class Split:
    """Deterministic calibration/held-out split."""

    calibration: list[str]
    heldout: list[str]
    strategy: str
    calibration_routes: list[str] = field(default_factory=list)
    heldout_routes: list[str] = field(default_factory=list)


@dataclass
class CachedFrame:
    """Per-frame predictions with the threshold-independent caches."""

    name: str
    route: str | None
    image_shape: tuple[int, int]
    base: SegOutput
    base_metrics: dict[str, Any]
    prompts: np.ndarray
    sam_mask: np.ndarray | None
    sam_score: float | None
    sam_reason: str | None
    iou: float | None
    seg_ms: float
    sam_ms: float
    veto_metrics: dict[float, dict[str, Any]] = field(default_factory=dict)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_present(values: Sequence[Any]) -> float | None:
    finite = [_finite(value) for value in values]
    present = [value for value in finite if value is not None]
    return float(np.mean(present)) if present else None


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        field: _mean_present([row.get(field) for row in rows])
        for field in METRIC_FIELDS
    }
    n_frames = len(rows)
    n_accepted = sum(1 for row in rows if bool(row.get("accepted")))
    aggregate["n_frames"] = n_frames
    aggregate["n_accepted"] = n_accepted
    aggregate["accepted_rate"] = (
        float(n_accepted / n_frames) if n_frames else None
    )
    return aggregate


def _parse_grid(text: str, name: str) -> tuple[float, ...]:
    try:
        values = [float(part) for part in str(text).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated numbers") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(not (0.0 <= value <= 1.0) for value in values):
        raise ValueError(f"{name} values must be in [0, 1]")
    return tuple(sorted(set(values)))


def split_frames(
    names: Sequence[str],
    *,
    calibration_fraction: float,
    seed: int,
    route_map: dict[str, str] | None = None,
) -> Split:
    """Split frame names deterministically, optionally keeping routes intact."""
    ordered = sorted(set(str(name) for name in names))
    if len(ordered) < 2:
        raise ValueError("need at least two frames to split")
    fraction = float(calibration_fraction)
    if not (0.0 < fraction < 1.0):
        raise ValueError(
            f"calibration_fraction must be strictly between 0 and 1, got {fraction!r}"
        )
    target = int(round(fraction * len(ordered)))
    target = min(max(target, 1), len(ordered) - 1)
    rng = np.random.default_rng(int(seed))

    if route_map is None:
        order = rng.permutation(len(ordered))
        calibration = sorted(ordered[int(index)] for index in order[:target])
        heldout = sorted(ordered[int(index)] for index in order[target:])
        return Split(calibration=calibration, heldout=heldout, strategy="frame")

    groups: dict[str, list[str]] = {}
    for name in ordered:
        route = route_map.get(name)
        if not isinstance(route, str) or not route.strip():
            raise ValueError(f"route map has no non-empty route for frame {name!r}")
        groups.setdefault(route.strip(), []).append(name)
    routes = sorted(groups)
    if len(routes) < 2:
        raise ValueError("a route split needs at least two distinct routes")

    shuffled = [routes[int(index)] for index in rng.permutation(len(routes))]
    # 0/1 subset sum keeps whole routes together while staying near the target.
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, route in enumerate(shuffled):
        count = len(groups[route])
        for total in sorted(reachable, reverse=True):
            candidate = total + count
            if candidate <= len(ordered) - 1 and candidate not in reachable:
                reachable[candidate] = reachable[total] + (index,)
    options = [
        (abs(total - target), total, picks)
        for total, picks in reachable.items()
        if total >= 1
    ]
    if not options:
        raise ValueError("a single route contains every frame; cannot split routes")
    _, _, picks = min(options)
    calibration_routes = sorted(shuffled[index] for index in picks)
    heldout_routes = sorted(set(routes) - set(calibration_routes))
    calibration = sorted(
        name for route in calibration_routes for name in groups[route]
    )
    heldout = sorted(name for route in heldout_routes for name in groups[route])
    if not calibration or not heldout:
        raise ValueError("route split produced an empty side")
    return Split(
        calibration=calibration,
        heldout=heldout,
        strategy="route",
        calibration_routes=calibration_routes,
        heldout_routes=heldout_routes,
    )


def _describe_exception(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {message[:160]}"


def cache_frame(
    name: str,
    route: str | None,
    image: np.ndarray,
    gt_labels: np.ndarray,
    seg_backend: Any,
    sam_backend: Any,
    *,
    veto_probs: Sequence[float],
    min_accept_floor: float,
) -> CachedFrame:
    """Run segmentation and one promptable query, then cache veto variants."""
    seg_start = time.perf_counter()
    base = seg_backend.predict(image)
    seg_ms = (time.perf_counter() - seg_start) * 1000.0
    base_errors = base.validate()
    if base_errors:
        raise ValueError("base segmentation is invalid: " + "; ".join(base_errors))
    if base.labels.shape != gt_labels.shape:
        raise ValueError(
            f"segmentation shape {base.labels.shape} != ground truth {gt_labels.shape}"
        )
    base_metrics = label_metrics(base.labels, gt_labels)

    prompts = select_floor_prompts(base.labels, base.probs, **PROMPT_KWARGS)
    sam_mask: np.ndarray | None = None
    sam_score: float | None = None
    sam_reason: str | None = None
    sam_ms = 0.0
    iou: float | None = None

    if prompts.shape[0] == 0:
        sam_reason = "no_prompts"
    else:
        sam_start = time.perf_counter()
        try:
            candidate: PromptMaskOutput = sam_backend.predict(image, prompts)
        except Exception as exc:  # noqa: BLE001 - a mask failure is a fallback
            sam_ms = (time.perf_counter() - sam_start) * 1000.0
            sam_reason = "sam_error:" + _describe_exception(exc)
        else:
            sam_ms = (time.perf_counter() - sam_start) * 1000.0
            mask_errors = candidate.validate(image.shape[:2])
            if mask_errors:
                sam_reason = "invalid_sam:" + "; ".join(mask_errors)
            else:
                mask = np.asarray(candidate.mask, dtype=bool)
                sam_score = _finite(candidate.score)
                if not mask.any():
                    sam_reason = "empty_sam"
                else:
                    sam_mask = mask
                    base_floor = np.asarray(base.labels) == MASK_FLOOR
                    union = int(np.count_nonzero(base_floor | mask))
                    intersection = int(np.count_nonzero(base_floor & mask))
                    iou = float(intersection / union) if union else float("nan")

    veto_metrics: dict[float, dict[str, Any]] = {}
    if sam_mask is not None and iou is not None and iou >= float(min_accept_floor):
        for veto in sorted(set(float(value) for value in veto_probs)):
            decision = apply_floor_mask(
                base,
                sam_mask,
                min_accept_iou=0.0,
                veto_prob=veto,
            )
            if not decision.accepted or decision.seg is None:
                raise RuntimeError("cached mask was unexpectedly rejected at IoU 0")
            fused_errors = decision.seg.validate()
            if fused_errors:
                raise ValueError(
                    "fused segmentation is invalid: " + "; ".join(fused_errors)
                )
            veto_metrics[veto] = label_metrics(decision.seg.labels, gt_labels)

    return CachedFrame(
        name=name,
        route=route,
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        base=base,
        base_metrics=base_metrics,
        prompts=prompts,
        sam_mask=sam_mask,
        sam_score=sam_score,
        sam_reason=sam_reason,
        iou=_finite(iou),
        seg_ms=seg_ms,
        sam_ms=sam_ms,
        veto_metrics=veto_metrics,
    )


def _metrics_for(
    frame: CachedFrame,
    min_accept_iou: float,
    veto_prob: float,
) -> tuple[dict[str, Any], bool, str]:
    """Return (metrics, accepted, reason) for one configuration on one frame."""
    if (
        frame.sam_mask is not None
        and frame.iou is not None
        and frame.iou >= float(min_accept_iou)
    ):
        metrics = frame.veto_metrics.get(float(veto_prob))
        if metrics is None:
            raise RuntimeError(
                f"frame {frame.name}: veto {veto_prob} was not cached; "
                "this is a tuning-tool bug"
            )
        return metrics, True, "accepted"
    if frame.sam_mask is None:
        return frame.base_metrics, False, f"fallback:{frame.sam_reason}"
    assert frame.iou is not None
    return frame.base_metrics, False, f"fallback:low_iou:{frame.iou:.4f}"


def _decision_rows(
    frames: dict[str, CachedFrame],
    names: Sequence[str],
    *,
    variant: str,
    min_accept_iou: float | None,
    veto_prob: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        frame = frames[name]
        if min_accept_iou is None or veto_prob is None:
            metrics = frame.base_metrics
            accepted = False
            reason = "segformer_only"
            prompt_count = 0
            sam_iou = None
            latency_ms = frame.seg_ms
        else:
            metrics, accepted, reason = _metrics_for(
                frame, min_accept_iou, veto_prob
            )
            prompt_count = int(frame.prompts.shape[0])
            sam_iou = frame.iou
            latency_ms = frame.seg_ms + frame.sam_ms
        row = {
            "frame": name,
            "variant": variant,
            "min_accept_iou": min_accept_iou,
            "veto_prob": veto_prob,
            **{field: metrics.get(field) for field in METRIC_FIELDS},
            "floor_pixels": metrics.get("floor_pixels"),
            "accepted": bool(accepted),
            "reason": reason,
            "prompt_count": prompt_count,
            "sam_iou": sam_iou,
            "latency_ms": latency_ms,
        }
        rows.append(row)
    return rows


def _eligibility(
    row: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    floor_iou = _finite(row.get("floor_iou"))
    baseline_floor_iou = _finite(baseline.get("floor_iou"))
    if baseline_floor_iou is not None:
        if floor_iou is None or floor_iou < baseline_floor_iou - EPS:
            reasons.append("floor_iou_below_segformer")
    for field, label in (
        ("unsafe_free_rate", "unsafe_free_rate_above_segformer"),
        ("wall_as_floor_rate", "wall_as_floor_rate_above_segformer"),
    ):
        value = _finite(row.get(field))
        reference = _finite(baseline.get(field))
        if reference is not None and (value is None or value > reference + EPS):
            reasons.append(label)
    return (not reasons), reasons


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    unsafe = _finite(row.get("unsafe_free_rate"))
    wall = _finite(row.get("wall_as_floor_rate"))
    floor_iou = _finite(row.get("floor_iou"))
    floor_recall = _finite(row.get("floor_recall"))
    accepted_rate = _finite(row.get("accepted_rate"))
    return (
        round(unsafe, 12) if unsafe is not None else math.inf,
        round(wall, 12) if wall is not None else math.inf,
        -round(floor_iou, 12) if floor_iou is not None else -math.inf,
        -round(floor_recall, 12) if floor_recall is not None else -math.inf,
        -float(row["min_accept_iou"]),
        -float(row["veto_prob"]),
        -round(accepted_rate, 12) if accepted_rate is not None else -math.inf,
    )


def _metric_view(row: dict[str, Any]) -> dict[str, Any]:
    view = {field: row.get(field) for field in METRIC_FIELDS}
    view["n_frames"] = row.get("n_frames")
    view["n_accepted"] = row.get("n_accepted")
    view["accepted_rate"] = row.get("accepted_rate")
    return view


def _improves(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    checks = []
    for field in ("unsafe_free_rate", "wall_as_floor_rate"):
        value = _finite(row.get(field))
        reference = _finite(baseline.get(field))
        if reference is not None:
            checks.append(value is not None and value < reference - EPS)
    for field in ("floor_iou", "floor_recall"):
        value = _finite(row.get(field))
        reference = _finite(baseline.get(field))
        if reference is not None:
            checks.append(value is not None and value > reference + EPS)
    return any(checks)


def _segformer_selection(baseline: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "variant": "segformer_only",
        "min_accept_iou": None,
        "veto_prob": None,
        "reason": reason,
        "eligible": False,
        "improved": False,
        "calibration_metrics": _metric_view(baseline),
        "best_candidate": None,
    }


def select_candidate(
    rows: Sequence[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Choose a grid configuration from calibration rows only."""
    eligible = [row for row in rows if row.get("eligible")]
    if not eligible:
        return _segformer_selection(
            baseline, "no_grid_config_matched_segformer_floor_iou_and_safety"
        )
    best = min(eligible, key=_selection_key)
    if not _improves(best, baseline):
        selection = _segformer_selection(
            baseline, "best_eligible_config_did_not_improve_on_calibration"
        )
        selection["best_candidate"] = {
            "min_accept_iou": best["min_accept_iou"],
            "veto_prob": best["veto_prob"],
            "accepted_rate": best.get("accepted_rate"),
            "calibration_metrics": _metric_view(best),
        }
        return selection
    return {
        "variant": "fusion",
        "min_accept_iou": float(best["min_accept_iou"]),
        "veto_prob": float(best["veto_prob"]),
        "reason": "calibration_safety_first_winner",
        "eligible": True,
        "improved": True,
        "calibration_metrics": _metric_view(best),
        "best_candidate": {
            "min_accept_iou": best["min_accept_iou"],
            "veto_prob": best["veto_prob"],
            "accepted_rate": best.get("accepted_rate"),
            "calibration_metrics": _metric_view(best),
        },
    }


def _bootstrap_ci(
    values: Sequence[Any],
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    finite = [_finite(value) for value in values]
    array = np.asarray(
        [value for value in finite if value is not None], dtype=np.float64
    )
    if array.size == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    mean = float(array.mean())
    if array.size == 1:
        return {"mean": mean, "ci95": [mean, mean], "n": 1}
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, array.size, size=(int(n_resamples), array.size))
    means = array[draw].mean(axis=1)
    return {
        "mean": mean,
        "ci95": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "n": int(array.size),
    }


def _heldout_aggregate(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    aggregate = _aggregate(rows)
    aggregate["ci95"] = {
        field: _bootstrap_ci(
            [row.get(field) for row in rows],
            seed=seed + index,
            n_resamples=n_resamples,
        )
        for index, field in enumerate(CI_FIELDS)
    }
    return aggregate


def _build_heldout(
    frames: dict[str, CachedFrame],
    split: Split,
    selection: dict[str, Any],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variants: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    configs = {
        "segformer": (None, None),
        "fusion_default": (DEFAULT_MIN_ACCEPT_IOU, DEFAULT_VETO_PROB),
        "fusion_selected": (
            selection.get("min_accept_iou"),
            selection.get("veto_prob"),
        ),
    }
    for index, variant in enumerate(VARIANT_FIELDS):
        iou, veto = configs[variant]
        variant_rows = _decision_rows(
            frames,
            split.heldout,
            variant=variant,
            min_accept_iou=iou,
            veto_prob=veto,
        )
        for row in variant_rows:
            row["split"] = "heldout"
            row["route"] = frames[row["frame"]].route
        variants[variant] = _heldout_aggregate(
            variant_rows,
            seed=bootstrap_seed + 100 * index,
            n_resamples=bootstrap_resamples,
        )
        rows.extend(variant_rows)
    return {
        "n_frames": len(split.heldout),
        "variants": variants,
    }, rows


def _build_calibration_rows(
    frames: dict[str, CachedFrame],
    split: Split,
    iou_grid: Sequence[float],
    veto_grid: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = _decision_rows(
        frames,
        split.calibration,
        variant="segformer",
        min_accept_iou=None,
        veto_prob=None,
    )
    baseline = _aggregate(baseline_rows)
    default_rows = _decision_rows(
        frames,
        split.calibration,
        variant="fusion_default",
        min_accept_iou=DEFAULT_MIN_ACCEPT_IOU,
        veto_prob=DEFAULT_VETO_PROB,
    )
    default_aggregate = _aggregate(default_rows)

    candidates: list[dict[str, Any]] = []
    for iou in iou_grid:
        for veto in veto_grid:
            rows = _decision_rows(
                frames,
                split.calibration,
                variant="fusion",
                min_accept_iou=float(iou),
                veto_prob=float(veto),
            )
            aggregate = _aggregate(rows)
            eligible, reasons = _eligibility(aggregate, baseline)
            candidates.append(
                {
                    "min_accept_iou": float(iou),
                    "veto_prob": float(veto),
                    "is_default": bool(
                        math.isclose(float(iou), DEFAULT_MIN_ACCEPT_IOU, abs_tol=EPS)
                        and math.isclose(
                            float(veto), DEFAULT_VETO_PROB, abs_tol=EPS
                        )
                    ),
                    **aggregate,
                    "eligible": bool(eligible),
                    "ineligible_reasons": reasons,
                }
            )
    calibration = {
        "n_frames": len(split.calibration),
        "segformer": baseline,
        "fusion_default": default_aggregate,
        "n_eligible": sum(1 for row in candidates if row["eligible"]),
        "candidates": candidates,
    }
    return calibration, baseline_rows, default_rows


def _frame_summaries(frames: dict[str, CachedFrame], split: Split) -> list[dict[str, Any]]:
    split_of_frame = {
        name: ("calibration" if name in split.calibration else "heldout")
        for name in frames
    }
    summaries = []
    for name in sorted(frames):
        frame = frames[name]
        summaries.append(
            {
                "name": name,
                "split": split_of_frame[name],
                "route": frame.route,
                "image_shape": list(frame.image_shape),
                "prompt_count": int(frame.prompts.shape[0]),
                "sam_iou": frame.iou,
                "sam_score": frame.sam_score,
                "sam_reason": frame.sam_reason,
                "seg_latency_ms": frame.seg_ms,
                "sam_latency_ms": frame.sam_ms,
            }
        )
    return summaries


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def _load_route_map(path: Path, names: Sequence[str]) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read route map {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("route map must be a JSON object of frame stem -> route")
    route_map: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key:
            raise ValueError("route map keys must be non-empty frame stems")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"route map value for {key!r} must be a non-empty string")
        route_map[key] = value.strip()
    missing = [name for name in names if name not in route_map]
    if missing:
        preview = ", ".join(repr(name) for name in missing[:5])
        raise ValueError(f"route map is missing {len(missing)} frame(s): {preview}")
    return route_map


def _fake_components(
    name: str,
    index: int,
    image_shape: tuple[int, int],
    wall_as_floor_fraction: float,
) -> tuple[Any, Any, np.ndarray, str]:
    from perception.fakes import FakePromptMaskBackend, FakeSegBackend

    h, w = int(image_shape[0]), int(image_shape[1])
    seed = int(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    K = Intrinsics(
        fx=0.9 * w,
        fy=0.9 * w,
        cx=w / 2.0,
        cy=h / 2.0,
        width=w,
        height=h,
    )
    params = {
        "K": K,
        "H": h,
        "W": w,
        "h": 1.3 + 0.1 * float(seed % 3),
        "pitch_deg": 12.0 + 2.0 * float(seed % 4),
        "wall_distance": 3.5 + 0.5 * float(seed % 4),
        "boxes": (
            ((-0.3, 0.3, 2.6, 3.0, 0.2, 0.8),) if seed % 3 else ()
        ),
        "seed": seed,
    }
    seg_backend = FakeSegBackend(
        **params, wall_as_floor_fraction=float(wall_as_floor_fraction)
    )
    mode = (
        "floor",
        "floor",
        "floor",
        "floor",
        "floor_plus_wall",
        "empty",
        "raise",
    )[int(index) % 7]
    sam_backend = FakePromptMaskBackend(**params, mode=mode)
    gt_labels = np.full((h, w), MASK_OTHER, dtype=np.uint8)
    gt_labels[seg_backend.floor_mask] = MASK_FLOOR
    gt_labels[seg_backend.wall_mask] = MASK_WALL
    for box_mask in seg_backend.box_masks:
        gt_labels[box_mask] = MASK_OBSTACLE
    return seg_backend, sam_backend, gt_labels, mode


def _print_frame(frame: CachedFrame, index: int, total: int) -> None:
    if frame.sam_mask is None:
        state = frame.sam_reason or "unavailable"
        iou = "-"
    else:
        state = "ok"
        iou = f"{frame.iou:.3f}" if frame.iou is not None else "-"
    print(
        f"[{index}/{total}] {frame.name}: prompts={frame.prompts.shape[0]} "
        f"sam={state} iou={iou}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Safety-first calibration/held-out threshold sweep for SegFormer + "
            "promptable floor refinement"
        )
    )
    parser.add_argument("--images", required=True, help="folder of still frames")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    parser.add_argument(
        "--fake", action="store_true", help="use analytic fake backends"
    )
    parser.add_argument(
        "--fake-wall-as-floor",
        type=float,
        default=DEFAULT_FAKE_WALL_AS_FLOOR,
        help="fake mode only: relabel this fraction of wall pixels as floor",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="real mode: folder of <frame>.npz/.npy/.png ground-truth masks",
    )
    parser.add_argument(
        "--route-map",
        default=None,
        help="JSON object mapping frame stem -> route id for grouped splits",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help="fraction of frames (or whole routes) used for calibration",
    )
    parser.add_argument("--seed", type=int, default=0, help="split seed")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help="minimum number of labelled frames required",
    )
    parser.add_argument(
        "--iou-grid",
        default=",".join(f"{value:g}" for value in DEFAULT_IOU_GRID),
        help="comma-separated min_accept_iou grid",
    )
    parser.add_argument(
        "--veto-grid",
        default=",".join(f"{value:g}" for value in DEFAULT_VETO_GRID),
        help="comma-separated veto_prob grid",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(REPO_ROOT / "weights"),
        help="real mode: local weights folder",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="bootstrap resamples for held-out confidence intervals",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="bootstrap seed",
    )
    args = parser.parse_args(argv)

    try:
        iou_grid = _parse_grid(args.iou_grid, "--iou-grid")
        veto_grid = _parse_grid(args.veto_grid, "--veto-grid")
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    if not (0.0 < float(args.calibration_fraction) < 1.0):
        print("ERROR: --calibration-fraction must be strictly between 0 and 1")
        return 2
    if int(args.min_frames) < 2:
        print("ERROR: --min-frames must be at least 2")
        return 2
    if int(args.bootstrap_resamples) < 1:
        print("ERROR: --bootstrap-resamples must be at least 1")
        return 2
    if not (0.0 <= float(args.fake_wall_as_floor) <= 1.0):
        print("ERROR: --fake-wall-as-floor must be in [0, 1]")
        return 2
    if not args.fake and float(args.fake_wall_as_floor) != DEFAULT_FAKE_WALL_AS_FLOOR:
        print("ERROR: --fake-wall-as-floor only applies to --fake")
        return 2
    if args.fake and args.labels_dir:
        print("ERROR: --labels-dir is for real mode; --fake renders ground truth")
        return 2
    if not args.fake and not args.labels_dir:
        print("ERROR: real mode requires --labels-dir with a mask per frame")
        return 2

    image_dir = Path(args.images)
    if not image_dir.is_dir():
        print(f"ERROR: image folder not found: {image_dir}")
        return 2
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        print(
            f"ERROR: no frames found in {image_dir} "
            f"({', '.join(sorted(IMAGE_SUFFIXES))})"
        )
        return 2
    names = sorted(path.stem for path in paths)
    if len(set(names)) != len(names):
        print("ERROR: frame stems must be unique (extensions collide)")
        return 2
    if len(names) < int(args.min_frames):
        print(
            f"ERROR: need at least {int(args.min_frames)} labelled frames; "
            f"found {len(names)}"
        )
        return 2
    by_name = {path.stem: path for path in paths}

    route_map: dict[str, str] | None = None
    if args.route_map:
        try:
            route_map = _load_route_map(Path(args.route_map), names)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2

    labels: dict[str, np.ndarray] = {}
    if not args.fake:
        labels_dir = Path(args.labels_dir)
        if not labels_dir.is_dir():
            print(f"ERROR: labels folder not found: {labels_dir}")
            return 2
        for name in names:
            label_path = _find_sidecar(labels_dir, name, LABEL_SUFFIXES)
            if label_path is None:
                print(
                    f"ERROR: no ground-truth mask for frame {name!r} in "
                    f"{labels_dir} ({', '.join(LABEL_SUFFIXES)})"
                )
                return 2
            try:
                labels[name] = _load_labels(label_path)
            except Exception as exc:  # noqa: BLE001 - user-supplied label file
                print(
                    f"ERROR: cannot load {label_path.name}: "
                    f"{_describe_exception(exc)}"
                )
                return 2

    try:
        split = split_frames(
            names,
            calibration_fraction=float(args.calibration_fraction),
            seed=int(args.seed),
            route_map=route_map,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not args.fake:
        weights_dir = Path(args.weights_dir)
        missing = _check_weights(weights_dir, need_depth=False)
        if missing:
            print("ERROR: real model weights are missing:")
            for name, expected in missing:
                print(f"  - {name} (expected at {expected})")
            print("\nFetch them with:")
            print("  python scripts/fetch_models.py --dry-run")
            print("  python scripts/fetch_models.py --yes")
            print("  MobileSAM is a manual download; see docs/perception_backends.md")
            return 2
        from perception.seg_backend import SegFormerBackend
        from perception.seg_fusion import MobileSAMBackend

        seg_backend = SegFormerBackend(weights_dir=weights_dir / "segformer-b0-ade")
        sam_backend = MobileSAMBackend(
            checkpoint=weights_dir / "mobile-sam" / "mobile_sam.pt"
        )
        sam_errors = sam_backend.preflight()
        if sam_errors:
            print("ERROR: MobileSAM is not ready:")
            for error in sam_errors:
                print(f"  - {error}")
            return 2

    cache_veto_probs = sorted(set(veto_grid) | {float(DEFAULT_VETO_PROB)})
    min_accept_floor = min(set(iou_grid) | {float(DEFAULT_MIN_ACCEPT_IOU)})
    frames: dict[str, CachedFrame] = {}

    for index, name in enumerate(names, start=1):
        path = by_name[name]
        try:
            image = load_image(path)
        except Exception as exc:  # noqa: BLE001 - report the file
            print(f"ERROR: cannot load {path.name}: {_describe_exception(exc)}")
            return 2

        if args.fake:
            seg_backend, sam_backend, gt_labels, _mode = _fake_components(
                name, index - 1, image.shape[:2], float(args.fake_wall_as_floor)
            )
        else:
            gt_labels = labels[name]
            if gt_labels.shape != image.shape[:2]:
                print(
                    f"ERROR: {name}: mask shape {gt_labels.shape} "
                    f"!= image {image.shape[:2]}"
                )
                return 2

        try:
            frame = cache_frame(
                name,
                route_map.get(name) if route_map is not None else None,
                image,
                gt_labels,
                seg_backend,
                sam_backend,
                veto_probs=cache_veto_probs,
                min_accept_floor=min_accept_floor,
            )
        except Exception as exc:  # noqa: BLE001 - stage-specific message
            print(f"ERROR: frame {name}: {_describe_exception(exc)}")
            return 2
        frames[name] = frame
        _print_frame(frame, index, len(names))

    calibration, baseline_rows, default_rows = _build_calibration_rows(
        frames, split, iou_grid, veto_grid
    )
    selection = select_candidate(calibration["candidates"], calibration["segformer"])
    heldout, heldout_rows = _build_heldout(
        frames,
        split,
        selection,
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_resamples=int(args.bootstrap_resamples),
    )

    for row in baseline_rows:
        row["split"] = "calibration"
        row["route"] = frames[row["frame"]].route
    csv_rows = list(baseline_rows)
    for row in default_rows:
        row["split"] = "calibration"
        row["route"] = frames[row["frame"]].route
    csv_rows.extend(default_rows)
    if selection["variant"] == "fusion":
        selected_rows = _decision_rows(
            frames,
            split.calibration,
            variant="fusion_selected",
            min_accept_iou=selection["min_accept_iou"],
            veto_prob=selection["veto_prob"],
        )
    else:
        selected_rows = _decision_rows(
            frames,
            split.calibration,
            variant="fusion_selected",
            min_accept_iou=None,
            veto_prob=None,
        )
    for row in selected_rows:
        row["split"] = "calibration"
        row["route"] = frames[row["frame"]].route
    csv_rows.extend(selected_rows)
    csv_rows.extend(heldout_rows)

    report = {
        "schema_version": 1,
        "generated": datetime.datetime.now().isoformat(),
        "mode": "fake" if args.fake else "real",
        "ground_truth": "synthetic" if args.fake else "provided",
        "weights_dir": None if args.fake else str(Path(args.weights_dir)),
        "n_frames": len(names),
        "split": {
            "strategy": split.strategy,
            "seed": int(args.seed),
            "calibration_fraction": float(args.calibration_fraction),
            "n_calibration": len(split.calibration),
            "n_heldout": len(split.heldout),
            "calibration_frames": split.calibration,
            "heldout_frames": split.heldout,
            "calibration_routes": split.calibration_routes,
            "heldout_routes": split.heldout_routes,
        },
        "grid": {
            "min_accept_iou": list(iou_grid),
            "veto_prob": list(veto_grid),
            "default_min_accept_iou": DEFAULT_MIN_ACCEPT_IOU,
            "default_veto_prob": DEFAULT_VETO_PROB,
        },
        "policy": {
            "selection_basis": (
                "calibration frames only; held-out is reported but never used "
                "to choose thresholds"
            ),
            "eligibility": (
                "floor IoU >= SegFormer-only and unsafe_free_rate and "
                "wall_as_floor_rate <= SegFormer-only on calibration"
            ),
            "order": (
                "min unsafe_free_rate, min wall_as_floor_rate, max floor_iou, "
                "max floor_recall, higher min_accept_iou, higher veto_prob, "
                "max accepted_rate"
            ),
            "fallback": (
                "if no grid configuration is eligible, or the winner has no "
                "measured improvement over SegFormer-only, select SegFormer-only"
            ),
            "split": (
                "seeded deterministic split; with a route map every route stays "
                "entirely on one side"
            ),
        },
        "calibration": calibration,
        "selection": selection,
        "heldout": heldout,
        "frames": _frame_summaries(frames, split),
        "notes": [
            "Synthetic runs exercise the tuning workflow; they are not real-world accuracy.",
            "Every numeric default here is TO TUNE ON REAL DATA; none is validated.",
            "Thresholds selected on these calibration frames do not guarantee held-out safety.",
            "A held-out split from the same routes or session is not an independent test set; use a separate route/session for final claims.",
            "Held-out CIs bootstrap frames, treat correlated frames as independent evidence, and are descriptive only.",
            "Fusion latency here excludes the label-application step and is machine-specific.",
        ],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fusion_tuning.json"
    csv_path = out_dir / "fusion_tuning.csv"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(csv_path, csv_rows)

    baseline = calibration["segformer"]
    print("\n" + "=" * 68)
    print(
        f"Split: {len(split.calibration)} calibration / "
        f"{len(split.heldout)} held-out ({split.strategy}, seed {int(args.seed)})"
    )
    print(
        f"SegFormer calibration: floor_iou={_csv_value(baseline['floor_iou']) or '-'} "
        f"unsafe_free={_csv_value(baseline['unsafe_free_rate']) or '-'} "
        f"wall_as_floor={_csv_value(baseline['wall_as_floor_rate']) or '-'}"
    )
    if selection["variant"] == "fusion":
        print(
            f"Selected: min_accept_iou={selection['min_accept_iou']:.2f} "
            f"veto_prob={selection['veto_prob']:.2f} "
            f"({selection['reason']})"
        )
    else:
        print(f"Selected: SegFormer-only ({selection['reason']})")

    print("\nHeld-out:")
    for variant in VARIANT_FIELDS:
        entry = heldout["variants"][variant]
        unsafe_ci = entry["ci95"]["unsafe_free_rate"]["ci95"]
        floor_ci = entry["ci95"]["floor_iou"]["ci95"]
        print(
            f"  {variant:<16} floor_iou={_csv_value(entry['floor_iou']) or '-'} "
            f"[{_csv_value(floor_ci[0]) or '-'}, {_csv_value(floor_ci[1]) or '-'}] "
            f"unsafe_free={_csv_value(entry['unsafe_free_rate']) or '-'} "
            f"[{_csv_value(unsafe_ci[0]) or '-'}, {_csv_value(unsafe_ci[1]) or '-'}] "
            f"accepted={_csv_value(entry['accepted_rate']) or '-'}"
        )
    print(f"\nWrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
