"""Abstention rules for certified floor navigation decisions.

A frame is retained only when every independent safety signal is acceptable:

* enough certified floor in the forward corridor;
* a sufficiently narrow relative interval for the nearest obstacle;
* a plausible, high-inlier ground-plane fit;
* an acceptable metric-scale status;
* a low fraction of floor-labelled points rejected by the geometric veto.

The rule is intentionally conservative and modular: each failed check emits a
human-readable reason, so an operator can see *why* the system abstained.  The
defaults are starting points only.  TO TUNE ON REAL DATA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.metrics_abstain import compute_abstention_metrics

# Corridor geometry: 1.0 m wide, 0.5-3.0 m forward.
# TO TUNE ON REAL DATA.
DEFAULT_CORRIDOR_WIDTH_M = 1.0
DEFAULT_CORRIDOR_Y_MIN_M = 0.5
DEFAULT_CORRIDOR_Y_MAX_M = 3.0
DEFAULT_MIN_CERTIFIED_FLOOR_FRACTION = 0.30
DEFAULT_MAX_RELATIVE_INTERVAL_WIDTH = 0.50
DEFAULT_MIN_PLANE_INLIER_RATIO = 0.50
DEFAULT_MAX_VETO_FRACTION = 0.50
DEFAULT_ALLOWED_PLANE_STATUSES = ("OK",)
DEFAULT_ALLOWED_SCALE_STATUSES = ("OK",)


def _status_text(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value)


@dataclass(frozen=True)
class AbstainConfig:
    """Thresholds for :class:`AbstainRule`.

    All numeric thresholds are initial values to be tuned on real routes.
    """

    corridor_width_m: float = DEFAULT_CORRIDOR_WIDTH_M
    corridor_y_min_m: float = DEFAULT_CORRIDOR_Y_MIN_M
    corridor_y_max_m: float = DEFAULT_CORRIDOR_Y_MAX_M
    min_certified_floor_fraction: float = DEFAULT_MIN_CERTIFIED_FLOOR_FRACTION
    max_relative_interval_width: float = DEFAULT_MAX_RELATIVE_INTERVAL_WIDTH
    min_plane_inlier_ratio: float = DEFAULT_MIN_PLANE_INLIER_RATIO
    max_veto_fraction: float = DEFAULT_MAX_VETO_FRACTION
    allowed_plane_statuses: tuple[str, ...] = DEFAULT_ALLOWED_PLANE_STATUSES
    allowed_scale_statuses: tuple[str, ...] = DEFAULT_ALLOWED_SCALE_STATUSES

    def __post_init__(self) -> None:
        if self.corridor_width_m <= 0:
            raise ValueError("corridor_width_m must be > 0")
        if not (0.0 <= self.corridor_y_min_m < self.corridor_y_max_m):
            raise ValueError("corridor forward range must satisfy 0 <= min < max")
        for name in (
            "min_certified_floor_fraction",
            "max_relative_interval_width",
            "min_plane_inlier_ratio",
            "max_veto_fraction",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True)
class AbstainFeatures:
    """Inference-time features used by the abstention rule."""

    certified_floor_fraction: float
    relative_interval_width: float | None
    plane_inlier_ratio: float
    plane_status: str
    scale_status: str
    veto_fraction: float
    corridor_points: int = 0
    floor_labelled_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AbstainDecision:
    """Decision plus the evidence that produced it."""

    abstain: bool
    reasons: list[str]
    features: AbstainFeatures
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstain": self.abstain,
            "reasons": list(self.reasons),
            "features": self.features.to_dict(),
            "checks": dict(self.checks),
        }


def _decision_from_features(
    features: AbstainFeatures,
    cfg: AbstainConfig,
) -> AbstainDecision:
    reasons: list[str] = []
    notes: list[str] = []
    checks: dict[str, bool] = {}

    floor_ok = features.certified_floor_fraction >= cfg.min_certified_floor_fraction
    checks["certified_floor_fraction"] = floor_ok
    if not floor_ok:
        reasons.append(
            "certified floor fraction in corridor "
            f"{features.certified_floor_fraction:.3f} < "
            f"{cfg.min_certified_floor_fraction:.3f}"
        )

    if features.relative_interval_width is None:
        checks["relative_interval_width"] = True
        notes.append("nearest-obstacle interval unavailable; interval check skipped")
    elif not math.isfinite(features.relative_interval_width):
        checks["relative_interval_width"] = False
        reasons.append("nearest-obstacle relative interval width is not finite")
    else:
        width_ok = (
            features.relative_interval_width <= cfg.max_relative_interval_width
        )
        checks["relative_interval_width"] = width_ok
        if not width_ok:
            reasons.append(
                "nearest-obstacle relative interval width "
                f"{features.relative_interval_width:.3f} > "
                f"{cfg.max_relative_interval_width:.3f}"
            )

    inlier_ok = (
        math.isfinite(features.plane_inlier_ratio)
        and features.plane_inlier_ratio >= cfg.min_plane_inlier_ratio
    )
    checks["plane_inlier_ratio"] = inlier_ok
    if not inlier_ok:
        reasons.append(
            "plane inlier ratio "
            f"{features.plane_inlier_ratio:.3f} < "
            f"{cfg.min_plane_inlier_ratio:.3f}"
        )

    plane_ok = features.plane_status in cfg.allowed_plane_statuses
    checks["plane_status"] = plane_ok
    if not plane_ok:
        reasons.append(f"plane status {features.plane_status!r} is not certified")

    scale_ok = features.scale_status in cfg.allowed_scale_statuses
    checks["scale_status"] = scale_ok
    if not scale_ok:
        reasons.append(f"scale status {features.scale_status!r} is not certified")

    veto_ok = (
        math.isfinite(features.veto_fraction)
        and features.veto_fraction <= cfg.max_veto_fraction
    )
    checks["veto_fraction"] = veto_ok
    if not veto_ok:
        reasons.append(
            "floor-labelled veto fraction "
            f"{features.veto_fraction:.3f} > "
            f"{cfg.max_veto_fraction:.3f}"
        )

    abstain = not all(checks.values())
    if not reasons:
        reasons.append("all abstention checks passed")
    reasons.extend(notes)
    return AbstainDecision(
        abstain=abstain,
        reasons=reasons,
        features=features,
        checks=checks,
    )


class AbstainRule:
    """Apply the corridor, interval, plane, scale and veto checks."""

    def __init__(self, config: AbstainConfig | None = None) -> None:
        self.config = config if config is not None else AbstainConfig()

    def evaluate(
        self,
        points_g: np.ndarray,
        certified_floor_mask: np.ndarray,
        *,
        plane_inlier_ratio: float,
        plane_status: Any,
        scale_status: Any,
        floor_label_mask: np.ndarray | None = None,
        veto_mask: np.ndarray | None = None,
        nearest_obstacle_pred_m: float | None = None,
        nearest_obstacle_lower_m: float | None = None,
        nearest_obstacle_upper_m: float | None = None,
    ) -> AbstainDecision:
        """Evaluate one frame.

        Args:
            points_g: ``(N, 3)`` ground-frame points for the frame.
            certified_floor_mask: Boolean mask for the same points.
            plane_inlier_ratio: Ground-plane RANSAC inlier ratio.
            plane_status: ``FitStatus`` or status string.
            scale_status: ``ScaleResult.status`` string.
            floor_label_mask: Boolean mask of semantically floor-labelled
                points (before the veto).  Required to compute the veto
                fraction; without it the veto fraction is zero.
            veto_mask: Boolean geometric-veto output for the same points.
            nearest_obstacle_pred_m: Predicted nearest obstacle distance.
            nearest_obstacle_lower_m: Conformal lower bound for that obstacle.
            nearest_obstacle_upper_m: Conformal upper bound.  Ignored for the
                one-sided score if it is infinite.

        Returns:
            :class:`AbstainDecision` with reasons.
        """
        pts = np.asarray(points_g, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"points_g must have shape (N, 3), got {pts.shape}")
        certified = np.asarray(certified_floor_mask, dtype=bool).reshape(-1)
        if certified.size != pts.shape[0]:
            raise ValueError("certified_floor_mask must match points_g length")

        half_width = 0.5 * float(self.config.corridor_width_m)
        in_corridor = (
            (np.abs(pts[:, 0]) <= half_width)
            & (pts[:, 1] >= self.config.corridor_y_min_m)
            & (pts[:, 1] <= self.config.corridor_y_max_m)
        )
        corridor_count = int(np.sum(in_corridor))
        floor_fraction = (
            float(np.mean(certified[in_corridor])) if corridor_count else 0.0
        )

        veto_fraction = 0.0
        floor_count = 0
        if floor_label_mask is not None:
            floor_labels = np.asarray(floor_label_mask, dtype=bool).reshape(-1)
            if floor_labels.size != pts.shape[0]:
                raise ValueError("floor_label_mask must match points_g length")
            floor_count = int(np.sum(floor_labels))
            if veto_mask is not None:
                veto = np.asarray(veto_mask, dtype=bool).reshape(-1)
                if veto.size != pts.shape[0]:
                    raise ValueError("veto_mask must match points_g length")
                if floor_count:
                    veto_fraction = float(np.mean(veto[floor_labels]))
            else:
                veto_fraction = 0.0

        rel_width: float | None = None
        if nearest_obstacle_pred_m is not None:
            pred = float(nearest_obstacle_pred_m)
            if (
                not math.isfinite(pred)
                or pred <= 0.0
                or nearest_obstacle_lower_m is None
                or nearest_obstacle_upper_m is None
            ):
                rel_width = math.nan
            else:
                lo = float(nearest_obstacle_lower_m)
                hi = float(nearest_obstacle_upper_m)
                if math.isfinite(hi):
                    width = (hi - lo) / pred
                    rel_width = width if width >= 0.0 else math.nan
                else:
                    # One-sided interval: upper is +inf, so use the certified
                    # downward margin as the available relative width.  This
                    # keeps the rule usable with the overestimation score.
                    rel_width = max(0.0, (pred - lo) / pred)

        features = AbstainFeatures(
            certified_floor_fraction=floor_fraction,
            relative_interval_width=rel_width,
            plane_inlier_ratio=float(plane_inlier_ratio),
            plane_status=_status_text(plane_status),
            scale_status=_status_text(scale_status),
            veto_fraction=veto_fraction,
            corridor_points=corridor_count,
            floor_labelled_points=floor_count,
        )
        return _decision_from_features(features, self.config)


def _coerce_features(frame: AbstainFeatures | Mapping[str, Any]) -> tuple[AbstainFeatures, bool]:
    if isinstance(frame, AbstainFeatures):
        raise TypeError("sweep_thresholds frames must include an 'unsafe' flag")
    if not isinstance(frame, Mapping):
        raise TypeError(f"unsupported frame type {type(frame)!r}")
    if "unsafe" not in frame:
        raise ValueError("each sweep frame must contain an 'unsafe' boolean")
    data = dict(frame)
    unsafe = bool(data.pop("unsafe"))
    features = AbstainFeatures(
        certified_floor_fraction=float(data["certified_floor_fraction"]),
        relative_interval_width=(
            None
            if data.get("relative_interval_width") is None
            else float(data["relative_interval_width"])
        ),
        plane_inlier_ratio=float(data.get("plane_inlier_ratio", 1.0)),
        plane_status=str(data.get("plane_status", "OK")),
        scale_status=str(data.get("scale_status", "OK")),
        veto_fraction=float(data.get("veto_fraction", 0.0)),
        corridor_points=int(data.get("corridor_points", 1)),
        floor_labelled_points=int(data.get("floor_labelled_points", 1)),
    )
    return features, unsafe


def frame_is_unsafe(
    certified_floor_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    corridor_mask: np.ndarray,
    *,
    true_nearest_obstacle_m: float | None = None,
    conformal_lower_bound_m: float | None = None,
) -> tuple[bool, list[str]]:
    """Apply the two-part unsafe-frame definition.

    A frame is unsafe when either:

    1. a certified-floor cell inside the forward corridor is actually
       non-floor in the ground truth; or
    2. the true nearest-obstacle distance is below the conformal lower bound.
    """
    certified = np.asarray(certified_floor_mask, dtype=bool)
    gt = np.asarray(ground_truth_mask)
    corridor = np.asarray(corridor_mask, dtype=bool)
    if certified.shape != gt.shape or certified.shape != corridor.shape:
        raise ValueError("certified, ground-truth and corridor masks must match")

    reasons: list[str] = []
    valid_gt = (gt != 0) & (gt != 255)
    nonfloor = valid_gt & (gt != 1)
    false_certified = int(np.sum(certified & nonfloor & corridor))
    if false_certified > 0:
        reasons.append(
            f"{false_certified} certified-floor corridor cell(s) are non-floor in GT"
        )

    if (
        true_nearest_obstacle_m is not None
        and conformal_lower_bound_m is not None
        and math.isfinite(float(true_nearest_obstacle_m))
    ):
        true_d = float(true_nearest_obstacle_m)
        lower = float(conformal_lower_bound_m)
        if not math.isfinite(lower) or true_d < lower:
            reasons.append(
                f"true nearest obstacle {true_d:.3f} m is below conformal lower "
                f"bound {lower:.3f} m"
            )

    return bool(reasons), reasons


def sweep_thresholds(
    frames: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float] | None = None,
    config: AbstainConfig | None = None,
) -> dict[str, Any]:
    """Sweep the certified-floor corridor threshold.

    Each frame mapping must contain ``unsafe`` plus the keys of
    :class:`AbstainFeatures` (missing non-critical keys use permissive
    defaults).  For each threshold the rule is re-evaluated with that
    ``min_certified_floor_fraction``; the function reports coverage, risk, and
    the existing AURC/risk-coverage metrics from
    :func:`evaluation.metrics_abstain.compute_abstention_metrics`.

    Returns:
        Dict with ``rows``, ``aurc``, ``random_aurc_mean``, and
        ``risk_coverage_monotonic``.  ``risk_coverage_monotonic`` is True when
        risk does not decrease as coverage increases.
    """
    base = config if config is not None else AbstainConfig()
    if thresholds is None:
        thresholds_arr = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    else:
        thresholds_arr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if thresholds_arr.size == 0:
        raise ValueError("thresholds must not be empty")

    data = [_coerce_features(frame) for frame in frames]
    if not data:
        raise ValueError("frames must not be empty")

    rows: list[dict[str, Any]] = []
    all_unsafe = np.array([unsafe for _, unsafe in data], dtype=np.float64)
    confidence = np.array(
        [features.certified_floor_fraction for features, _ in data], dtype=np.float64
    )
    for threshold in thresholds_arr:
        cfg = replace(base, min_certified_floor_fraction=float(threshold))
        decisions = [_decision_from_features(features, cfg) for features, _ in data]
        kept = np.array([not decision.abstain for decision in decisions], dtype=bool)
        coverage = float(np.mean(kept))
        risk = float(np.mean(all_unsafe[kept])) if np.any(kept) else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "risk": risk,
                "abstention_rate": float(1.0 - coverage),
                "kept": int(np.sum(kept)),
            }
        )

    metrics = compute_abstention_metrics(
        errors=all_unsafe,
        confidence=confidence,
        thresholds=thresholds_arr,
    )
    ordered = sorted(rows, key=lambda row: row["coverage"])
    risks = [row["risk"] for row in ordered if math.isfinite(row["risk"])]
    monotonic = True
    for previous, current in zip(risks, risks[1:]):
        if current + 1e-12 < previous:
            monotonic = False
            break

    return {
        "rows": rows,
        "aurc": metrics["aurc"],
        "random_aurc_mean": metrics["random_aurc_mean"],
        "risk_coverage_monotonic": monotonic,
        "aurc_definition": "area under risk-coverage curve; lower is better",
    }


__all__ = [
    "DEFAULT_ALLOWED_PLANE_STATUSES",
    "DEFAULT_ALLOWED_SCALE_STATUSES",
    "DEFAULT_CORRIDOR_WIDTH_M",
    "DEFAULT_CORRIDOR_Y_MIN_M",
    "DEFAULT_CORRIDOR_Y_MAX_M",
    "DEFAULT_MAX_RELATIVE_INTERVAL_WIDTH",
    "DEFAULT_MAX_VETO_FRACTION",
    "DEFAULT_MIN_CERTIFIED_FLOOR_FRACTION",
    "DEFAULT_MIN_PLANE_INLIER_RATIO",
    "AbstainConfig",
    "AbstainDecision",
    "AbstainFeatures",
    "AbstainRule",
    "frame_is_unsafe",
    "sweep_thresholds",
]
