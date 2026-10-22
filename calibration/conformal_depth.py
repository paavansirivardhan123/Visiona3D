"""Split-conformal calibration for monocular metric depth.

The score is relative to the prediction so that the same calibration can be
used across depth ranges.  Two scores are supported:

``two_sided``
    ``s = |gt - pred| / pred``.  The resulting interval is
    ``[pred * (1 - q), pred * (1 + q)]`` clipped at zero.

``overestimation``
    ``s = (pred - gt) / pred`` for ``pred > gt`` and zero otherwise.  This is
    a one-sided score: it certifies a lower bound ``pred * (1 - q)`` but makes
    no claim about the upper side.  It is the score intended for obstacle
    distance, where the dangerous failure is overestimating distance.

Statistical guarantee
---------------------
Under exchangeability of the calibration and test points, split conformal
prediction gives marginal coverage at least ``1 - alpha`` for the two-sided
score.  The finite-sample correction is the ``ceil((n + 1) * (1 - alpha))``
order statistic.  Exchangeability is strong: it can fail under distribution
shift, route-level correlation, repeated annotations of the same frame, or
group-dependent noise.  Mondrian (group) calibration improves conditional
coverage only for groups that have enough calibration points.  Small groups
fall back to the pooled quantile, which restores marginal coverage but not
the group guarantee.

Nothing in this module uses ground-truth depth at prediction time.  The
``groups`` argument must be derived from inference-time information, for
example a bin of the *predicted* depth.  ``depth_bin_groups`` is provided for
that purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCORE_TWO_SIDED = "two_sided"
SCORE_OVERESTIMATION = "overestimation"
_VALID_SCORES = (SCORE_TWO_SIDED, SCORE_OVERESTIMATION)

# The minimum group size is a statistical and operational choice, not a
# validated real-world threshold.  TO TUNE ON REAL DATA.
DEFAULT_ALPHA = 0.10
DEFAULT_MIN_GROUP_SIZE = 30

# Range edges match core.config.RANGE_BIN_EDGES and are used only by the
# convenience helper for prediction-time Mondrian groups.
DEPTH_BIN_EDGES = (0.0, 2.0, 4.0, 6.0, float("inf"))


def _as_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def conformal_quantile(scores: Sequence[float] | np.ndarray, alpha: float) -> float:
    """Return the split-conformal quantile of non-NaN scores.

    The one-based order is ``k = ceil((n + 1) * (1 - alpha))``.  If ``k > n``
    the calibration set is too small to support the requested level and the
    function returns positive infinity.  NaNs are ignored; infinities are
    retained because they express a conservative score.

    Args:
        scores: Calibration nonconformity scores.
        alpha: Miscoverage level in (0, 1).

    Returns:
        The k-th smallest finite/infinite score, or ``+inf``.
    """
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    arr = _as_1d(scores, "scores")
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return math.inf

    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return math.inf
    # One-based k-th smallest == zero-based k-1.
    return float(np.sort(arr)[k - 1])


def _normalise_group_key(group: Any) -> str:
    """Return a stable JSON-friendly key for a group label."""
    if isinstance(group, np.generic):
        group = group.item()
    return str(group)


def depth_bin_groups(
    predictions: Sequence[float] | np.ndarray,
    edges: Sequence[float] = DEPTH_BIN_EDGES,
) -> np.ndarray:
    """Assign predicted depths to left-inclusive range bins.

    This helper deliberately takes predictions, not ground truth.  Using the
    true depth to choose a group would leak information that is unavailable at
    inference time and invalidates the conformal guarantee.
    """
    pred = _as_1d(predictions, "predictions")
    boundaries = np.asarray(edges, dtype=np.float64)
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError("edges must be a one-dimensional sequence with >= 2 entries")
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("edges must be strictly increasing")
    # np.digitize is right-exclusive for the interior bins; the final +inf
    # boundary puts values >= the last finite edge in the last bin.
    internal = boundaries[1:-1]
    groups = np.digitize(pred, internal, right=False)
    return groups.astype(np.int64)


@dataclass(frozen=True)
class ConformalPrediction:
    """Prediction intervals and the information needed to audit their use.

    Attributes:
        lower: Lower interval bound (metres), NaN for invalid predictions.
        upper: Upper interval bound (metres), NaN for invalid predictions.
        quantile: Relative score quantile used for each point.
        used_pooled: True when the point used the pooled quantile (no group,
            unknown group, or a group below ``min_group_size``).
        valid: False for NaN/non-positive predictions.
    """

    lower: np.ndarray
    upper: np.ndarray
    quantile: np.ndarray
    used_pooled: np.ndarray
    valid: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "quantile": self.quantile,
            "used_pooled": self.used_pooled,
            "valid": self.valid,
        }


class ConformalDepth:
    """Split-conformal relative-depth calibration.

    Args:
        alpha: Target miscoverage level.  ``0.10`` means a 90% interval.
            TO TUNE ON REAL DATA.
        score: ``"two_sided"`` or ``"overestimation"``.
        min_group_size: Minimum calibration points for a Mondrian group.
            Smaller groups fall back to the pooled quantile.
            TO TUNE ON REAL DATA.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        score: str = SCORE_TWO_SIDED,
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    ) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if score not in _VALID_SCORES:
            raise ValueError(f"score must be one of {_VALID_SCORES}, got {score!r}")
        if int(min_group_size) < 1:
            raise ValueError("min_group_size must be >= 1")

        self.alpha = float(alpha)
        self.score = str(score)
        self.min_group_size = int(min_group_size)

        self.pooled_quantile_ = math.inf
        self.pooled_count_ = 0
        self.group_quantiles_: dict[str, float] = {}
        self.group_counts_: dict[str, int] = {}
        self.fallback_groups_: dict[str, dict[str, Any]] = {}
        self.n_valid_ = 0
        self.n_invalid_ = 0
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _scores(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        if self.score == SCORE_TWO_SIDED:
            return np.abs(gt - pred) / pred
        # One-sided overestimation score: only positive overestimates count.
        return np.maximum((pred - gt) / pred, 0.0)

    def fit(
        self,
        predictions: Sequence[float] | np.ndarray,
        ground_truth: Sequence[float] | np.ndarray,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> "ConformalDepth":
        """Fit calibrated quantiles.

        Invalid rows (NaN/non-positive prediction, NaN/non-positive GT) are
        ignored and counted, never raised.  ``groups`` must be derived from
        inference-time metadata.
        """
        pred = _as_1d(predictions, "predictions")
        gt = _as_1d(ground_truth, "ground_truth")
        if pred.size != gt.size:
            raise ValueError(
                f"predictions and ground_truth must have equal length; got {pred.size} and {gt.size}"
            )

        groups_arr: np.ndarray | None
        if groups is None:
            groups_arr = None
        else:
            groups_arr = np.asarray(groups, dtype=object).reshape(-1)
            if groups_arr.size != pred.size:
                raise ValueError(
                    f"groups must have the same length as predictions; got {groups_arr.size} and {pred.size}"
                )

        valid = (
            np.isfinite(pred)
            & (pred > 0.0)
            & np.isfinite(gt)
            & (gt > 0.0)
        )
        self.n_valid_ = int(np.sum(valid))
        self.n_invalid_ = int(pred.size - self.n_valid_)

        scores = self._scores(pred[valid], gt[valid])
        self.pooled_quantile_ = conformal_quantile(scores, self.alpha)
        self.pooled_count_ = int(scores.size)

        self.group_quantiles_ = {}
        self.group_counts_ = {}
        self.fallback_groups_ = {}

        if groups_arr is not None:
            valid_groups = groups_arr[valid]
            for raw_group in sorted({_normalise_group_key(g) for g in valid_groups}):
                mask = np.array(
                    [_normalise_group_key(g) == raw_group for g in valid_groups],
                    dtype=bool,
                )
                count = int(np.sum(mask))
                self.group_counts_[raw_group] = count
                if count >= self.min_group_size:
                    self.group_quantiles_[raw_group] = conformal_quantile(
                        scores[mask], self.alpha
                    )
                else:
                    self.fallback_groups_[raw_group] = {
                        "n": count,
                        "reason": f"n < min_group_size ({self.min_group_size})",
                        "used_pooled": True,
                    }

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _quantiles_for(
        self,
        pred: np.ndarray,
        groups: Sequence[Any] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("ConformalDepth.fit() must be called before prediction")

        n = pred.size
        quantiles = np.full(n, self.pooled_quantile_, dtype=np.float64)
        used_pooled = np.ones(n, dtype=bool)

        if groups is None:
            return quantiles, used_pooled

        groups_arr = np.asarray(groups, dtype=object).reshape(-1)
        if groups_arr.size != n:
            raise ValueError(
                f"groups must have the same length as predictions; got {groups_arr.size} and {n}"
            )
        for index, group in enumerate(groups_arr):
            key = _normalise_group_key(group)
            if key in self.group_quantiles_:
                quantiles[index] = self.group_quantiles_[key]
                used_pooled[index] = False
            # Unknown groups and recorded small groups keep the pooled value.
        return quantiles, used_pooled

    def predict(
        self,
        predictions: Sequence[float] | np.ndarray,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> ConformalPrediction:
        """Construct intervals for predictions.

        NaN and non-positive predictions return NaN bounds with ``valid=False``.
        A group unknown at fit time silently uses the pooled quantile and is
        reported through ``used_pooled``.
        """
        pred = _as_1d(predictions, "predictions")
        quantiles, used_pooled = self._quantiles_for(pred, groups)

        valid = np.isfinite(pred) & (pred > 0.0)
        safe_pred = np.where(valid, pred, np.nan)

        lower = safe_pred * (1.0 - quantiles)
        lower = np.maximum(lower, 0.0)
        if self.score == SCORE_TWO_SIDED:
            upper = safe_pred * (1.0 + quantiles)
        else:
            upper = np.full_like(safe_pred, np.inf)
        upper = np.where(valid, upper, np.nan)
        lower = np.where(valid, lower, np.nan)
        quantiles = np.where(valid, quantiles, np.nan)

        return ConformalPrediction(
            lower=lower,
            upper=upper,
            quantile=quantiles,
            used_pooled=used_pooled,
            valid=valid,
        )

    def intervals(
        self,
        predictions: Sequence[float] | np.ndarray,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` arrays."""
        result = self.predict(predictions, groups)
        return result.lower, result.upper

    def lower_bounds(
        self,
        predictions: Sequence[float] | np.ndarray,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the one-sided-safe lower bound (also valid for two-sided)."""
        return self.predict(predictions, groups).lower

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        ``+inf`` quantiles are written as ``None``; ``from_dict`` restores
        them.  This keeps the document valid JSON.
        """
        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(value) else None

        return {
            "schema_version": 1,
            "alpha": self.alpha,
            "score": self.score,
            "min_group_size": self.min_group_size,
            "pooled_quantile": finite_or_none(self.pooled_quantile_),
            "pooled_count": self.pooled_count_,
            "group_quantiles": {
                key: finite_or_none(value)
                for key, value in sorted(self.group_quantiles_.items())
            },
            "group_counts": dict(sorted(self.group_counts_.items())),
            "fallback_groups": {
                key: dict(value) for key, value in sorted(self.fallback_groups_.items())
            },
            "n_valid": self.n_valid_,
            "n_invalid": self.n_invalid_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformalDepth":
        """Restore a model written by :meth:`to_dict`."""
        model = cls(
            alpha=float(payload["alpha"]),
            score=str(payload.get("score", SCORE_TWO_SIDED)),
            min_group_size=int(payload.get("min_group_size", DEFAULT_MIN_GROUP_SIZE)),
        )

        def inf_if_none(value: Any) -> float:
            return math.inf if value is None else float(value)

        model.pooled_quantile_ = inf_if_none(payload.get("pooled_quantile"))
        model.pooled_count_ = int(payload.get("pooled_count", 0))
        model.group_quantiles_ = {
            str(key): inf_if_none(value)
            for key, value in dict(payload.get("group_quantiles", {})).items()
        }
        model.group_counts_ = {
            str(key): int(value)
            for key, value in dict(payload.get("group_counts", {})).items()
        }
        model.fallback_groups_ = {
            str(key): dict(value)
            for key, value in dict(payload.get("fallback_groups", {})).items()
        }
        model.n_valid_ = int(payload.get("n_valid", 0))
        model.n_invalid_ = int(payload.get("n_invalid", 0))
        model._fitted = True
        return model

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ConformalDepth(alpha={self.alpha!r}, score={self.score!r}, "
            f"min_group_size={self.min_group_size!r}, pooled_q={self.pooled_quantile_!r})"
        )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_GROUP_SIZE",
    "DEPTH_BIN_EDGES",
    "SCORE_OVERESTIMATION",
    "SCORE_TWO_SIDED",
    "ConformalDepth",
    "ConformalPrediction",
    "conformal_quantile",
    "depth_bin_groups",
]
