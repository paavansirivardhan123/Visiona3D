"""Certified floor detection with conformal risk control.

For a floor-probability map and a threshold ``lambda``, define the per-image
non-floor false-positive rate as

``L_i(lambda) = count(p_floor >= lambda AND gt is non-floor) /
                count(gt is non-floor)``.

Ground-truth labels 0 (unlabeled) and 255 (ignore) are excluded, and pixels in
the optional sky mask are excluded from both numerator and denominator.  An
optional maximum ground-truth depth restricts calibration to the range in
which the navigation decision is made.

The selected threshold is the smallest grid value satisfying

``(n / (n + 1)) * mean_i L_i(lambda) + 1 / (n + 1) <= delta``.

This is the split-conformal risk-control correction for a loss bounded in
``[0, 1]``.  It controls the *marginal* expected false-positive rate over
exchangeable images, not the per-frame worst case and not the rate conditional
on a route, scene, or floor-probability value.

Why this loss?
--------------
Normalising by the number of non-floor ground-truth pixels measures how much
of the dangerous (non-walkable) surface is mistakenly certified as floor.  A
denominator of predicted-floor pixels would let a model improve its score by
predicting almost no floor, which is exactly the failure mode a navigation
system must not reward.  Averaging per image, rather than pooling all pixels
into one giant image, prevents a single large frame from dominating the
certificate.  The price is that rare small obstacles still receive the same
per-image weight as large walls; the risk target must therefore be chosen with
the application's tolerance for rare false walkable regions in mind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from core.config import MASK_FLOOR, MASK_IGNORE, MASK_UNLABELED

# Every threshold and default here is a starting point, not a validated
# real-navigation setting.  TO TUNE ON REAL DATA.
DEFAULT_DELTA = 0.05
DEFAULT_LAMBDA_GRID: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.0, 1.0, 101))


def _floor_probabilities(probs: np.ndarray) -> np.ndarray:
    """Accept ``(H, W, 4)`` probabilities or a ``(H, W)`` floor map."""
    arr = np.asarray(probs)
    if arr.ndim == 3:
        if arr.shape[-1] < 1:
            raise ValueError("probability map has no channels")
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"probs must be (H, W) or (H, W, C), got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def certified_floor(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Return the boolean certified-floor mask ``p_floor >= threshold``.

    NaN probabilities are never certified.  This function is deliberately a
    single comparison so the direction is easy to audit and mutate in negative
    controls.
    """
    floor = _floor_probabilities(probs)
    threshold = float(threshold)
    if math.isnan(threshold):
        return np.zeros(floor.shape, dtype=bool)
    return np.isfinite(floor) & (floor >= threshold)


def _valid_ground_truth(gt: np.ndarray) -> np.ndarray:
    gt_arr = np.asarray(gt)
    return (gt_arr != MASK_UNLABELED) & (gt_arr != MASK_IGNORE)


def _depth_restriction(
    depth: np.ndarray | None,
    shape: tuple[int, ...],
    max_depth_m: float | None,
) -> np.ndarray:
    if depth is None or max_depth_m is None:
        return np.ones(shape, dtype=bool)
    depth_arr = np.asarray(depth, dtype=np.float64)
    if depth_arr.shape != shape:
        raise ValueError(f"depth shape {depth_arr.shape} != mask shape {shape}")
    return np.isfinite(depth_arr) & (depth_arr > 0.0) & (depth_arr <= float(max_depth_m))


def floor_loss(
    probs: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float,
    sky_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    max_depth_m: float | None = None,
) -> dict[str, float]:
    """Compute one image's floor false-positive and related rates.

    Returns a dict with ``loss`` (the non-floor false-positive rate), the
    numerator/denominator, ``false_walkable_rate`` (fraction of predicted-floor
    pixels that are actually non-floor), and ``floor_recall``.  Rates with an
    empty denominator are NaN rather than invented zeros.
    """
    floor_prob = _floor_probabilities(probs)
    gt_arr = np.asarray(gt_mask)
    if floor_prob.shape != gt_arr.shape:
        raise ValueError(
            f"probs shape {floor_prob.shape} != gt_mask shape {gt_arr.shape}"
        )

    valid = _valid_ground_truth(gt_arr)
    if sky_mask is not None:
        sky = np.asarray(sky_mask, dtype=bool)
        if sky.shape != gt_arr.shape:
            raise ValueError(f"sky_mask shape {sky.shape} != gt_mask shape {gt_arr.shape}")
        valid &= ~sky
    valid &= _depth_restriction(depth, gt_arr.shape, max_depth_m)

    nonfloor_gt = valid & (gt_arr != MASK_FLOOR)
    floor_gt = valid & (gt_arr == MASK_FLOOR)
    predicted_floor = certified_floor(floor_prob, threshold) & valid

    nonfloor_count = int(np.sum(nonfloor_gt))
    false_positive_count = int(np.sum(predicted_floor & nonfloor_gt))
    predicted_floor_count = int(np.sum(predicted_floor))
    floor_count = int(np.sum(floor_gt))
    true_positive_count = int(np.sum(predicted_floor & floor_gt))

    loss = (
        float(false_positive_count / nonfloor_count)
        if nonfloor_count > 0
        else float("nan")
    )
    false_walkable = (
        float(false_positive_count / predicted_floor_count)
        if predicted_floor_count > 0
        else float("nan")
    )
    recall = (
        float(true_positive_count / floor_count) if floor_count > 0 else float("nan")
    )

    return {
        "loss": loss,
        "false_positive_count": float(false_positive_count),
        "nonfloor_count": float(nonfloor_count),
        "false_walkable_rate": false_walkable,
        "floor_recall": recall,
        "predicted_floor_count": float(predicted_floor_count),
        "floor_count": float(floor_count),
    }


def _loss_curve(
    prob_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    thresholds: Sequence[float],
    sky_masks: Sequence[np.ndarray] | None = None,
    depth_maps: Sequence[np.ndarray] | None = None,
    max_depth_m: float | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return an ``(n_images, n_thresholds)`` loss matrix and per-image info."""
    n_images = len(prob_maps)
    if len(gt_masks) != n_images:
        raise ValueError("prob_maps and gt_masks must have equal length")
    if sky_masks is not None and len(sky_masks) != n_images:
        raise ValueError("sky_masks must have the same length as prob_maps")
    if depth_maps is not None and len(depth_maps) != n_images:
        raise ValueError("depth_maps must have the same length as prob_maps")

    losses = np.full((n_images, len(thresholds)), np.nan, dtype=np.float64)
    info: list[dict[str, Any]] = []
    for index in range(n_images):
        per_image: dict[str, Any] = {"index": index}
        for t_index, threshold in enumerate(thresholds):
            row = floor_loss(
                prob_maps[index],
                gt_masks[index],
                threshold,
                sky_mask=None if sky_masks is None else sky_masks[index],
                depth=None if depth_maps is None else depth_maps[index],
                max_depth_m=max_depth_m,
            )
            losses[index, t_index] = row["loss"]
            if t_index == 0:
                per_image.update(
                    {
                        "nonfloor_count": int(row["nonfloor_count"]),
                        "floor_count": int(row["floor_count"]),
                    }
                )
        info.append(per_image)
    return losses, info


def select_floor_threshold(
    per_image_losses: Sequence[float] | np.ndarray,
    delta: float = DEFAULT_DELTA,
    thresholds: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Select the smallest certified threshold.

    Args:
        per_image_losses: Losses for one threshold per image, or an
            ``(n_images, n_thresholds)`` matrix.
        delta: Target marginal false-positive rate in ``(0, 1)``.
        thresholds: Candidate thresholds in ascending order.

    Returns:
        ``(selected_threshold, mean_losses, risk_bounds)``.  When no threshold
        certifies the target, the selected threshold is ``+inf`` and all
        certified masks are empty.
    """
    delta = float(delta)
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    thresholds_arr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if thresholds_arr.ndim != 1 or thresholds_arr.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")

    losses = np.asarray(per_image_losses, dtype=np.float64)
    if losses.ndim == 1:
        losses = losses.reshape(-1, 1)
    if losses.ndim != 2:
        raise ValueError("per_image_losses must be 1D or 2D")
    if losses.shape[0] == 0:
        return math.inf, np.full(thresholds_arr.size, np.nan), np.full(thresholds_arr.size, np.nan)
    if losses.shape[1] not in (1, thresholds_arr.size):
        raise ValueError(
            "loss matrix columns must be 1 or match the number of thresholds"
        )

    mean_losses = np.nanmean(losses, axis=0)
    n = losses.shape[0]
    risk_bounds = (n / (n + 1.0)) * mean_losses + 1.0 / (n + 1.0)
    if losses.shape[1] == 1:
        mean_losses = np.repeat(mean_losses, thresholds_arr.size)
        risk_bounds = np.repeat(risk_bounds, thresholds_arr.size)

    satisfying = np.isfinite(risk_bounds) & (risk_bounds <= delta)
    if not np.any(satisfying):
        return math.inf, mean_losses, risk_bounds
    selected_index = int(np.flatnonzero(satisfying)[0])
    return float(thresholds_arr[selected_index]), mean_losses, risk_bounds


@dataclass
class FloorCalibration:
    """Certified-floor calibration state.

    Attributes:
        delta: Target marginal non-floor false-positive rate.  TO TUNE.
        thresholds: Candidate grid.  TO TUNE.
        max_depth_m: Optional calibration restriction to GT depth <= R.
        threshold: Selected threshold, or ``+inf`` if no certificate exists.
        losses_by_image: ``(n_images, n_thresholds)`` calibration losses.
        mean_losses: Mean loss at each threshold.
        risk_bounds: Conformal risk bound at each threshold.
        per_image: Metadata for each calibration image.
        monotonic: True if every image loss is non-increasing with threshold.
    """

    delta: float = DEFAULT_DELTA
    thresholds: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    max_depth_m: float | None = None
    threshold: float = math.inf
    losses_by_image: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    mean_losses: np.ndarray = field(default_factory=lambda: np.empty(0))
    risk_bounds: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_image: list[dict[str, Any]] = field(default_factory=list)
    monotonic: bool = True
    n_images: int = 0
    n_images_with_nonfloor: int = 0
    calibration_false_positive_rate: float = float("nan")
    calibration_false_walkable_rate: float = float("nan")
    calibration_floor_recall: float = float("nan")
    _fitted: bool = False

    def fit(
        self,
        prob_maps: Sequence[np.ndarray],
        gt_masks: Sequence[np.ndarray],
        sky_masks: Sequence[np.ndarray] | None = None,
        depth_maps: Sequence[np.ndarray] | None = None,
    ) -> "FloorCalibration":
        """Fit the certified threshold on calibration images."""
        if len(prob_maps) != len(gt_masks):
            raise ValueError("prob_maps and gt_masks must have equal length")
        if len(prob_maps) == 0:
            self._fitted = True
            return self

        losses, info = _loss_curve(
            prob_maps,
            gt_masks,
            self.thresholds,
            sky_masks=sky_masks,
            depth_maps=depth_maps,
            max_depth_m=self.max_depth_m,
        )
        self.losses_by_image = losses
        self.per_image = info
        self.n_images = int(losses.shape[0])
        usable = np.any(np.isfinite(losses), axis=1)
        self.n_images_with_nonfloor = int(np.sum(usable))

        # Recompute the certificate using only images that contain at least
        # one valid non-floor pixel.  Images without a denominator carry no
        # information about false floor positives.
        usable_losses = losses[usable]
        if usable_losses.shape[0] == 0:
            self.threshold = math.inf
            self.mean_losses = np.full(len(self.thresholds), np.nan)
            self.risk_bounds = np.full(len(self.thresholds), np.nan)
            self.monotonic = True
            self._fitted = True
            return self

        self.threshold, self.mean_losses, self.risk_bounds = select_floor_threshold(
            usable_losses, self.delta, self.thresholds
        )

        # Monotonicity is a property of the predicate p >= lambda: raising the
        # threshold can only remove predicted-floor pixels.  Check the
        # computed matrix rather than assuming it.
        finite = np.isfinite(usable_losses)
        if finite.any():
            diffs = np.diff(usable_losses, axis=1)
            self.monotonic = bool(np.all(diffs[finite[:, 1:]] <= 1e-12))
        else:
            self.monotonic = True

        # Aggregate calibration metrics at the selected threshold.
        if math.isfinite(self.threshold):
            total_nonfloor = 0
            total_false = 0
            total_pred_floor = 0
            total_floor = 0
            total_true = 0
            for prob, gt, sky, depth in zip(
                prob_maps,
                gt_masks,
                sky_masks if sky_masks is not None else [None] * len(prob_maps),
                depth_maps if depth_maps is not None else [None] * len(prob_maps),
            ):
                row = floor_loss(
                    prob,
                    gt,
                    self.threshold,
                    sky_mask=sky,
                    depth=depth,
                    max_depth_m=self.max_depth_m,
                )
                total_nonfloor += int(row["nonfloor_count"])
                total_false += int(row["false_positive_count"])
                total_pred_floor += int(row["predicted_floor_count"])
                total_floor += int(row["floor_count"])
                total_true += int(
                    row["predicted_floor_count"] - row["false_positive_count"]
                )
            if total_nonfloor > 0:
                self.calibration_false_positive_rate = float(total_false / total_nonfloor)
            if total_pred_floor > 0:
                self.calibration_false_walkable_rate = float(
                    total_false / total_pred_floor
                )
            if total_floor > 0:
                self.calibration_floor_recall = float(total_true / total_floor)

        self._fitted = True
        return self

    def certify(self, probs: np.ndarray) -> np.ndarray:
        """Apply the selected threshold to one probability map."""
        if not self._fitted:
            raise RuntimeError("FloorCalibration.fit() must be called first")
        return certified_floor(probs, self.threshold)

    def metrics(
        self,
        probs: np.ndarray,
        gt_mask: np.ndarray,
        sky_mask: np.ndarray | None = None,
        depth: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Evaluate one map at the selected threshold."""
        return floor_loss(
            probs,
            gt_mask,
            self.threshold,
            sky_mask=sky_mask,
            depth=depth,
            max_depth_m=self.max_depth_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "delta": self.delta,
            "thresholds": [float(x) for x in self.thresholds],
            "max_depth_m": self.max_depth_m,
            "threshold": self.threshold if math.isfinite(self.threshold) else None,
            "mean_losses": [float(x) for x in self.mean_losses],
            "risk_bounds": [float(x) for x in self.risk_bounds],
            "monotonic": self.monotonic,
            "n_images": self.n_images,
            "n_images_with_nonfloor": self.n_images_with_nonfloor,
            "calibration_false_positive_rate": self.calibration_false_positive_rate,
            "calibration_false_walkable_rate": self.calibration_false_walkable_rate,
            "calibration_floor_recall": self.calibration_floor_recall,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FloorCalibration":
        threshold = payload.get("threshold")
        obj = cls(
            delta=float(payload["delta"]),
            thresholds=tuple(float(x) for x in payload["thresholds"]),
            max_depth_m=payload.get("max_depth_m"),
            threshold=math.inf if threshold is None else float(threshold),
        )
        obj.mean_losses = np.asarray(payload.get("mean_losses", []), dtype=np.float64)
        obj.risk_bounds = np.asarray(payload.get("risk_bounds", []), dtype=np.float64)
        obj.monotonic = bool(payload.get("monotonic", True))
        obj.n_images = int(payload.get("n_images", 0))
        obj.n_images_with_nonfloor = int(
            payload.get("n_images_with_nonfloor", 0)
        )
        obj.calibration_false_positive_rate = float(
            payload.get("calibration_false_positive_rate", float("nan"))
        )
        obj.calibration_false_walkable_rate = float(
            payload.get("calibration_false_walkable_rate", float("nan"))
        )
        obj.calibration_floor_recall = float(
            payload.get("calibration_floor_recall", float("nan"))
        )
        obj._fitted = True
        return obj


__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_LAMBDA_GRID",
    "FloorCalibration",
    "certified_floor",
    "floor_loss",
    "select_floor_threshold",
]
