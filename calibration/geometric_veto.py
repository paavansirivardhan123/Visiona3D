"""Geometric veto for floor-labelled points that sit above the ground plane.

The certified-floor stage is semantic: it can certify a vertical wall base, a
curb, or a low obstacle because their floor probability is high.  This stage
uses the metric lifted point cloud and the fitted ground plane to reject any
floor-labelled point whose height above the plane exceeds

``tau(y) = tau0 + tau1 * y``

where ``y`` is forward distance in ground frame G.  The linear term allows the
tolerance to grow with range, where depth noise and plane tilt are larger.

``certified_final = certified_floor AND NOT vetoed``.

Every threshold is a starting point to be tuned on real data.  The veto cannot
recover an obstacle that is entirely below ``tau`` (a low curb, a step, or the
base of a wall within the tolerance band).  It also cannot certify transparent
or reflective surfaces, and it inherits every error in the fitted plane,
metric scale, and depth map.  It is a veto, not a complete safety proof.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Both values are deliberately conservative initialisations.
# TO TUNE ON REAL DATA.
DEFAULT_TAU0_M = 0.10
DEFAULT_TAU1 = 0.02  # metres of tolerance per metre of forward distance

DEFAULT_WALL_BANDS_M: tuple[tuple[float, float | None], ...] = (
    (0.0, 0.10),
    (0.10, 0.30),
    (0.30, None),
)


def _points(points_g: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_g, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points_g must have shape (N, 3), got {pts.shape}")
    return pts[:, :3]


def height_threshold(
    forward_distance_m: Sequence[float] | np.ndarray,
    tau0_m: float = DEFAULT_TAU0_M,
    tau1: float = DEFAULT_TAU1,
) -> np.ndarray:
    """Return ``tau(y) = tau0 + tau1 * max(y, 0)``.

    Clipping forward distance at zero avoids shrinking the tolerance for
    points behind the camera plane.
    """
    y = np.asarray(forward_distance_m, dtype=np.float64)
    return float(tau0_m) + float(tau1) * np.maximum(y, 0.0)


def geometric_veto(
    points_g: np.ndarray,
    tau0_m: float = DEFAULT_TAU0_M,
    tau1: float = DEFAULT_TAU1,
) -> np.ndarray:
    """Return a boolean veto mask for points in ground frame G.

    Non-finite points are vetoed (True) because they cannot be certified.
    A point is vetoed when ``Z > tau(Y)``.  Points at or below the threshold
    are retained; this is why low obstacles can still pass.
    """
    pts = _points(points_g)
    finite = np.isfinite(pts).all(axis=1)
    y = pts[:, 1]
    z = pts[:, 2]
    with np.errstate(invalid="ignore"):
        above = z > height_threshold(y, tau0_m, tau1)
    return (~finite) | above


def combine_certified_floor(
    certified_mask: np.ndarray,
    veto_mask: np.ndarray,
) -> np.ndarray:
    """Combine semantic certification and geometric veto.

    Shapes must match.  The result is True only where the semantic mask is
    True and the veto is False.
    """
    certified = np.asarray(certified_mask, dtype=bool)
    veto = np.asarray(veto_mask, dtype=bool)
    if certified.shape != veto.shape:
        raise ValueError(
            f"certified_mask shape {certified.shape} != veto_mask shape {veto.shape}"
        )
    return certified & ~veto


def veto_floor_mask(
    certified_mask: np.ndarray,
    points_g_full: np.ndarray,
    tau0_m: float = DEFAULT_TAU0_M,
    tau1: float = DEFAULT_TAU1,
) -> tuple[np.ndarray, np.ndarray]:
    """Veto a full-image certified mask using full-image ground-frame points.

    Args:
        certified_mask: ``(H, W)`` boolean semantic certification.
        points_g_full: ``(H, W, 3)`` ground-frame points.
        tau0_m, tau1: Veto threshold parameters.

    Returns:
        ``(final_mask, veto_mask)``, both ``(H, W)``.
    """
    certified = np.asarray(certified_mask, dtype=bool)
    pts = np.asarray(points_g_full, dtype=np.float64)
    if pts.shape[:2] != certified.shape or pts.shape[-1] < 3:
        raise ValueError(
            f"points_g_full must be (H, W, 3) matching {certified.shape}, got {pts.shape}"
        )
    veto = geometric_veto(pts.reshape(-1, pts.shape[-1])[:, :3], tau0_m, tau1)
    veto = veto.reshape(certified.shape)
    return combine_certified_floor(certified, veto), veto


def veto_recall_by_band(
    points_g: np.ndarray,
    ground_truth_labels: np.ndarray,
    veto_mask: np.ndarray,
    bands: Sequence[tuple[float, float | None]] = DEFAULT_WALL_BANDS_M,
) -> dict[str, dict[str, float]]:
    """Report veto recall for non-floor points by height band.

    Only labels that are valid (not 0/255) and not floor are included.  Sky is
    expected to have been removed by the caller or represented as other; if it
    remains, it is counted as non-floor and should be excluded upstream.
    """
    pts = _points(points_g)
    labels = np.asarray(ground_truth_labels).reshape(-1)
    veto = np.asarray(veto_mask, dtype=bool).reshape(-1)
    if labels.size != pts.shape[0] or veto.size != pts.shape[0]:
        raise ValueError("points_g, ground_truth_labels and veto_mask must have equal length")

    valid = (labels != 0) & (labels != 255)
    nonfloor = valid & (labels != 1)
    z = pts[:, 2]
    result: dict[str, dict[str, float]] = {}
    for low, high in bands:
        if high is None:
            in_band = nonfloor & (z >= low)
            name = f">{low:.1f}m"
        else:
            in_band = nonfloor & (z >= low) & (z < high)
            name = f"{low:.1f}-{high:.1f}m"
        count = int(np.sum(in_band))
        recall = float(np.mean(veto[in_band])) if count else float("nan")
        result[name] = {"count": float(count), "recall": recall}
    return result


def false_veto_rates_by_range(
    points_g: np.ndarray,
    floor_mask: np.ndarray,
    veto_mask: np.ndarray,
    ranges_m: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    half_width_m: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Report false-veto rates for true-floor points in forward range bins.

    Args:
        points_g: ``(N, 3)`` lifted points.
        floor_mask: Boolean mask of true floor points.
        veto_mask: Boolean veto output for the same points.
        ranges_m: Bin centres in metres.
        half_width_m: Bin half-width in metres.

    The caller is responsible for producing points from a known floor scene
    and applying the depth-noise model under test.
    """
    pts = _points(points_g)
    floor = np.asarray(floor_mask, dtype=bool).reshape(-1)
    veto = np.asarray(veto_mask, dtype=bool).reshape(-1)
    if floor.size != pts.shape[0] or veto.size != pts.shape[0]:
        raise ValueError("points_g, floor_mask and veto_mask must have equal length")

    result: dict[str, dict[str, float]] = {}
    y = pts[:, 1]
    for centre in ranges_m:
        in_bin = floor & (y >= float(centre) - half_width_m) & (y < float(centre) + half_width_m)
        count = int(np.sum(in_bin))
        rate = float(np.mean(veto[in_bin])) if count else float("nan")
        result[f"{float(centre):.1f}m"] = {"count": float(count), "false_veto_rate": rate}
    return result


def evaluation_summary(
    points_g: np.ndarray,
    ground_truth_labels: np.ndarray,
    veto_mask: np.ndarray,
    bands: Sequence[tuple[float, float | None]] = DEFAULT_WALL_BANDS_M,
) -> dict[str, Any]:
    """Convenience wrapper returning band recall and overall veto fraction."""
    pts = _points(points_g)
    labels = np.asarray(ground_truth_labels).reshape(-1)
    veto = np.asarray(veto_mask, dtype=bool).reshape(-1)
    valid = (labels != 0) & (labels != 255)
    floor = valid & (labels == 1)
    return {
        "recall_by_band": veto_recall_by_band(pts, labels, veto, bands),
        "overall_veto_fraction": float(np.mean(veto[valid])) if np.any(valid) else float("nan"),
        "floor_veto_fraction": float(np.mean(veto[floor])) if np.any(floor) else float("nan"),
    }


__all__ = [
    "DEFAULT_TAU0_M",
    "DEFAULT_TAU1",
    "DEFAULT_WALL_BANDS_M",
    "combine_certified_floor",
    "evaluation_summary",
    "false_veto_rates_by_range",
    "geometric_veto",
    "height_threshold",
    "veto_floor_mask",
    "veto_recall_by_band",
]
