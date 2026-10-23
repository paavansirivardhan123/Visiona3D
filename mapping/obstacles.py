"""Nearest-obstacle telemetry in the ground frame G.

For every obstacle pixel ``i`` the true distance lies in
``[lower_i, upper_i]`` when conformal bounds are supplied. The true nearest
distance ``D = min_i d_i`` therefore satisfies ``min_i lower_i <= D <=
min_i upper_i``, so the minima are valid bounds even when the nearest nominal
pixel is not the pixel that attains a bound.

Missing bounds are omitted rather than replaced by the nominal distance. This
module makes no coverage claim; coverage comes only from the calibration used
to fit the interval, and every threshold remains TO TUNE ON REAL DATA.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _flatten_points(points: Any, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr.reshape(-1, arr.shape[2])
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{name} must be (H, W, 3) or (N, 3), got {arr.shape}")
    return arr[:, :3]


def _nearest_bound(
    pointmap: Any,
    obstacle_mask: np.ndarray,
    n_points: int,
    name: str,
) -> float | None:
    if pointmap is None:
        return None
    pts = _flatten_points(pointmap, name)
    if pts.shape[0] != n_points:
        raise ValueError(
            f"{name} must contain the same number of points as pointmap_g; "
            f"got {pts.shape[0]} and {n_points}"
        )
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    selected = obstacle_mask & finite
    if not np.any(selected):
        return None
    return float(np.min(np.hypot(pts[selected, 0], pts[selected, 1])))


def nearest_obstacle_telemetry(
    pointmap_g: Any,
    obstacle_mask: Any,
    *,
    lower_pointmap_g: Any | None = None,
    upper_pointmap_g: Any | None = None,
) -> dict[str, Any]:
    """Return nominal and conformal-bound distances to the nearest obstacle.

    Args:
        pointmap_g: Nominal ground-frame points, ``(H, W, 3)`` or ``(N, 3)``.
        obstacle_mask: Boolean pixels/points classified as obstacle.
        lower_pointmap_g: Optional conformal lower-bound points.
        upper_pointmap_g: Optional conformal upper-bound points.

    Returns:
        ``{"label", "distance_nominal_m", "bearing_deg"}`` plus
        ``"distance_lower_m"``/``"distance_upper_m"`` when a finite bound
        exists. With no obstacle the label is ``"none"`` and the distance is
        ``+inf``.
    """
    pts = _flatten_points(pointmap_g, "pointmap_g")
    mask = np.asarray(obstacle_mask, dtype=bool).reshape(-1)
    if mask.size != pts.shape[0]:
        raise ValueError(
            "obstacle_mask must match pointmap_g length; "
            f"got {mask.size} and {pts.shape[0]}"
        )

    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    selected = mask & finite
    if not np.any(selected):
        return {
            "label": "none",
            "distance_nominal_m": math.inf,
            "bearing_deg": 0.0,
        }

    indices = np.flatnonzero(selected)
    distances = np.hypot(pts[indices, 0], pts[indices, 1])
    nearest_index = int(indices[int(np.argmin(distances))])
    x = float(pts[nearest_index, 0])
    y = float(pts[nearest_index, 1])

    telemetry: dict[str, Any] = {
        "label": "obstacle",
        "distance_nominal_m": float(math.hypot(x, y)),
        "bearing_deg": float(math.degrees(math.atan2(x, y))),
    }

    lower = _nearest_bound(lower_pointmap_g, selected, pts.shape[0], "lower_pointmap_g")
    if lower is not None:
        telemetry["distance_lower_m"] = lower
    upper = _nearest_bound(upper_pointmap_g, selected, pts.shape[0], "upper_pointmap_g")
    if upper is not None:
        telemetry["distance_upper_m"] = upper
    return telemetry


__all__ = ["nearest_obstacle_telemetry"]
