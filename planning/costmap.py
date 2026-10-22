"""Costmap and chamfer distance transform for 2D ground-frame planning.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Union

import numpy as np

from mapping.grid import (
    CELL_FREE,
    CELL_UNCERTAIN,
    CELL_UNKNOWN,
    MapConfig,
)

# Inflation parameters - labelled TO TUNE ON REAL DATA
DEFAULT_USER_HALF_WIDTH_M: float = 0.35  # TO TUNE ON REAL DATA
DEFAULT_SAFETY_MARGIN_M: float = 0.10  # TO TUNE ON REAL DATA
DEFAULT_CLEARANCE_WEIGHT_K: float = 2.0  # TO TUNE ON REAL DATA
DEFAULT_C_MAX_M: float = 1.5  # TO TUNE ON REAL DATA: max clearance consideration distance


SQRT2 = math.sqrt(2.0)


def chamfer_distance_transform(
    blocked_mask: np.ndarray,
    res: float = 0.10,
) -> np.ndarray:
    """Numpy-only two-pass Chamfer distance transform (1, sqrt(2) metric).

    Computes distance in metres from every cell to the nearest blocked cell.
    Blocked cells have distance 0.0.

    Implemented as the classic two-pass dynamic programme with the per-row
    horizontal scans vectorised via prefix minimum (mathematically identical
    recurrence: d[c] = min(cand[c], d[c-1] + 1)).

    Args:
        blocked_mask: 2D boolean array where True indicates a blocked cell.
        res: Grid resolution in metres.

    Returns:
        2D float64 array of metric clearance distances in metres.
    """
    mask = np.asarray(blocked_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"blocked_mask must be 2D, got shape {mask.shape}")

    n_rows, n_cols = mask.shape
    dist = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
    dist[mask] = 0.0

    # If no blocked cells, return large finite distance
    if not np.any(mask):
        return dist

    idx = np.arange(n_cols, dtype=np.float64)

    # Forward pass: top-to-bottom, left-to-right
    for r in range(n_rows):
        cand = dist[r].copy()
        if r > 0:
            up = dist[r - 1]
            np.minimum(cand, up + 1.0, out=cand)
            np.minimum(cand[1:], up[:-1] + SQRT2, out=cand[1:])
            np.minimum(cand[:-1], up[1:] + SQRT2, out=cand[:-1])
        # Left-to-right scan: d[c] = min(cand[c], d[c-1] + 1)
        u = cand - idx
        np.minimum.accumulate(u, out=u)
        dist[r] = u + idx

    # Backward pass: bottom-to-top, right-to-left
    for r in range(n_rows - 1, -1, -1):
        cand = dist[r].copy()
        if r + 1 < n_rows:
            down = dist[r + 1]
            np.minimum(cand, down + 1.0, out=cand)
            np.minimum(cand[1:], down[:-1] + SQRT2, out=cand[1:])
            np.minimum(cand[:-1], down[1:] + SQRT2, out=cand[:-1])
        # Right-to-left scan: d[c] = min(cand[c], d[c+1] + 1)
        u = cand + idx
        u = u[::-1].copy()
        np.minimum.accumulate(u, out=u)
        dist[r] = u[::-1] - idx

    return dist * float(res)


def brute_force_chamfer_distance(
    blocked_mask: np.ndarray,
    res: float = 0.10,
) -> np.ndarray:
    """Exact brute-force distance to blocked cells using the (1, sqrt(2)) metric.

    Used for validation in tests.
    """
    mask = np.asarray(blocked_mask, dtype=bool)
    n_rows, n_cols = mask.shape
    dist = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
    blocked_coords = np.argwhere(mask)  # (N_blocked, 2) [r, c]

    if len(blocked_coords) == 0:
        return dist

    b_r = blocked_coords[:, 0]
    b_c = blocked_coords[:, 1]

    for r in range(n_rows):
        for c in range(n_cols):
            if mask[r, c]:
                dist[r, c] = 0.0
                continue
            dr = np.abs(b_r - r)
            dc = np.abs(b_c - c)
            # Octile metric: max(dr, dc) + (sqrt(2) - 1) * min(dr, dc)
            cell_dists = np.maximum(dr, dc) + (SQRT2 - 1.0) * np.minimum(dr, dc)
            dist[r, c] = np.min(cell_dists)

    return dist * float(res)


@dataclass
class Costmap:
    """Planning costmap with clearance field and inflated obstacles."""

    grid: np.ndarray  # Original cell states (CELL_UNKNOWN, etc.)
    blocked_mask: np.ndarray  # Raw blocked cells (OCCUPIED, UNKNOWN, UNCERTAIN)
    clearance_m: np.ndarray  # Distance in metres to nearest raw blocked cell
    inflated_blocked: np.ndarray  # Cells with clearance <= inflation_radius
    inflation_radius_m: float  # User half width + safety margin
    cfg: MapConfig
    k_clearance: float = DEFAULT_CLEARANCE_WEIGHT_K  # TO TUNE ON REAL DATA
    c_max_m: float = DEFAULT_C_MAX_M  # TO TUNE ON REAL DATA

    def __post_init__(self) -> None:
        # Cache grid dimensions: is_passable and the planner loops call these
        # millions of times and the MapConfig properties are too slow per call.
        self.n_rows = int(self.cfg.n_rows)
        self.n_cols = int(self.cfg.n_cols)

    def is_passable(self, r: int, c: int) -> bool:
        """Check if cell (r, c) is traversable (not inflated blocked)."""
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return not bool(self.inflated_blocked[r, c])

    def cell_cost_multiplier(self, r: int, c: int) -> float:
        """Compute traversal cost multiplier for cell (r, c).

        cost = step * (1 + k * ((c_max - clearance)/c_max)^2 for clearance < c_max).
        """
        if not self.is_passable(r, c):
            return math.inf
        clr = self.clearance_m[r, c]
        if clr >= self.c_max_m:
            return 1.0
        ratio = (self.c_max_m - clr) / self.c_max_m
        return 1.0 + self.k_clearance * (ratio * ratio)


def build_costmap(
    bev_grid: np.ndarray,
    cfg: MapConfig | None = None,
    user_half_width_m: float = DEFAULT_USER_HALF_WIDTH_M,
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M,
    k_clearance: float = DEFAULT_CLEARANCE_WEIGHT_K,
    c_max_m: float = DEFAULT_C_MAX_M,
) -> Costmap:
    """Build a planning costmap from a BEV grid.

    Safety Principle 1 enforcement:
    Free space must be certified. Unknown and uncertain cells are NOT traversable.
    Therefore: blocked = (grid != CELL_FREE).

    Args:
        bev_grid: 2D array of cell states.
        cfg: MapConfig instance (defaults if None).
        user_half_width_m: User half-width in metres (default 0.35 m). TO TUNE.
        safety_margin_m: Safety margin in metres (default 0.10 m). TO TUNE.
        k_clearance: Clearance penalty weight. TO TUNE.
        c_max_m: Maximum clearance cutoff in metres. TO TUNE.

    Returns:
        Costmap instance.
    """
    if cfg is None:
        cfg = MapConfig()

    grid = np.asarray(bev_grid, dtype=np.uint8)
    if grid.shape != (cfg.n_rows, cfg.n_cols):
        raise ValueError(f"bev_grid shape {grid.shape} != cfg shape {cfg.shape}")

    # Safety Principle 1: Free space must be certified. Unknown and uncertain are NOT traversable.
    raw_blocked = grid != CELL_FREE

    clearance_m = chamfer_distance_transform(raw_blocked, res=cfg.res)

    inflation_radius_m = float(user_half_width_m) + float(safety_margin_m)

    # A cell is inflated blocked if clearance <= inflation_radius
    # Small tolerance for numerical roundoff
    if inflation_radius_m > 0.0:
        inflated_blocked = raw_blocked | (clearance_m <= inflation_radius_m + 1e-9)
    else:
        inflated_blocked = raw_blocked.copy()

    return Costmap(
        grid=grid,
        blocked_mask=raw_blocked,
        clearance_m=clearance_m,
        inflated_blocked=inflated_blocked,
        inflation_radius_m=inflation_radius_m,
        cfg=cfg,
        k_clearance=float(k_clearance),
        c_max_m=float(c_max_m),
    )
