"""Path smoothing with moving-average filtering and strict collision re-validation.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from mapping.grid import cell_to_world, in_bounds, world_to_cell
from planning.costmap import Costmap


def _segment_collides(
    p1: tuple[float, float],
    p2: tuple[float, float],
    costmap: Costmap,
    step_m: float,
) -> bool:
    """Check if linear segment between p1 and p2 passes through blocked/inflated cells."""
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1e-6:
        c, r = world_to_cell(x1, y1, costmap.cfg)
        return not costmap.is_passable(r, c)

    n_steps = max(int(math.ceil(dist / step_m)), 1)
    for s in range(n_steps + 1):
        alpha = s / float(n_steps)
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        c, r = world_to_cell(x, y, costmap.cfg)
        if not costmap.is_passable(r, c):
            return True
    return False


def smooth_path(
    path_world: Sequence[tuple[float, float]],
    costmap: Costmap,
    iterations: int = 3,  # TO TUNE ON REAL DATA
    weight_smooth: float = 0.5,  # TO TUNE ON REAL DATA
) -> tuple[list[tuple[float, float]], bool]:
    """Smooth a 2D metric path keeping endpoints pinned and re-validating against obstacles.

    If the smoothed path intersects any blocked cell, falls back to the raw path.

    Args:
        path_world: List of (X, Y) waypoints in ground frame G.
        costmap: Planning costmap.
        iterations: Number of smoothing filter passes.
        weight_smooth: Weight applied to neighboring points in [0, 1].

    Returns:
        (smoothed_path, fell_back_to_raw)
    """
    pts = list(path_world)
    if len(pts) <= 2:
        return pts, False

    arr = np.array(pts, dtype=np.float64)  # (N, 2)
    smoothed = arr.copy()

    w = float(np.clip(weight_smooth, 0.0, 0.8))
    w_curr = 1.0 - w
    w_neigh = 0.5 * w

    for _ in range(iterations):
        for i in range(1, len(arr) - 1):
            smoothed[i] = w_curr * smoothed[i] + w_neigh * (smoothed[i - 1] + smoothed[i + 1])

    smoothed_list = [tuple(float(v) for v in row) for row in smoothed]

    # Strict re-validation against blocked mask
    step_m = 0.5 * costmap.cfg.res
    for i in range(len(smoothed_list) - 1):
        if _segment_collides(smoothed_list[i], smoothed_list[i + 1], costmap, step_m):
            # Collision detected: FALL BACK TO RAW PATH
            return pts, True

    return smoothed_list, False
