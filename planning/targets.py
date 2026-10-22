"""Target selection strategies: explore mode, anchor mode, and multi-path candidates.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from mapping.grid import CELL_FREE, MapConfig, cell_to_world, world_to_cell
from planning.astar import astar_plan, PathResult
from planning.costmap import Costmap

# Target parameters - labelled TO TUNE ON REAL DATA
DEFAULT_EXPLORE_CONE_DEG: float = 60.0  # TO TUNE ON REAL DATA
DEFAULT_MIN_EXPLORE_DIST_M: float = 1.5  # TO TUNE ON REAL DATA
DEFAULT_ANCHOR_SEARCH_RADIUS_M: float = 1.5  # TO TUNE ON REAL DATA


def _reachable_mask(
    costmap: Costmap,
    start_cell: tuple[int, int],
) -> np.ndarray:
    """Vectorised 8-connected flood fill with the corner-cut rule.

    Iteratively dilates the reached set with numpy shifts until fixpoint.
    Produces exactly the same set as the equivalent breadth-first search:
    the move relation is symmetric (a diagonal (r, c) -> (nr, nc) is allowed
    iff both orthogonal neighbours are passable, which is direction-independent).
    """
    start_col, start_row = int(start_cell[0]), int(start_cell[1])
    n_rows, n_cols = costmap.n_rows, costmap.n_cols
    passable = ~costmap.inflated_blocked

    reach = np.zeros((n_rows, n_cols), dtype=bool)
    if not (0 <= start_row < n_rows and 0 <= start_col < n_cols):
        return reach
    if not passable[start_row, start_col]:
        return reach

    reach[start_row, start_col] = True
    while True:
        prev = reach
        new = reach.copy()
        # Orthogonal dilation
        new[1:, :] |= prev[:-1, :] & passable[1:, :]
        new[:-1, :] |= prev[1:, :] & passable[:-1, :]
        new[:, 1:] |= prev[:, :-1] & passable[:, 1:]
        new[:, :-1] |= prev[:, 1:] & passable[:, :-1]
        # Diagonal dilation with the corner-cut rule:
        # (r, c) -> (r + dr, c + dc) requires passable[r, c + dc] and passable[r + dr, c]
        new[1:, 1:] |= prev[:-1, :-1] & passable[1:, 1:] & passable[:-1, 1:] & passable[1:, :-1]
        new[1:, :-1] |= prev[:-1, 1:] & passable[1:, :-1] & passable[:-1, :-1] & passable[1:, 1:]
        new[:-1, 1:] |= prev[1:, :-1] & passable[:-1, 1:] & passable[1:, 1:] & passable[:-1, :-1]
        new[:-1, :-1] |= prev[1:, 1:] & passable[:-1, :-1] & passable[1:, :-1] & passable[:-1, 1:]
        if np.array_equal(new, prev):
            return new
        reach = new


def get_reachable_cells(
    costmap: Costmap,
    start_cell: tuple[int, int],
) -> set[tuple[int, int]]:
    """Fast 8-connected flood fill returning all passable (col, row) cells reachable from start."""
    reach = _reachable_mask(costmap, start_cell)
    return {(int(c), int(r)) for r, c in np.argwhere(reach)}


def find_explore_target(
    costmap: Costmap,
    start_cell: tuple[int, int],
    cone_deg: float = DEFAULT_EXPLORE_CONE_DEG,
    min_dist_m: float = DEFAULT_MIN_EXPLORE_DIST_M,
) -> Optional[tuple[int, int]]:
    """Select farthest reachable certified FREE cell in forward cone of +-cone_deg.

    Ties are broken by distance to the centerline (smallest |X|).

    Args:
        costmap: Planning costmap.
        start_cell: (col, row) of planner start.
        cone_deg: Half-angle of forward exploration cone in degrees.
        min_dist_m: Minimum distance ahead in metres.

    Returns:
        (col, row) of chosen frontier target, or None if no reachable candidate.
    """
    reach = _reachable_mask(costmap, start_cell)
    if not reach.any():
        return None

    cfg = costmap.cfg
    n_rows, n_cols = costmap.n_rows, costmap.n_cols
    wx_all = cfg.x_min + (np.arange(n_cols, dtype=np.float64) + 0.5) * cfg.res
    wy_all = cfg.y_min + (np.arange(n_rows, dtype=np.float64) + 0.5) * cfg.res
    start_wx = float(wx_all[int(start_cell[0])])
    start_wy = float(wy_all[int(start_cell[1])])

    dx = wx_all[None, :] - start_wx
    dy = wy_all[:, None] - start_wy
    dist = np.hypot(dx, dy)

    candidates = (
        reach
        & (costmap.grid == CELL_FREE)
        & (dist >= min_dist_m)
        & (dy > 0.0)
        & (np.abs(np.degrees(np.arctan2(dx, dy))) <= cone_deg)
    )
    if not candidates.any():
        return None

    r_idx, c_idx = np.nonzero(candidates)
    dists = dist[r_idx, c_idx]
    abs_wx = np.abs(wx_all[c_idx])

    # Farthest frontier, ties by centreline (smallest |X|)
    order = np.lexsort((abs_wx, -dists))
    best = order[0]
    return int(c_idx[best]), int(r_idx[best])


def find_anchor_target(
    costmap: Costmap,
    anchor_pos_g: Sequence[float],
    start_cell: tuple[int, int],
    radius_m: float = DEFAULT_ANCHOR_SEARCH_RADIUS_M,
) -> Optional[tuple[int, int]]:
    """Find nearest reachable FREE cell to a lifted object anchor within radius_m.

    Args:
        costmap: Planning costmap.
        anchor_pos_g: [X, Y, Z] or [X, Y] in Ground frame G.
        start_cell: (col, row) of planner start.
        radius_m: Search radius in metres around anchor position.

    Returns:
        (col, row) of nearest reachable target cell, or None if unreachable.
    """
    ax = float(anchor_pos_g[0])
    ay = float(anchor_pos_g[1])

    reachable = get_reachable_cells(costmap, start_cell)
    if not reachable:
        return None

    cfg = costmap.cfg
    candidates: list[tuple[float, int, int]] = []

    for c, r in reachable:
        if costmap.grid[r, c] != CELL_FREE:
            continue

        wx, wy = cell_to_world(c, r, cfg)
        dist_to_anchor = math.hypot(wx - ax, wy - ay)
        if dist_to_anchor <= radius_m:
            candidates.append((dist_to_anchor, c, r))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return (candidates[0][1], candidates[0][2])


def find_multipath_candidates(
    costmap: Costmap,
    start_cell: tuple[int, int],
    sectors: Sequence[tuple[float, float]] = (
        (-60.0, -30.0),
        (-30.0, 0.0),
        (0.0, 30.0),
        (30.0, 60.0),
    ),
    min_dist_m: float = DEFAULT_MIN_EXPLORE_DIST_M,
    max_candidates: int = 3,
) -> list[PathResult]:
    """Find up to max_candidates paths across distinct angular sectors.

    For each 30-degree sector, picks the farthest reachable FREE cell and plans A*.
    Candidate paths are ranked by path length (shorter first), then clearance.

    Returns:
        List of PathResult instances sorted by preference.
    """
    reachable = get_reachable_cells(costmap, start_cell)
    if not reachable:
        return []

    cfg = costmap.cfg
    start_wx, start_wy = cell_to_world(start_cell[0], start_cell[1], cfg)
    candidate_paths: list[PathResult] = []

    for b_min, b_max in sectors:
        best_target: Optional[tuple[int, int]] = None
        best_dist = -1.0
        best_centerline = math.inf

        for c, r in reachable:
            if costmap.grid[r, c] != CELL_FREE:
                continue
            wx, wy = cell_to_world(c, r, cfg)
            dx = wx - start_wx
            dy = wy - start_wy
            dist = math.hypot(dx, dy)
            if dist < min_dist_m or dy <= 0.0:
                continue

            bearing_deg = math.degrees(math.atan2(dx, dy))
            if b_min <= bearing_deg < b_max:
                abs_wx = abs(wx)
                if dist > best_dist or (abs(dist - best_dist) < 1e-4 and abs_wx < best_centerline):
                    best_dist = dist
                    best_centerline = abs_wx
                    best_target = (c, r)

        if best_target is not None:
            res = astar_plan(costmap, start_cell, best_target)
            if res.status == "OK":
                candidate_paths.append(res)

    # Rank by length (shorter first), then clearance (larger mean clearance first)
    candidate_paths.sort(
        key=lambda p: (
            p.length_m,
            -np.mean(p.clearances) if p.clearances else 0.0,
        )
    )

    return candidate_paths[:max_candidates]
