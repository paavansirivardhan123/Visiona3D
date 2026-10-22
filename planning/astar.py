"""8-connected A* path planner with octile heuristic and clearance-cost weighting.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from mapping.grid import MapConfig, cell_to_world
from planning.costmap import Costmap

SQRT2 = math.sqrt(2.0)

# Movement offsets: (dr, dc, step_distance_units)
# 4 cardinal moves (step 1.0) and 4 diagonal moves (step sqrt(2))
NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, -1, SQRT2),
    (1, 1, SQRT2),
]


@dataclass
class PathResult:
    """Outcome of A* path planning."""

    status: str  # "OK", "NO_PATH", "START_BLOCKED"
    path: list[tuple[int, int]]  # List of (col, row) cells from start to goal
    path_world: list[tuple[float, float]]  # List of (X, Y) positions in ground frame G
    cost: float  # Cumulative A* path cost
    length_m: float  # Metric Euclidean / octile path length in metres
    clearances: list[float]  # Clearance at each cell in metres
    min_clearance_m: float  # Minimum clearance along the path in metres


def octile_heuristic(dr: int, dc: int, res: float) -> float:
    """Admissible octile heuristic in metres for 8-connected grid."""
    adr = abs(dr)
    adc = abs(dc)
    return (max(adr, adc) + (SQRT2 - 1.0) * min(adr, adc)) * res


class AStarPlanner:
    """8-connected A* planner on Costmap."""

    def __init__(self, allow_corner_cutting: bool = False) -> None:
        self.allow_corner_cutting = allow_corner_cutting

    def plan(
        self,
        costmap: Costmap,
        start_cell: tuple[int, int],  # (col, row)
        goal_cell: tuple[int, int],  # (col, row)
    ) -> PathResult:
        """Find optimal path from start_cell to goal_cell on costmap.

        Never raises an exception; failures return status codes.
        """
        start_col, start_row = int(start_cell[0]), int(start_cell[1])
        goal_col, goal_row = int(goal_cell[0]), int(goal_cell[1])
        cfg = costmap.cfg
        res = cfg.res

        # 1. Validate start and goal within grid bounds
        if not (0 <= start_col < cfg.n_cols and 0 <= start_row < cfg.n_rows):
            return PathResult("START_BLOCKED", [], [], math.inf, 0.0, [], 0.0)
        if not (0 <= goal_col < cfg.n_cols and 0 <= goal_row < cfg.n_rows):
            return PathResult("NO_PATH", [], [], math.inf, 0.0, [], 0.0)

        # 2. Safety check: start cell must be passable
        if not costmap.is_passable(start_row, start_col):
            return PathResult("START_BLOCKED", [], [], math.inf, 0.0, [], 0.0)

        # 3. Goal cell must be passable
        if not costmap.is_passable(goal_row, goal_col):
            return PathResult("NO_PATH", [], [], math.inf, 0.0, [], 0.0)

        # Trivial path
        if start_col == goal_col and start_row == goal_row:
            clr = float(costmap.clearance_m[start_row, start_col])
            wx, wy = cell_to_world(start_col, start_row, cfg)
            return PathResult(
                status="OK",
                path=[(start_col, start_row)],
                path_world=[(wx, wy)],
                cost=0.0,
                length_m=0.0,
                clearances=[clr],
                min_clearance_m=clr,
            )

        # Priority queue entries: (f_score, g_score, row, col)
        start_h = octile_heuristic(goal_row - start_row, goal_col - start_col, res)
        open_set: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (start_h, 0.0, start_row, start_col))

        g_scores: dict[tuple[int, int], float] = {(start_row, start_col): 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        visited: set[tuple[int, int]] = set()

        # Hot-loop caches: plain-Python passable flags and the precomputed
        # clearance cost multiplier per cell (identical values to
        # Costmap.cell_cost_multiplier at passable cells).
        passable = (~costmap.inflated_blocked).tolist()
        clearance = costmap.clearance_m
        k_w = costmap.k_clearance
        c_max = costmap.c_max_m
        ratio = (c_max - clearance) / c_max
        mult_arr = np.where(clearance >= c_max, 1.0, 1.0 + k_w * ratio * ratio)
        mult = mult_arr.tolist()
        n_rows, n_cols = costmap.n_rows, costmap.n_cols

        goal_found = False

        while open_set:
            f, g, r, c = heapq.heappop(open_set)

            if (r, c) == (goal_row, goal_col):
                goal_found = True
                break

            if (r, c) in visited:
                continue
            visited.add((r, c))

            # Explore 8 neighbors
            for dr, dc, step_units in NEIGHBORS:
                nr, nc = r + dr, c + dc

                # Out of bounds
                if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                    continue

                # Must be passable
                if not passable[nr][nc]:
                    continue

                # Diagonal move: check for corner cutting
                if dr != 0 and dc != 0 and not self.allow_corner_cutting:
                    # Both orthogonal adjacent cells must be passable
                    if not passable[r][nc] or not passable[nr][c]:
                        continue

                # Cell traversal cost:
                # cost = step * (1 + k * ((c_max - clearance)/c_max)^2 for clearance < c_max)
                step_cost = step_units * res * mult[nr][nc]

                tentative_g = g + step_cost

                prev_g = g_scores.get((nr, nc), math.inf)
                if tentative_g < prev_g:
                    g_scores[(nr, nc)] = tentative_g
                    came_from[(nr, nc)] = (r, c)
                    h = octile_heuristic(goal_row - nr, goal_col - nc, res)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, nr, nc))

        if not goal_found:
            return PathResult("NO_PATH", [], [], math.inf, 0.0, [], 0.0)

        # Reconstruct path
        path_cells: list[tuple[int, int]] = []
        curr = (goal_row, goal_col)
        while curr in came_from:
            path_cells.append((curr[1], curr[0]))  # (col, row)
            curr = came_from[curr]
        path_cells.append((start_col, start_row))
        path_cells.reverse()

        # Compute metric length, world coords and clearances
        path_world: list[tuple[float, float]] = []
        clearances: list[float] = []
        total_length_m = 0.0

        for i, (c_idx, r_idx) in enumerate(path_cells):
            wx, wy = cell_to_world(c_idx, r_idx, cfg)
            path_world.append((wx, wy))
            clearances.append(float(costmap.clearance_m[r_idx, c_idx]))
            if i > 0:
                prev_c, prev_r = path_cells[i - 1]
                dc_step = abs(c_idx - prev_c)
                dr_step = abs(r_idx - prev_r)
                step_len = SQRT2 * res if (dc_step > 0 and dr_step > 0) else res
                total_length_m += step_len

        min_clr = min(clearances) if clearances else 0.0
        return PathResult(
            status="OK",
            path=path_cells,
            path_world=path_world,
            cost=float(g_scores[(goal_row, goal_col)]),
            length_m=float(total_length_m),
            clearances=clearances,
            min_clearance_m=float(min_clr),
        )


def astar_plan(
    costmap: Costmap,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    allow_corner_cutting: bool = False,
) -> PathResult:
    """Convenience functional wrapper for AStarPlanner."""
    planner = AStarPlanner(allow_corner_cutting=allow_corner_cutting)
    return planner.plan(costmap, start_cell, goal_cell)
