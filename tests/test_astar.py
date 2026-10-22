"""Tests M5 and M3: A* path planner, corner cutting prevention, and certified-only space."""

from __future__ import annotations

import math
import numpy as np
import pytest

from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNCERTAIN,
    CELL_UNKNOWN,
    MapConfig,
)
from planning.astar import astar_plan, AStarPlanner, octile_heuristic
from planning.costmap import build_costmap


def test_m5_open_field_hand_computed_octile_length():
    """M5: Open field gives exact hand-computed octile path length."""
    cfg = MapConfig(res=1.0, x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
    grid = np.full((10, 10), CELL_FREE, dtype=np.uint8)
    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0, c_max_m=0.1)

    start = (1, 1)  # (col, row)
    goal = (4, 5)   # dc = 3, dr = 4
    res = astar_plan(costmap, start, goal)

    assert res.status == "OK"
    # Hand-computed octile length: 3 * sqrt(2) + (4 - 3) * 1.0 = 3*1.41421356 + 1.0 = 5.242640687
    expected_length = 3.0 * math.sqrt(2.0) + 1.0
    assert math.isclose(res.length_m, expected_length, rel_tol=1e-5)


def test_m5_wall_with_gap():
    """M5: Planner navigates through a gap in a wall."""
    cfg = MapConfig(res=1.0, x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
    grid = np.full((10, 10), CELL_FREE, dtype=np.uint8)
    # Wall at row 5 from col 0 to 9 except gap at col 5
    for c in range(10):
        if c != 5:
            grid[5, c] = CELL_OCCUPIED

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    start = (2, 2)
    goal = (2, 8)
    res = astar_plan(costmap, start, goal)

    assert res.status == "OK"
    assert (5, 5) in res.path, "Path must pass through gap at (5, 5)"


def test_m5_u_shaped_trap():
    """M5: Planner escapes a U-shaped local minimum obstacle."""
    cfg = MapConfig(res=1.0, x_min=0.0, x_max=12.0, y_min=0.0, y_max=12.0)
    grid = np.full((12, 12), CELL_FREE, dtype=np.uint8)
    # U-shaped cup open to the south (low row)
    # North wall: row 7, cols 3..8
    # West wall: rows 4..7, col 3
    # East wall: rows 4..7, col 8
    for c in range(3, 9):
        grid[7, c] = CELL_OCCUPIED
    for r in range(4, 8):
        grid[r, 3] = CELL_OCCUPIED
        grid[r, 8] = CELL_OCCUPIED

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    start = (5, 6)  # Deep inside U-trap
    goal = (5, 10)  # Beyond north wall
    res = astar_plan(costmap, start, goal)

    assert res.status == "OK"
    assert len(res.path) > 0


def test_m5_no_path_returns_status():
    """M5: Completely walled off goal returns NO_PATH status without raising."""
    cfg = MapConfig(res=1.0, x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
    grid = np.full((10, 10), CELL_FREE, dtype=np.uint8)
    # Wall dividing entire grid at row 5
    grid[5, :] = CELL_OCCUPIED

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    start = (2, 2)
    goal = (2, 8)
    res = astar_plan(costmap, start, goal)

    assert res.status == "NO_PATH"
    assert res.path == []
    assert math.isinf(res.cost)


def test_m5_blocked_start_returns_status():
    """M5: Blocked start returns START_BLOCKED status without raising."""
    cfg = MapConfig(res=1.0, x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
    grid = np.full((10, 10), CELL_FREE, dtype=np.uint8)
    grid[2, 2] = CELL_OCCUPIED

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    start = (2, 2)
    goal = (8, 8)
    res = astar_plan(costmap, start, goal)

    assert res.status == "START_BLOCKED"
    assert res.path == []


def test_m5_no_diagonal_corner_cutting():
    """M5: Diagonal moves cannot cut through orthogonal blocked corners."""
    cfg = MapConfig(res=1.0, x_min=0.0, x_max=4.0, y_min=0.0, y_max=4.0)
    grid = np.full((4, 4), CELL_FREE, dtype=np.uint8)
    # Pinch point: (1, 1) and (2, 2) are FREE, but (1, 2) and (2, 1) are OCCUPIED
    grid[1, 2] = CELL_OCCUPIED
    grid[2, 1] = CELL_OCCUPIED

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    start = (1, 1)
    goal = (2, 2)

    # With corner cutting disabled (default), cannot step directly (1,1) -> (2,2)
    res = astar_plan(costmap, start, goal, allow_corner_cutting=False)
    if res.status == "OK":
        # Must detour, never direct diagonal step (1,1) -> (2,2)
        steps = list(zip(res.path[:-1], res.path[1:]))
        assert ((1, 1), (2, 2)) not in steps


def test_m3_certified_only_never_uses_unknown_or_uncertain():
    """M3: UNKNOWN and UNCERTAIN cells are never used even when a path through them would be shorter."""
    cfg = MapConfig(res=1.0, x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
    grid = np.full((10, 10), CELL_OCCUPIED, dtype=np.uint8)

    # Certified C-shaped detour
    # Direct path from (1, 5) to (5, 5) is made of UNKNOWN and UNCERTAIN cells
    for c in range(1, 6):
        grid[5, c] = CELL_UNKNOWN if c % 2 == 0 else CELL_UNCERTAIN

    # Certified detour path along row 2:
    for r in range(2, 6):
        grid[r, 1] = CELL_FREE
        grid[r, 5] = CELL_FREE
    for c in range(1, 6):
        grid[2, c] = CELL_FREE

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    start = (1, 5)
    goal = (5, 5)

    res = astar_plan(costmap, start, goal)
    assert res.status == "OK"

    # Verify that NO cell in the path is UNKNOWN or UNCERTAIN
    for c, r in res.path:
        cell_state = grid[r, c]
        assert cell_state == CELL_FREE, f"Cell ({c}, {r}) has uncertified state {cell_state}"
