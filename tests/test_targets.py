"""Test M6: Target selection strategies for explore, anchor, and multi-path modes."""

from __future__ import annotations

import math
import numpy as np
import pytest

from mapping.grid import CELL_FREE, CELL_OCCUPIED, MapConfig, cell_to_world, world_to_cell
from planning.costmap import build_costmap
from planning.targets import (
    find_anchor_target,
    find_explore_target,
    find_multipath_candidates,
)


def test_m6_farthest_frontier_choice_and_tie_breaking():
    """M6: Explore mode picks farthest frontier in forward cone, breaking ties by centreline."""
    cfg = MapConfig(res=0.10, x_min=-3.0, x_max=3.0, y_min=0.0, y_max=6.0)
    grid = np.full((cfg.n_rows, cfg.n_cols), CELL_OCCUPIED, dtype=np.uint8)

    start_c, start_r = world_to_cell(0.0, 0.5, cfg)

    # Create certified free corridor
    for r in range(start_r, cfg.n_rows):
        for c in range(start_c - 15, start_c + 16):
            grid[r, c] = CELL_FREE

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)

    # 1. Standard: picks farthest reachable FREE cell
    target = find_explore_target(costmap, (start_c, start_r), cone_deg=60.0, min_dist_m=1.5)
    assert target is not None
    twx, twy = cell_to_world(target[0], target[1], cfg)
    # The farthest reachable row in grid is row 59 (near Y = 5.95)
    assert math.isclose(twy, 5.95, abs_tol=0.1)

    # 2. Tie breaking: two cells at identical distance from the start; the one
    # closer to the centreline (|X|) must win.
    # Free corridor along the centre column plus a horizontal free segment at
    # row test_r. Start centre is (0.05, 0.55); the two segment ends are at
    # (-0.55, 3.05) and (0.65, 3.05), both at distance hypot(0.60, 2.50) =
    # 2.5708 m -> an exact tie. Tie-break by centreline |X|: |-0.55| < |0.65|,
    # so the left end (col start_c - 4) wins.
    test_r = start_r + 25  # row 30, centre Y = 3.05 m
    test_grid = np.full((cfg.n_rows, cfg.n_cols), CELL_OCCUPIED, dtype=np.uint8)
    for r in range(start_r, test_r + 1):
        test_grid[r, start_c] = CELL_FREE  # Centreline corridor
    for c in range(start_c - 4, start_c + 5):
        test_grid[test_r, c] = CELL_FREE  # Horizontal segment: both ends equidistant

    cm = build_costmap(test_grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    chosen = find_explore_target(cm, (start_c, start_r), cone_deg=60.0, min_dist_m=1.5)
    assert chosen is not None
    assert chosen[1] == test_r
    # Closer to centreline (X = -0.55 m) must win the tie over X = 0.65 m
    assert chosen[0] == start_c - 4


def test_m6_anchor_mode_nearest_reachable_cell():
    """M6: Anchor mode picks nearest reachable FREE cell within radius."""
    cfg = MapConfig(res=0.10, x_min=-3.0, x_max=3.0, y_min=0.0, y_max=6.0)
    grid = np.full((cfg.n_rows, cfg.n_cols), CELL_FREE, dtype=np.uint8)
    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)

    start = world_to_cell(0.0, 0.5, cfg)
    anchor_pos = [1.2, 3.0, 0.5]  # Anchor in ground frame G

    target = find_anchor_target(costmap, anchor_pos, start, radius_m=1.5)
    assert target is not None
    tx, ty = cell_to_world(target[0], target[1], cfg)
    dist = math.hypot(tx - anchor_pos[0], ty - anchor_pos[1])
    assert dist <= 1.5


def test_m6_unreachable_goal_status():
    """M6: Unreachable anchor returns None."""
    cfg = MapConfig(res=0.10, x_min=-3.0, x_max=3.0, y_min=0.0, y_max=6.0)
    grid = np.full((cfg.n_rows, cfg.n_cols), CELL_OCCUPIED, dtype=np.uint8)

    start = world_to_cell(0.0, 0.5, cfg)
    grid[start[1], start[0]] = CELL_FREE

    # Anchor at (1.0, 4.0), surrounded completely by obstacles
    anchor_pos = [1.0, 4.0, 0.0]

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    target = find_anchor_target(costmap, anchor_pos, start, radius_m=1.0)
    assert target is None


def test_m6_two_gaps_give_two_candidate_paths_in_different_sectors():
    """M6: Wall with two gaps gives candidate paths in distinct angular sectors."""
    cfg = MapConfig(res=0.10, x_min=-3.0, x_max=3.0, y_min=0.0, y_max=6.0)
    grid = np.full((cfg.n_rows, cfg.n_cols), CELL_FREE, dtype=np.uint8)

    # Horizontal wall at row 30 (Y = 3.0)
    wall_r = 30
    grid[wall_r, :] = CELL_OCCUPIED

    # Left gap at X = -1.5 (bearing ~ -35 deg from (0, 1.0))
    # Right gap at X = +1.5 (bearing ~ +35 deg from (0, 1.0))
    c_left, _ = world_to_cell(-1.5, 3.0, cfg)
    c_right, _ = world_to_cell(1.5, 3.0, cfg)

    for dc in range(-2, 3):
        grid[wall_r, c_left + dc] = CELL_FREE
        grid[wall_r, c_right + dc] = CELL_FREE

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)
    start = world_to_cell(0.0, 1.0, cfg)

    candidates = find_multipath_candidates(
        costmap,
        start,
        sectors=((-60.0, -10.0), (10.0, 60.0)),
        min_dist_m=1.5,
        max_candidates=2,
    )

    assert len(candidates) == 2, f"Expected 2 candidate paths, got {len(candidates)}"
    # Verify paths go through different sectors (one left, one right)
    final_x_0 = candidates[0].path_world[-1][0]
    final_x_1 = candidates[1].path_world[-1][0]
    assert (final_x_0 * final_x_1) < 0, "Candidate paths should be in opposite horizontal sectors"
