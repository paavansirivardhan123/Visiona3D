"""Test M7: Path smoothing, collision re-validation, and fallback."""

from __future__ import annotations

import math
import numpy as np
import pytest

from mapping.grid import CELL_FREE, CELL_OCCUPIED, MapConfig, world_to_cell
from planning.astar import astar_plan
from planning.costmap import build_costmap
from planning.smooth import smooth_path
from planning.targets import find_explore_target


def test_m7_collision_fallback_fires_on_constructed_case():
    """M7: The collision fallback fires and preserves raw path on a constructed corner-cutting case."""
    cfg = MapConfig(res=0.10, x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0)
    grid = np.full((cfg.n_rows, cfg.n_cols), CELL_FREE, dtype=np.uint8)

    # Place obstacle at inner corner: X in [0.5, 2.5], Y in [0.5, 2.5]
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            wx = c * 0.10
            wy = r * 0.10
            if 0.5 <= wx <= 2.5 and 0.5 <= wy <= 2.5:
                grid[r, c] = CELL_OCCUPIED

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.0, safety_margin_m=0.0)

    # Raw L-shaped path skirting the corner:
    # (0.2, 0.2) -> (0.2, 3.0) -> (3.0, 3.0)
    raw_path = [
        (0.2, 0.2),
        (0.2, 1.5),
        (0.2, 3.0),
        (1.5, 3.0),
        (3.0, 3.0),
    ]

    smoothed, fell_back = smooth_path(raw_path, costmap, iterations=5, weight_smooth=0.6)
    assert fell_back is True, "Smoothing through inner obstacle corner must trigger fallback"
    assert smoothed == raw_path, "Smoothed path must match raw path when falling back"


def test_m7_smoothing_never_collides_over_200_random_maps():
    """M7: Smoothing never introduces a collision over 200 random maps; length ratio bounded."""
    rng = np.random.default_rng(12345)
    cfg = MapConfig(res=0.10, x_min=-2.0, x_max=2.0, y_min=0.0, y_max=4.0)

    tested_count = 0

    for i in range(200):
        # Generate random obstacle grid
        grid = np.full((cfg.n_rows, cfg.n_cols), CELL_FREE, dtype=np.uint8)
        # Random boxes
        n_obs = rng.integers(1, 4)
        for _ in range(n_obs):
            bx = rng.uniform(-1.2, 1.2)
            by = rng.uniform(1.0, 3.0)
            c0, r0 = world_to_cell(bx - 0.2, by - 0.2, cfg)
            c1, r1 = world_to_cell(bx + 0.2, by + 0.2, cfg)
            grid[max(r0, 0) : min(r1 + 1, cfg.n_rows), max(c0, 0) : min(c1 + 1, cfg.n_cols)] = CELL_OCCUPIED

        costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.10, safety_margin_m=0.0)
        start_c, start_r = world_to_cell(0.0, 0.2, cfg)
        if not costmap.is_passable(start_r, start_c):
            continue

        target = find_explore_target(costmap, (start_c, start_r), min_dist_m=1.0)
        if target is None:
            continue

        path_res = astar_plan(costmap, (start_c, start_r), target)
        if path_res.status != "OK" or len(path_res.path_world) < 3:
            continue

        smoothed_path, fell_back = smooth_path(path_res.path_world, costmap)

        # Check endpoints kept
        assert math.isclose(smoothed_path[0][0], path_res.path_world[0][0], abs_tol=1e-5)
        assert math.isclose(smoothed_path[0][1], path_res.path_world[0][1], abs_tol=1e-5)
        assert math.isclose(smoothed_path[-1][0], path_res.path_world[-1][0], abs_tol=1e-5)
        assert math.isclose(smoothed_path[-1][1], path_res.path_world[-1][1], abs_tol=1e-5)

        # Re-verify no collisions along smoothed path
        step_m = 0.5 * cfg.res
        for seg_idx in range(len(smoothed_path) - 1):
            p1 = smoothed_path[seg_idx]
            p2 = smoothed_path[seg_idx + 1]
            seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            n_sub = max(int(math.ceil(seg_dist / step_m)), 1)
            for s in range(n_sub + 1):
                px = p1[0] + (s / n_sub) * (p2[0] - p1[0])
                py = p1[1] + (s / n_sub) * (p2[1] - p1[1])
                c_p, r_p = world_to_cell(px, py, cfg)
                assert costmap.is_passable(r_p, c_p), f"Trial {i}: collision on smoothed path at ({px:.2f}, {py:.2f})"

        # Check length ratio bounded
        raw_len = path_res.length_m
        smooth_len = sum(
            math.hypot(smoothed_path[j][0] - smoothed_path[j - 1][0], smoothed_path[j][1] - smoothed_path[j - 1][1])
            for j in range(1, len(smoothed_path))
        )
        ratio = smooth_len / max(raw_len, 1e-4)
        assert 0.7 <= ratio <= 1.2, f"Length ratio {ratio:.2f} out of bounds"

        tested_count += 1

    assert tested_count >= 50, f"Expected at least 50 valid test paths, got {tested_count}"
