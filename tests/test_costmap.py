"""Test M4: Costmap inflation and chamfer distance transform validation."""

from __future__ import annotations

import numpy as np
import pytest

from mapping.grid import CELL_FREE, CELL_OCCUPIED, MapConfig, cell_to_world, world_to_cell
from planning.costmap import (
    brute_force_chamfer_distance,
    build_costmap,
    chamfer_distance_transform,
)


def test_m4_inflation_hand_verified_corridor_widths():
    """M4: A 0.9m corridor is blocked at half-width 0.35 + margin 0.10, and passable at 0.30 + 0.05."""
    cfg = MapConfig(res=0.05, x_min=-2.0, x_max=2.0, y_min=0.0, y_max=4.0)

    # Construct a 0.9m corridor:
    # Corridor interior: X in (-0.45, 0.45). Total width = 0.90 m.
    # Left wall: X <= -0.45 m. Right wall: X >= 0.45 m.
    grid = np.full((cfg.n_rows, cfg.n_cols), CELL_FREE, dtype=np.uint8)
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            wx, wy = cell_to_world(c, r, cfg)
            if wx <= -0.45 or wx >= 0.45:
                grid[r, c] = CELL_OCCUPIED

    # Case 1: user half-width 0.35 m + margin 0.10 m = 0.45 m inflation radius
    # Distance from centerline (X=0) to wall is 0.45 m.
    # Because clearance <= 0.45 m, the entire corridor is blocked!
    costmap_blocked = build_costmap(
        grid,
        cfg=cfg,
        user_half_width_m=0.35,
        safety_margin_m=0.10,
    )
    center_c, _ = world_to_cell(0.0, 2.0, cfg)
    for r in range(cfg.n_rows):
        assert not costmap_blocked.is_passable(r, center_c), f"Row {r} at centerline should be blocked"

    # Case 2: user half-width 0.30 m + margin 0.05 m = 0.35 m inflation radius
    # Distance from centerline (X=0) to wall is 0.45 m > 0.35 m.
    # Corridor is passable along the centerline!
    costmap_passable = build_costmap(
        grid,
        cfg=cfg,
        user_half_width_m=0.30,
        safety_margin_m=0.05,
    )
    centerline_passable_count = sum(costmap_passable.is_passable(r, center_c) for r in range(cfg.n_rows))
    assert centerline_passable_count == cfg.n_rows, f"Centerline should be fully passable, got {centerline_passable_count}/{cfg.n_rows}"


def test_m4_chamfer_matches_brute_force_on_random_maps():
    """M4: Chamfer distance transform matches brute-force distance exactly on random maps."""
    rng = np.random.default_rng(42)

    for trial in range(5):
        # Generate random 25x25 grid with 10% obstacle probability
        blocked = rng.random((25, 25)) < 0.10
        # Ensure at least one obstacle
        blocked[rng.integers(0, 25), rng.integers(0, 25)] = True

        res = 0.10
        chamfer_dist = chamfer_distance_transform(blocked, res=res)
        brute_dist = brute_force_chamfer_distance(blocked, res=res)

        max_err = float(np.max(np.abs(chamfer_dist - brute_dist)))
        assert max_err < 1e-6, f"Trial {trial}: Chamfer transform discrepancy {max_err:.2e} m"
