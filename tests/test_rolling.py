"""Test M9: Rolling temporal map, odometry warping, and exponential decay."""

from __future__ import annotations

import math
import numpy as np
import pytest

from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    MapConfig,
    cell_to_world,
    world_to_cell,
)
from mapping.rolling import Odometry2D, RollingMap


def test_m9_odometry_forward_shift_ten_rows():
    """M9: (0, 1.0) m odometry shifts occupied cells by 10 rows."""
    cfg = MapConfig(res=0.10, x_min=-3.0, x_max=3.0, y_min=-0.5, y_max=7.0)
    rolling = RollingMap(cfg=cfg, half_life_s=10.0)

    # Place an occupied cell at (col 30, row 45) -> (0.05m, 4.05m)
    r_initial, c_initial = 45, 30
    rolling.score_occ[r_initial, c_initial] = 5.0
    rolling.update({"occupied": np.zeros_like(rolling.score_occ)}, dt=0.0)
    assert rolling.grid[r_initial, c_initial] == CELL_OCCUPIED

    # Robot moves forward by dy = 1.0 m (odometry: dx=0, dy=1.0)
    # The world obstacle moves 1.0 m closer to robot: Y_new = 4.05 - 1.0 = 3.05 m
    # row_new = 45 - 10 = 35
    odom = Odometry2D(dx=0.0, dy=1.0, dyaw=0.0)
    empty = {"occupied": np.zeros_like(rolling.score_occ)}
    grid_new = rolling.update(empty, odometry=odom, dt=0.0)

    assert grid_new[35, 30] == CELL_OCCUPIED
    assert grid_new[45, 30] != CELL_OCCUPIED


def test_m9_ninety_degree_yaw_rotation():
    """M9: 90-degree yaw rotates occupied cells correctly."""
    cfg = MapConfig(res=0.10, x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0)
    rolling = RollingMap(cfg=cfg, half_life_s=10.0)

    # Evidence cell containing (X=0.0, Y=2.0): (col 30, row 50), centre (0.05, 2.05)
    c0, r0 = world_to_cell(0.0, 2.0, cfg)
    assert (c0, r0) == (30, 50)
    rolling.score_occ[r0, c0] = 5.0
    rolling.update({"occupied": np.zeros_like(rolling.score_occ)}, dt=0.0)
    assert rolling.grid[r0, c0] == CELL_OCCUPIED

    # Robot turns 90 deg clockwise (+pi/2): a point ahead appears to the LEFT.
    # Nearest-neighbour cell rotation moves the evidence cell CENTRE:
    # (0.05, 2.05) -> (-2.05, 0.05), which lies in cell (col 9, row 30).
    odom = Odometry2D(dx=0.0, dy=0.0, dyaw_deg=90.0)
    empty = {"occupied": np.zeros_like(rolling.score_occ)}
    rolling.update(empty, odometry=odom, dt=0.0)

    c_rot, r_rot = world_to_cell(-2.05, 0.05, cfg)
    assert (c_rot, r_rot) == (9, 30)
    assert rolling.grid[r_rot, c_rot] == CELL_OCCUPIED
    # Exactly one occupied cell remains: the rotated copy
    assert int(np.sum(rolling.grid == CELL_OCCUPIED)) == 1
    assert rolling.grid[r0, c0] != CELL_OCCUPIED


def test_m9_decay_halves_scores_at_half_life():
    """M9: Decay halves scores at the half-life."""
    cfg = MapConfig(res=0.10)
    half_life = 1.5
    rolling = RollingMap(cfg=cfg, half_life_s=half_life)

    r, c = 20, 20
    rolling.score_occ[r, c] = 10.0
    rolling.score_free[r, c] = 8.0

    # Step dt = 1.5 s
    empty = {
        "occupied": np.zeros_like(rolling.score_occ),
        "free": np.zeros_like(rolling.score_free),
    }
    rolling.update(empty, dt=half_life)

    assert math.isclose(rolling.score_occ[r, c], 5.0, abs_tol=1e-5)
    assert math.isclose(rolling.score_free[r, c], 4.0, abs_tol=1e-5)


def test_m9_identity_odometry_keeps_grid():
    """M9: Identity odometry with dt=0 keeps grid and scores unchanged."""
    cfg = MapConfig(res=0.10)
    rolling = RollingMap(cfg=cfg, half_life_s=1.5)

    r, c = 25, 30
    rolling.score_occ[r, c] = 4.2
    grid_before = rolling.update({"occupied": np.zeros_like(rolling.score_occ)}, dt=0.0).copy()

    odom = Odometry2D(dx=0.0, dy=0.0, dyaw=0.0)
    grid_after = rolling.update({"occupied": np.zeros_like(rolling.score_occ)}, odometry=odom, dt=0.0)

    assert math.isclose(rolling.score_occ[r, c], 4.2, abs_tol=1e-5)
    assert np.array_equal(grid_before, grid_after)
