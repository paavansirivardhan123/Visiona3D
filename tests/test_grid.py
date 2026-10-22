"""Test M1: Grid coordinate conversion, round-trip and boundary handling."""

from __future__ import annotations

import numpy as np
import pytest

from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNCERTAIN,
    CELL_UNKNOWN,
    MapConfig,
    cell_to_world,
    in_bounds,
    world_to_cell,
)


def test_m1_grid_hand_computed_points():
    """M1: Hand-computed cells for 6 points including boundaries and out-of-bounds."""
    cfg = MapConfig(res=0.10, x_min=-3.0, x_max=3.0, y_min=-0.5, y_max=7.0)
    assert cfg.n_cols == 60
    assert cfg.n_rows == 75

    # 1. Origin (0.0, 0.0) -> col=30, row=5
    c1, r1 = world_to_cell(0.0, 0.0, cfg)
    assert (c1, r1) == (30, 5)
    assert in_bounds(c1, r1, cfg) is True

    # 2. Min boundary (-3.0, -0.5) -> col=0, row=0
    c2, r2 = world_to_cell(-3.0, -0.5, cfg)
    assert (c2, r2) == (0, 0)
    assert in_bounds(c2, r2, cfg) is True

    # 3. Max boundary inside (2.99, 6.99) -> col=59, row=74
    c3, r3 = world_to_cell(2.99, 6.99, cfg)
    assert (c3, r3) == (59, 74)
    assert in_bounds(c3, r3, cfg) is True

    # 4. Out of bounds left (-3.05, 1.0) -> col=-1, row=15
    c4, r4 = world_to_cell(-3.05, 1.0, cfg)
    assert (c4, r4) == (-1, 15)
    assert in_bounds(c4, r4, cfg) is False

    # 5. Out of bounds top (0.0, 7.05) -> col=30, row=75
    c5, r5 = world_to_cell(0.0, 7.05, cfg)
    assert (c5, r5) == (30, 75)
    assert in_bounds(c5, r5, cfg) is False

    # 6. Interior point (1.23, 3.47) -> col=42, row=39
    c6, r6 = world_to_cell(1.23, 3.47, cfg)
    assert (c6, r6) == (42, 39)
    assert in_bounds(c6, r6, cfg) is True


def test_m1_grid_round_trip():
    """M1: cell_to_world followed by world_to_cell round-trips exactly."""
    cfg = MapConfig(res=0.10, x_min=-3.0, x_max=3.0, y_min=-0.5, y_max=7.0)

    for c in [0, 10, 25, 45, 59]:
        for r in [0, 15, 30, 50, 74]:
            wx, wy = cell_to_world(c, r, cfg)
            c_rec, r_rec = world_to_cell(wx, wy, cfg)
            assert (c_rec, r_rec) == (c, r)


def test_m1_cell_states_distinct():
    """M1: Cell state codes distinct from semantic mask indices."""
    assert CELL_UNKNOWN == 0
    assert CELL_FREE == 1
    assert CELL_OCCUPIED == 2
    assert CELL_UNCERTAIN == 3
    states = {CELL_UNKNOWN, CELL_FREE, CELL_OCCUPIED, CELL_UNCERTAIN}
    assert len(states) == 4
