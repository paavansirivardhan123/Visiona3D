"""Test M2: Bird's-Eye View (BEV) map from analytic scenes."""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.config import Intrinsics, MASK_FLOOR, MASK_OBSTACLE, MASK_WALL
from geometry.camera import backproject
from geometry.frames import camera_to_ground
from mapping.bev import build_bev
from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNCERTAIN,
    CELL_UNKNOWN,
    MapConfig,
    world_to_cell,
)
from tests.synthetic import render_scene


def test_m2_bev_floor_only_gives_free_ahead():
    """M2: Floor-only scene gives FREE ahead with no occupied cells."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    h = 1.4
    pitch = 15.0
    depth, floor_mask, wall_mask, box_masks, _ = render_scene(K, 240, 320, h, pitch, wall_distance=None)
    # Without a wall, rays above the horizon have no hit and render_scene encodes
    # them as depth 0.0. lift_frame marks depth <= 0 invalid (NaN in the per-pixel
    # point map), so the same convention is applied here: depth 0 pixels must not
    # backproject to the camera origin (which sits in the obstacle height band).
    depth = np.where(depth > 0.0, depth, np.nan)

    theta = math.radians(pitch)
    n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
    pts_g = camera_to_ground(backproject(depth, K), n, h)

    labels = np.zeros((240, 320), dtype=np.int64)
    labels[floor_mask] = MASK_FLOOR

    cfg = MapConfig()
    bev = build_bev(pts_g, labels, certified_final=floor_mask, cfg=cfg)

    # Ahead of camera (Y in [1.5, 5.0], X near 0) must have FREE cells
    col, row = world_to_cell(0.0, 2.5, cfg)
    assert bev.grid[row, col] == CELL_FREE
    # No occupied cells in floor-only scene
    assert np.sum(bev.grid == CELL_OCCUPIED) == 0


def test_m2_bev_wall_at_four_metres_gives_occupied_band():
    """M2: Wall at 4.0 m gives an OCCUPIED band within one cell of Y = 4 (row 45)."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    h = 1.4
    pitch = 15.0
    depth, floor_mask, wall_mask, box_masks, _ = render_scene(K, 240, 320, h, pitch, wall_distance=4.0)

    theta = math.radians(pitch)
    n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
    pts_g = camera_to_ground(backproject(depth, K), n, h)

    labels = np.zeros((240, 320), dtype=np.int64)
    labels[floor_mask] = 1
    labels[wall_mask] = 2

    cfg = MapConfig()
    bev = build_bev(pts_g, labels, certified_final=floor_mask, cfg=cfg)

    # In G, Y = 4.0 m corresponds to row = floor((4.0 - (-0.5)) / 0.10) = 45
    c_center, r_wall = world_to_cell(0.0, 4.0, cfg)
    assert r_wall == 45

    # Check that an occupied cell exists within 1 cell of row 45 (rows 44, 45, 46)
    wall_band_occupied = any(bev.grid[r, c_center] == CELL_OCCUPIED for r in (44, 45, 46))
    assert wall_band_occupied, f"Expected occupied cell near row 45, got rows: {[bev.grid[r, c_center] for r in (44, 45, 46)]}"

    # The pixel mask used for certified obstacle telemetry excludes floor.
    assert bev.obstacle_mask.shape == labels.shape
    assert bev.obstacle_mask[wall_mask].any()
    assert not bev.obstacle_mask[floor_mask].any()


def test_m2_bev_box_footprint_within_one_cell():
    """M2: Box at known G position gives occupied cells covering its footprint within one cell."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    h = 1.4
    pitch = 15.0
    box = (-0.5, 0.5, 2.5, 3.2, 0.0, 0.8)
    depth, floor_mask, wall_mask, box_masks, _ = render_scene(
        K, 240, 320, h, pitch, wall_distance=6.0, boxes=[box]
    )

    theta = math.radians(pitch)
    n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
    pts_g = camera_to_ground(backproject(depth, K), n, h)

    labels = np.zeros((240, 320), dtype=np.int64)
    labels[floor_mask] = 1
    labels[wall_mask] = 2
    labels[box_masks[0]] = 3

    cfg = MapConfig()
    bev = build_bev(pts_g, labels, certified_final=floor_mask, cfg=cfg)

    # Box center is at X = 0.0, Y = 2.85 m
    c_box, r_box = world_to_cell(0.0, 2.85, cfg)
    assert bev.grid[r_box, c_box] == CELL_OCCUPIED


def test_m2_bev_floating_box_overhang_is_occupied():
    """M2: A floating box at 1.0 to 1.5 m height is OCCUPIED (overhang caught by the height band).

    The box pixels are deliberately left UNLABELED (mask 0), simulating a
    segmentation miss, so the only rule that can mark the box occupied is the
    obstacle height band: non-certified points with Z in [0.15, 2.0] m.

    Hand computation: the camera sits at Z = 1.4, inside the box's Z slab
    [1.0, 1.5], so the only visible box surface is the front face at Y = 2.0.
    Its points (Z in [1.0, 1.5], non-certified, inside the height band) cast
    OCCUPIED votes at row = floor((2.0 - (-0.5)) / 0.10) = 25, spanning the box
    width X in [-0.4, 0.4] -> cols 26..33. The floor below the box stays visible
    through the gap under Z = 1.0 and is certified FREE; from a single view the
    footprint beyond the front face is unobservable, so only front-face cells
    can be OCCUPIED.
    """
    K = Intrinsics.from_fov(320, 240, 75.0)
    h = 1.4
    pitch = 15.0
    # Overhang: Z in [1.0, 1.5]
    overhang = (-0.4, 0.4, 2.0, 2.6, 1.0, 1.5)
    depth, floor_mask, wall_mask, box_masks, _ = render_scene(
        K, 240, 320, h, pitch, wall_distance=6.0, boxes=[overhang]
    )

    theta = math.radians(pitch)
    n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
    pts_g = camera_to_ground(backproject(depth, K), n, h)

    labels = np.zeros((240, 320), dtype=np.int64)
    labels[floor_mask] = MASK_FLOOR
    # Box pixels stay MASK_UNLABELED: the height band must catch them.

    cfg = MapConfig()
    bev = build_bev(pts_g, labels, certified_final=floor_mask, cfg=cfg)

    # Front face at Y = 2.0 -> row 25; centre column X = 0.0 -> col 30
    c_front, r_front = world_to_cell(0.0, 2.0, cfg)
    assert (c_front, r_front) == (30, 25)
    assert bev.grid[r_front, c_front] == CELL_OCCUPIED
    # Occupied band spans the box width X in [-0.4, 0.4) -> cols 26..33
    for c in range(26, 34):
        assert bev.grid[r_front, c] == CELL_OCCUPIED, f"col {c} at the overhang front face must be OCCUPIED"


def test_m2_cell_with_both_floor_and_obstacle_is_occupied():
    """M2: A cell with both floor and obstacle votes is OCCUPIED."""
    cfg = MapConfig()
    # Construct synthetic points falling into the exact same cell (col 30, row 20)
    wx, wy = 0.05, 1.55  # Maps to col 30, row 20
    # 10 floor points (Z = 0) and 2 obstacle points (Z = 0.5)
    floor_pts = np.tile([wx, wy, 0.0], (10, 1))
    obs_pts = np.tile([wx, wy, 0.5], (2, 1))
    pts_g = np.vstack([floor_pts, obs_pts])

    labels = np.array([MASK_FLOOR] * 10 + [MASK_OBSTACLE] * 2)
    certified = np.array([True] * 10 + [False] * 2)

    bev = build_bev(pts_g, labels, certified_final=certified, cfg=cfg)
    assert bev.free_votes[20, 30] == 10
    assert bev.occupied_votes[20, 30] == 2
    # Safety Principle: occupied wins!
    assert bev.grid[20, 30] == CELL_OCCUPIED
