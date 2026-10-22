"""Bird's-Eye View (BEV) map generation in Ground Frame G.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from core.config import MASK_FLOOR, MASK_OBSTACLE, MASK_WALL
from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNCERTAIN,
    CELL_UNKNOWN,
    MapConfig,
    in_bounds,
    world_to_cell,
)


@dataclass
class BevGrid:
    """Bird's-eye view 2D occupancy grid and evidence accumulator."""

    grid: np.ndarray  # (n_rows, n_cols) uint8 cell states
    free_votes: np.ndarray  # (n_rows, n_cols) int32
    occupied_votes: np.ndarray  # (n_rows, n_cols) int32
    uncertain_votes: np.ndarray  # (n_rows, n_cols) int32
    cfg: MapConfig


def build_bev(
    pointmap_g: np.ndarray,
    labels: np.ndarray,
    certified_final: np.ndarray | None = None,
    vetoed: np.ndarray | None = None,
    pessimistic_pointmap: np.ndarray | None = None,
    cfg: MapConfig | None = None,
) -> BevGrid:
    """Build bird's-eye view occupancy grid from 3D ground-frame points and masks.

    Rules (enforced in order):
    1. OCCUPIED votes from:
       - non-floor labels (wall, obstacle),
       - vetoed floor labels,
       - any non-certified point inside the obstacle height band [z_band_min, z_band_max].
         This catches overhangs.
    2. FREE votes from certified_final floor points with |Z| <= tau.
    3. The pessimistic layer adds OCCUPIED votes at lower-bound positions.
    4. A cell with both free and occupied votes is OCCUPIED.
    5. FREE needs at least min_points_per_cell votes.
    6. Floor-labelled cells without enough certified votes are UNCERTAIN.
    7. Cells with no votes are UNKNOWN.

    Args:
        pointmap_g: (H, W, 3) or (N, 3) ground frame points in metres.
        labels: (H, W) or (N,) class labels using core.config mask indices.
        certified_final: Boolean mask of certified floor points.
        vetoed: Boolean mask of geometric-vetoed floor points.
        pessimistic_pointmap: Optional (H, W, 3) lower-bound points.
        cfg: MapConfig instance (defaults if None).

    Returns:
        BevGrid containing cell states and vote counts.
    """
    if cfg is None:
        cfg = MapConfig()

    pts = np.asarray(pointmap_g, dtype=np.float64)
    lbls = np.asarray(labels, dtype=np.int64)

    if pts.ndim == 3 and pts.shape[2] >= 3:
        pts = pts.reshape(-1, pts.shape[2])
    elif pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"pointmap_g must be (H, W, 3) or (N, 3), got {pts.shape}")

    n_pts = pts.shape[0]
    lbls_flat = lbls.reshape(-1)
    if lbls_flat.size != n_pts:
        raise ValueError(f"labels size {lbls_flat.size} != points length {n_pts}")

    if certified_final is not None:
        cert_flat = np.asarray(certified_final, dtype=bool).reshape(-1)
        if cert_flat.size != n_pts:
            raise ValueError("certified_final shape mismatch")
    else:
        cert_flat = np.zeros(n_pts, dtype=bool)

    if vetoed is not None:
        veto_flat = np.asarray(vetoed, dtype=bool).reshape(-1)
        if veto_flat.size != n_pts:
            raise ValueError("vetoed shape mismatch")
    else:
        veto_flat = np.zeros(n_pts, dtype=bool)

    n_rows, n_cols = cfg.n_rows, cfg.n_cols
    free_votes = np.zeros((n_rows, n_cols), dtype=np.int32)
    occ_votes = np.zeros((n_rows, n_cols), dtype=np.int32)
    unc_votes = np.zeros((n_rows, n_cols), dtype=np.int32)

    # 1. Nominal points classification
    valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2])
    pts_val = pts[valid]
    lbls_val = lbls_flat[valid]
    cert_val = cert_flat[valid]
    veto_val = veto_flat[valid]

    if pts_val.size > 0:
        c, r = world_to_cell(pts_val[:, 0], pts_val[:, 1], cfg)
        b_mask = in_bounds(c, r, cfg)

        pts_val = pts_val[b_mask]
        lbls_val = lbls_val[b_mask]
        cert_val = cert_val[b_mask]
        veto_val = veto_val[b_mask]
        c = c[b_mask]
        r = r[b_mask]

        z_val = pts_val[:, 2]

        is_non_floor = (lbls_val == MASK_WALL) | (lbls_val == MASK_OBSTACLE)
        is_vetoed = veto_val
        in_height_band = (z_val >= cfg.z_band_min) & (z_val <= cfg.z_band_max) & (~cert_val)

        is_occupied = is_non_floor | is_vetoed | in_height_band
        is_free = cert_val & (np.abs(z_val) <= cfg.tau_floor) & (~is_occupied)
        is_uncertain = (lbls_val == MASK_FLOOR) & (~cert_val) & (~is_occupied) & (~is_free)

        # Accumulate votes (bincount over flattened cell indices; exact integer counts)
        flat_idx = r.astype(np.int64) * n_cols + c
        occ_votes = np.bincount(flat_idx[is_occupied], minlength=n_rows * n_cols).reshape(n_rows, n_cols).astype(np.int32)
        free_votes = np.bincount(flat_idx[is_free], minlength=n_rows * n_cols).reshape(n_rows, n_cols).astype(np.int32)
        unc_votes = np.bincount(flat_idx[is_uncertain], minlength=n_rows * n_cols).reshape(n_rows, n_cols).astype(np.int32)

    # 2. Pessimistic layer adds OCCUPIED votes at lower-bound positions
    if pessimistic_pointmap is not None:
        p_pts = np.asarray(pessimistic_pointmap, dtype=np.float64)
        if p_pts.ndim == 3 and p_pts.shape[2] >= 3:
            p_pts = p_pts.reshape(-1, p_pts.shape[2])
        if p_pts.shape[0] == n_pts:
            p_valid = np.isfinite(p_pts[:, 0]) & np.isfinite(p_pts[:, 1]) & np.isfinite(p_pts[:, 2])
            # Only obstacle / non-floor or height band points cast lower bound votes
            # (or all points flagged as obstacle)
            p_non_floor = (lbls_flat == MASK_WALL) | (lbls_flat == MASK_OBSTACLE) | veto_flat
            p_height = (p_pts[:, 2] >= cfg.z_band_min) & (p_pts[:, 2] <= cfg.z_band_max) & (~cert_flat)
            p_occ = (p_non_floor | p_height) & p_valid

            if np.any(p_occ):
                p_c, p_r = world_to_cell(p_pts[p_occ, 0], p_pts[p_occ, 1], cfg)
                p_b = in_bounds(p_c, p_r, cfg)
                p_flat = p_r[p_b].astype(np.int64) * n_cols + p_c[p_b]
                occ_votes += np.bincount(p_flat, minlength=n_rows * n_cols).reshape(n_rows, n_cols)

    # 3. Resolve cell states
    grid = np.full((n_rows, n_cols), CELL_UNKNOWN, dtype=np.uint8)

    # Rule: a cell with both free and occupied votes is OCCUPIED
    has_occ = occ_votes > 0
    grid[has_occ] = CELL_OCCUPIED

    # Free requires >= min_points_per_cell and no occupied votes
    has_free = (~has_occ) & (free_votes >= cfg.min_points_per_cell)
    grid[has_free] = CELL_FREE

    # Floor-labelled cells without enough certified votes are UNCERTAIN
    has_unc = (~has_occ) & (~has_free) & ((unc_votes > 0) | (free_votes > 0))
    grid[has_unc] = CELL_UNCERTAIN

    return BevGrid(
        grid=grid,
        free_votes=free_votes,
        occupied_votes=occ_votes,
        uncertain_votes=unc_votes,
        cfg=cfg,
    )
