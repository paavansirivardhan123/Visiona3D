"""Rolling 2D map with exponential temporal decay and odometry warping.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.

LIMITATION:
Temporal fusion is validated ONLY on synthetic motion.
The source of odometry is NOT part of this work.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Union

import numpy as np

from mapping.bev import BevGrid
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

DEFAULT_HALF_LIFE_S: float = 1.5  # TO TUNE ON REAL DATA
DEFAULT_OCC_SCORE_THRESHOLD: float = 1.0  # TO TUNE ON REAL DATA
DEFAULT_FREE_SCORE_THRESHOLD: float = 5.0  # TO TUNE ON REAL DATA (matches min_points_per_cell)


@dataclass(frozen=True)
class Odometry2D:
    """Planar ego-motion in Ground frame G.

    Attributes:
        dx: Forward/right translation along X in metres.
        dy: Forward translation along Y in metres.
        dyaw: Heading change in radians (or degrees if dyaw_deg specified).
    """

    dx: float = 0.0
    dy: float = 0.0
    dyaw: float = 0.0
    dyaw_deg: Optional[float] = None

    def get_yaw_rad(self) -> float:
        if self.dyaw_deg is not None:
            return math.radians(self.dyaw_deg)
        return float(self.dyaw)


class RollingMap:
    """Rolling temporal BEV map with exponential decay and 2D odometry updates."""

    def __init__(
        self,
        cfg: MapConfig | None = None,
        half_life_s: float = DEFAULT_HALF_LIFE_S,
        occ_threshold: float = DEFAULT_OCC_SCORE_THRESHOLD,
        free_threshold: float = DEFAULT_FREE_SCORE_THRESHOLD,
    ) -> None:
        self.cfg = cfg if cfg is not None else MapConfig()
        self.half_life_s = float(half_life_s)
        self.occ_threshold = float(occ_threshold)
        self.free_threshold = float(free_threshold)

        n_rows, n_cols = self.cfg.n_rows, self.cfg.n_cols
        self.score_occ = np.zeros((n_rows, n_cols), dtype=np.float64)
        self.score_free = np.zeros((n_rows, n_cols), dtype=np.float64)
        self.score_unc = np.zeros((n_rows, n_cols), dtype=np.float64)
        self.grid = np.full((n_rows, n_cols), CELL_UNKNOWN, dtype=np.uint8)

    def _warp(self, grid_layer: np.ndarray, odom: Odometry2D) -> np.ndarray:
        """Warp an evidence layer under integer-cell translation and nearest-neighbour rotation."""
        dx = float(odom.dx)
        dy = float(odom.dy)
        yaw = odom.get_yaw_rad()

        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(yaw) < 1e-6:
            return grid_layer.copy()

        n_rows, n_cols = self.cfg.n_rows, self.cfg.n_cols
        warped = np.zeros_like(grid_layer)

        # Vectorized grid centers in new frame
        c_idx, r_idx = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        x_new, y_new = cell_to_world(c_idx, r_idx, self.cfg)

        # Invert ego-motion: p_old = R(-yaw) * p_new + [dx, dy].
        # A +yaw heading change (positive = clockwise, X right / Y forward) turns
        # the robot, so prior evidence at p_old appears in the new frame at
        # p_new = R(+yaw) * (p_old - d); inverting gives the pull-back below.
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x_old = cos_y * x_new + sin_y * y_new + dx
        y_old = -sin_y * x_new + cos_y * y_new + dy

        c_old, r_old = world_to_cell(x_old, y_old, self.cfg)
        valid = in_bounds(c_old, r_old, self.cfg)

        warped[r_idx[valid], c_idx[valid]] = grid_layer[r_old[valid], c_old[valid]]
        return warped

    def update(
        self,
        bev: Union[BevGrid, dict[str, np.ndarray]],
        odometry: Optional[Odometry2D] = None,
        dt: float = 0.0,
    ) -> np.ndarray:
        """Update rolling map with new observation, odometry ego-motion and elapsed time.

        Args:
            bev: BevGrid or dictionary of vote arrays ('occupied', 'free', 'uncertain').
            odometry: Optional Odometry2D ego-motion since last update.
            dt: Elapsed time in seconds since last update.

        Returns:
            Resolved cell states grid (n_rows, n_cols) with values in {0, 1, 2, 3}.
        """
        # 1. Exponential decay by half-life: decay = 0.5 ** (dt / half_life)
        if dt > 0.0 and self.half_life_s > 0.0:
            decay = 0.5 ** (float(dt) / self.half_life_s)
            self.score_occ *= decay
            self.score_free *= decay
            self.score_unc *= decay

        # 2. Warp prior evidence with odometry
        if odometry is not None:
            self.score_occ = self._warp(self.score_occ, odometry)
            self.score_free = self._warp(self.score_free, odometry)
            self.score_unc = self._warp(self.score_unc, odometry)

        # 3. Accumulate new votes
        if isinstance(bev, BevGrid):
            new_occ = bev.occupied_votes
            new_free = bev.free_votes
            new_unc = bev.uncertain_votes
        elif isinstance(bev, dict):
            new_occ = bev.get("occupied", 0)
            new_free = bev.get("free", 0)
            new_unc = bev.get("uncertain", 0)
        else:
            raise TypeError(f"Expected BevGrid or dict, got {type(bev)}")

        self.score_occ += np.asarray(new_occ, dtype=np.float64)
        self.score_free += np.asarray(new_free, dtype=np.float64)
        self.score_unc += np.asarray(new_unc, dtype=np.float64)

        # 4. Conservative state rule:
        # Occupied score above threshold wins, then free score, then uncertain, else unknown.
        resolved = np.full((self.cfg.n_rows, self.cfg.n_cols), CELL_UNKNOWN, dtype=np.uint8)

        is_occ = self.score_occ >= self.occ_threshold
        is_free = (~is_occ) & (self.score_free >= self.free_threshold)
        is_unc = (~is_occ) & (~is_free) & ((self.score_unc > 0.0) | (self.score_free > 0.0))

        resolved[is_occ] = CELL_OCCUPIED
        resolved[is_free] = CELL_FREE
        resolved[is_unc] = CELL_UNCERTAIN

        self.grid = resolved
        return self.grid
