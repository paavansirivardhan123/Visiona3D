"""Grid definitions and conversions for Visiona3D Ground Frame G.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.

Conventions:
- Grid in ground frame G.
  col = floor((X - x_min)/res), row = floor((Y - y_min)/res); row grows with forward distance.
- Cell state codes (distinct from mask class indices):
  CELL_UNKNOWN = 0
  CELL_FREE = 1 (certified walkable)
  CELL_OCCUPIED = 2
  CELL_UNCERTAIN = 3 (floor-labelled but not certified or vetoed)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Tuple, Union

import numpy as np

# Cell state codes (distinct from mask class indices)
CELL_UNKNOWN: int = 0
CELL_FREE: int = 1
CELL_OCCUPIED: int = 2
CELL_UNCERTAIN: int = 3

CELL_STATE_NAMES = {
    CELL_UNKNOWN: "UNKNOWN",
    CELL_FREE: "FREE",
    CELL_OCCUPIED: "OCCUPIED",
    CELL_UNCERTAIN: "UNCERTAIN",
}

# Defaults - labelled TO TUNE ON REAL DATA
DEFAULT_RES_M: float = 0.10  # TO TUNE ON REAL DATA
DEFAULT_X_MIN_M: float = -3.0  # TO TUNE ON REAL DATA
DEFAULT_X_MAX_M: float = 3.0  # TO TUNE ON REAL DATA
DEFAULT_Y_MIN_M: float = -0.5  # TO TUNE ON REAL DATA
DEFAULT_Y_MAX_M: float = 7.0  # TO TUNE ON REAL DATA
DEFAULT_Z_BAND_MIN_M: float = 0.15  # TO TUNE ON REAL DATA: height band lower bound
DEFAULT_Z_BAND_MAX_M: float = 2.0  # TO TUNE ON REAL DATA: height band upper bound
DEFAULT_MIN_POINTS_PER_CELL: int = 5  # TO TUNE ON REAL DATA
DEFAULT_TAU_FLOOR_M: float = 0.10  # TO TUNE ON REAL DATA: max |Z| for floor points
DEFAULT_VOTE_THRESHOLD: float = 0.5  # TO TUNE ON REAL DATA


@dataclass(frozen=True)
class MapConfig:
    """Configuration for bird's-eye view (BEV) map.

    All numeric defaults are starting values to be tuned on real data.
    """

    res: float = DEFAULT_RES_M  # TO TUNE ON REAL DATA
    x_min: float = DEFAULT_X_MIN_M  # TO TUNE ON REAL DATA
    x_max: float = DEFAULT_X_MAX_M  # TO TUNE ON REAL DATA
    y_min: float = DEFAULT_Y_MIN_M  # TO TUNE ON REAL DATA
    y_max: float = DEFAULT_Y_MAX_M  # TO TUNE ON REAL DATA
    z_band_min: float = DEFAULT_Z_BAND_MIN_M  # TO TUNE ON REAL DATA
    z_band_max: float = DEFAULT_Z_BAND_MAX_M  # TO TUNE ON REAL DATA
    min_points_per_cell: int = DEFAULT_MIN_POINTS_PER_CELL  # TO TUNE ON REAL DATA
    tau_floor: float = DEFAULT_TAU_FLOOR_M  # TO TUNE ON REAL DATA
    vote_threshold: float = DEFAULT_VOTE_THRESHOLD  # TO TUNE ON REAL DATA

    @property
    def n_cols(self) -> int:
        return int(round((self.x_max - self.x_min) / self.res))

    @property
    def n_rows(self) -> int:
        return int(round((self.y_max - self.y_min) / self.res))

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols)."""
        return self.n_rows, self.n_cols


def world_to_cell(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    cfg: MapConfig,
) -> tuple[Any, Any]:
    """Convert ground frame G coordinates (X, Y) to grid cell (col, row).

    col = floor((X - x_min) / res)
    row = floor((Y - y_min) / res)
    Row grows with forward distance.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    col = np.floor((x_arr - cfg.x_min) / cfg.res + 1e-9).astype(np.int64)
    row = np.floor((y_arr - cfg.y_min) / cfg.res + 1e-9).astype(np.int64)

    if x_arr.ndim == 0 and y_arr.ndim == 0:
        return int(col), int(row)
    return col, row


def cell_to_world(
    col: Union[int, np.ndarray],
    row: Union[int, np.ndarray],
    cfg: MapConfig,
) -> tuple[Any, Any]:
    """Convert grid cell (col, row) to ground frame G coordinates (X, Y) at cell center.

    X = x_min + (col + 0.5) * res
    Y = y_min + (row + 0.5) * res
    """
    col_arr = np.asarray(col, dtype=np.float64)
    row_arr = np.asarray(row, dtype=np.float64)

    x = cfg.x_min + (col_arr + 0.5) * cfg.res
    y = cfg.y_min + (row_arr + 0.5) * cfg.res

    if col_arr.ndim == 0 and row_arr.ndim == 0:
        return float(x), float(y)
    return x, y


def in_bounds(
    col: Union[int, np.ndarray],
    row: Union[int, np.ndarray],
    cfg: MapConfig,
) -> Union[bool, np.ndarray]:
    """Check if cell (col, row) is within grid bounds [0, n_cols) x [0, n_rows)."""
    c = np.asarray(col, dtype=np.int64)
    r = np.asarray(row, dtype=np.int64)
    valid = (c >= 0) & (c < cfg.n_cols) & (r >= 0) & (r < cfg.n_rows)
    if c.ndim == 0 and r.ndim == 0:
        return bool(valid)
    return valid
