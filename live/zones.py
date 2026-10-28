"""Accumulate the walkable-zone layer drawn on the session map.

DISPLAY ONLY. Nothing here feeds guidance, certification or planning. The
layer answers "where was free floor, where was a wall, where was danger and
where was simply never detected" in the world frame fixed at session start.

Every processed frame's BEV grid is classified in the ego frame and then
rotated into the world with the same estimated pose that places the trail and
the object markers, so this layer drifts in the same way and is labelled as an
estimate on the image.

Classification per grid cell:
    wall      a wall-labelled point falls in the cell (permanent)
    danger    occupied for any other reason (obstacle, vetoed floor, any
              non-certified point inside the height band)
    walkable  certified free floor

A cell near a detected object is not voted as danger: the object already has a
class-coloured marker, and a person standing in front of the camera must not
paint a permanent red zone. Unknown cells are not stored at all; the renderer
draws a dark-red band one cell wide around the outer edge of the known area, so
the restricted boundary is visible without flooding the picture.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from core.config import MASK_WALL
from live.map_image import MAP_ZONE_CELL_M as ZONE_CELL_M
from live.map_image import MapZone
from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    MapConfig,
    cell_to_world,
    in_bounds,
    world_to_cell,
)

ZONE_OBJECT_CLEAR_M: float = 0.6  # TO TUNE ON REAL DATA: no danger votes this close to an object
ZONE_UNKNOWN_BAND_CELLS: int = 1  # TO TUNE ON REAL DATA: dark-red edge width in cells

_WALL, _DANGER, _WALKABLE = 0, 1, 2


def _object_xy(obj) -> tuple[float, float] | None:
    """Accept an ObjectAnchor or a plain (lateral, forward) pair."""
    position = getattr(obj, "position_g", obj)
    try:
        lateral, forward = float(position[0]), float(position[1])
    except (TypeError, IndexError, ValueError):
        return None
    if not (math.isfinite(lateral) and math.isfinite(forward)):
        return None
    return lateral, forward


def frame_zone_votes(
    grid: np.ndarray,
    labels: np.ndarray,
    pointmap_g: np.ndarray,
    cfg: MapConfig,
    *,
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    objects: Iterable[object] = (),
) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Classify one frame's BEV grid and return world-cell vote counts.

    ``pose`` is ``(x, y, heading_deg)`` of the estimated trail at the time the
    frame was processed. Returns an empty mapping on any shape mismatch; a
    display layer must never raise into the navigation loop.
    """
    grid_arr = np.asarray(grid)
    cfg_shape = (int(cfg.n_rows), int(cfg.n_cols))
    if grid_arr.ndim != 2 or grid_arr.shape != cfg_shape:
        return {}

    pts = np.asarray(pointmap_g, dtype=np.float64)
    if pts.ndim == 3 and pts.shape[2] >= 3:
        pts = pts.reshape(-1, pts.shape[2])[:, :3]
    elif pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]
    else:
        return {}
    lbls = np.asarray(labels).reshape(-1)
    if lbls.size != pts.shape[0]:
        return {}

    wall_cells = np.zeros(grid_arr.shape, dtype=bool)
    walls = np.isfinite(pts).all(axis=1) & (lbls == MASK_WALL)
    if walls.any():
        c, r = world_to_cell(pts[walls, 0], pts[walls, 1], cfg)
        inside = in_bounds(c, r, cfg)
        wall_cells[np.asarray(r)[inside], np.asarray(c)[inside]] = True

    danger = (grid_arr == CELL_OCCUPIED) & ~wall_cells
    walkable = (grid_arr == CELL_FREE) & ~wall_cells

    rows, cols = np.mgrid[0 : grid_arr.shape[0], 0 : grid_arr.shape[1]]
    x_cam, y_cam = cell_to_world(cols, rows, cfg)
    x_cam = np.asarray(x_cam, dtype=np.float64)
    y_cam = np.asarray(y_cam, dtype=np.float64)

    if danger.any():
        clear_sq = float(ZONE_OBJECT_CLEAR_M) ** 2
        for obj in objects:
            xy = _object_xy(obj)
            if xy is None:
                continue
            near = (x_cam - xy[0]) ** 2 + (y_cam - xy[1]) ** 2 <= clear_sq
            danger &= ~near

    heading = math.radians(float(pose[2]))
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    world_x = float(pose[0]) + cos_h * x_cam + sin_h * y_cam
    world_y = float(pose[1]) - sin_h * x_cam + cos_h * y_cam
    wcol = np.floor(world_x / ZONE_CELL_M).astype(np.int64)
    wrow = np.floor(world_y / ZONE_CELL_M).astype(np.int64)

    votes: dict[tuple[int, int], list[int]] = {}
    for index, mask in ((_WALL, wall_cells), (_DANGER, danger), (_WALKABLE, walkable)):
        if not mask.any():
            continue
        keys = np.stack([wrow[mask], wcol[mask]], axis=1)
        unique, counts = np.unique(keys, axis=0, return_counts=True)
        for (row, col), count in zip(unique, counts):
            slot = votes.setdefault((int(col), int(row)), [0, 0, 0])
            slot[index] += int(count)
    return {key: tuple(slot) for key, slot in votes.items()}


def merge_zone_votes(
    store: dict[tuple[int, int], list[int]],
    votes: dict[tuple[int, int], tuple[int, int, int]],
) -> None:
    """Add one frame's votes into the session accumulator."""
    for key, counts in votes.items():
        slot = store.get(key)
        if slot is None:
            store[key] = list(counts)
        else:
            slot[0] += counts[0]
            slot[1] += counts[1]
            slot[2] += counts[2]


def _resolve_kind(wall: int, danger: int, walkable: int) -> str | None:
    if wall > 0:
        return "wall"
    if danger > walkable:
        return "danger"
    if walkable > 0:
        return "walkable"
    return None


def resolve_zones(
    store: dict[tuple[int, int], list[int]],
    *,
    res: float = ZONE_CELL_M,
    unknown_band_cells: int = ZONE_UNKNOWN_BAND_CELLS,
) -> list[MapZone]:
    """Turn accumulated votes into world cells for the renderer.

    Wall evidence wins outright; danger needs to outvote walkable evidence, so
    an obstacle that appears for a frame or two and then moves leaves the cell
    walkable. A dark-red unknown band is added around the known area.
    """
    step = float(res)
    zones: list[MapZone] = []
    for (col, row) in sorted(store):
        kind = _resolve_kind(*store[(col, row)])
        if kind is None:
            continue
        zones.append(
            MapZone(kind=kind, x=(col + 0.5) * step, y=(row + 0.5) * step)
        )

    band = int(unknown_band_cells)
    if band > 0 and store:
        unknown: set[tuple[int, int]] = set()
        offsets = [
            (dc, dr)
            for dc in range(-band, band + 1)
            for dr in range(-band, band + 1)
            if dc or dr
        ]
        for (col, row) in store:
            for dc, dr in offsets:
                key = (col + dc, row + dr)
                if key not in store:
                    unknown.add(key)
        for (col, row) in sorted(unknown):
            zones.append(
                MapZone(
                    kind="unknown",
                    x=(col + 0.5) * step,
                    y=(row + 0.5) * step,
                )
            )
    return zones


def zone_summary(
    zones: Sequence[MapZone], *, res: float = ZONE_CELL_M
) -> dict[str, float]:
    """Measure the blue, grey, bright-red and dark-red areas in m^2."""
    cell_m2 = float(res) ** 2
    counts = {"walkable": 0, "wall": 0, "danger": 0, "unknown": 0}
    for zone in zones:
        if zone.kind in counts:
            counts[zone.kind] += 1
    return {kind: count * cell_m2 for kind, count in counts.items()}


__all__ = [
    "ZONE_CELL_M",
    "ZONE_OBJECT_CLEAR_M",
    "ZONE_UNKNOWN_BAND_CELLS",
    "frame_zone_votes",
    "merge_zone_votes",
    "resolve_zones",
    "zone_summary",
]
