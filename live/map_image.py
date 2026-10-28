"""Standalone session map image: estimated trail, current view, legend.

DISPLAY ONLY. The image is written for the user to inspect; no module reads it
back and it never influences guidance, certification or planning.

The trail comes from ``live.trace`` and therefore drifts. The image says so in
its footer instead of implying a measured path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from mapping.grid import (
    CELL_OCCUPIED,
    MapConfig,
    cell_to_world,
)

TRAIL_BGR = (255, 190, 0)
START_BGR = (90, 220, 90)
CAMERA_BGR = (0, 220, 255)
OBS_BGR = (70, 70, 230)
TEXT_BGR = (235, 235, 235)
MUTED_BGR = (165, 165, 165)
WARN_BGR = (80, 200, 255)
BG_BGR = (26, 26, 26)
GRID_BGR = (48, 48, 48)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# One colour per class, assigned in sorted class-name order. Neighbouring
# entries alternate across the hue wheel so alphabetically adjacent classes do
# not end up with near-identical colours. Classes past the end of the palette
# reuse colours cyclically.
OBJECT_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (40, 40, 220),    # red
    (255, 255, 130),  # light blue
    (0, 140, 255),    # orange
    (255, 120, 60),   # azure
    (0, 215, 255),    # yellow
    (200, 0, 200),    # magenta
    (0, 150, 0),      # dark green
    (255, 60, 180),   # pink
    (200, 90, 0),     # teal
    (180, 180, 255),  # lilac
    (160, 40, 90),    # indigo
    (90, 255, 180),   # yellow-green
)


@dataclass(frozen=True)
class MapObject:
    """One detected object placed in world metres (X right, Y forward)."""

    class_name: str
    x: float
    y: float
    track_id: int = -1


def object_colors(names: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """Assign one stable colour per distinct class name.

    Sorting makes the mapping independent of detection order, so the same
    class keeps the same colour between frames and between runs.
    """
    unique = sorted({str(name) for name in names})
    palette = OBJECT_PALETTE_BGR
    return {name: palette[i % len(palette)] for i, name in enumerate(unique)}


@dataclass(frozen=True)
class MapMeta:
    session_id: str = ""
    frame_index: int = 0
    decision: str = ""
    heading_deg: float = 0.0
    nearest_m: float | None = None
    uncalibrated: bool = False
    distance_m: float = 0.0
    steps: int = 0
    misses: int = 0


def _world_to_px(
    x: np.ndarray | float,
    y: np.ndarray | float,
    center: tuple[float, float],
    scale: float,
    origin: tuple[float, float],
) -> tuple[np.ndarray | float, np.ndarray | float]:
    px = origin[0] + (np.asarray(x) - center[0]) * scale
    py = origin[1] - (np.asarray(y) - center[1]) * scale
    return px, py


def _current_view_points(
    grid: np.ndarray,
    cfg: MapConfig,
    position: tuple[float, float],
    heading_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform occupied cells of the current camera-frame grid into world XY."""
    rows, cols = np.nonzero(np.asarray(grid) == CELL_OCCUPIED)
    if rows.size == 0:
        return np.empty(0), np.empty(0)
    x_cam, y_cam = cell_to_world(cols, rows, cfg)
    heading = math.radians(float(heading_deg))
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    world_x = position[0] + cos_h * x_cam + sin_h * y_cam
    world_y = position[1] - sin_h * x_cam + cos_h * y_cam
    return np.asarray(world_x), np.asarray(world_y)


def _draw_scale_bar(
    image: np.ndarray,
    scale: float,
    origin: tuple[int, int],
) -> None:
    target_px = 110.0
    metres = target_px / scale if scale > 0 else 1.0
    for candidate in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        if candidate >= metres * 0.6:
            metres = candidate
            break
    length_px = int(round(metres * scale))
    x0, y0 = origin
    cv2.line(image, (x0, y0), (x0 + length_px, y0), TEXT_BGR, 2, cv2.LINE_AA)
    cv2.line(image, (x0, y0 - 4), (x0, y0 + 4), TEXT_BGR, 1, cv2.LINE_AA)
    cv2.line(image, (x0 + length_px, y0 - 4), (x0 + length_px, y0 + 4), TEXT_BGR, 1, cv2.LINE_AA)
    label = f"{metres:g} m"
    cv2.putText(image, label, (x0, y0 - 8), FONT, 0.42, MUTED_BGR, 1, cv2.LINE_AA)


def _draw_objects(
    image: np.ndarray,
    objects: Sequence[MapObject],
    center: tuple[float, float],
    scale: float,
    origin: tuple[float, float],
    bounds: tuple[int, int, int, int],
) -> int:
    """Draw one labelled, class-coloured marker per object, clipped to the map.

    Every object gets a marker; a label is skipped when it would overlap an
    already drawn one, because stacked text is unreadable but the marker still
    shows where the object is. Returns the number of labels drawn.
    """
    if not objects:
        return 0
    colours = object_colors([obj.class_name for obj in objects])
    left, top, right, bottom = bounds
    boxes: list[tuple[int, int, int, int]] = []
    labels = 0
    for obj in sorted(objects, key=lambda item: (item.class_name, item.y, item.x)):
        if not (math.isfinite(obj.x) and math.isfinite(obj.y)):
            continue
        px, py = _world_to_px(obj.x, obj.y, center, scale, origin)
        px, py = float(px), float(py)
        if not (left <= px <= right and top <= py <= bottom):
            continue
        colour = colours[obj.class_name]
        point = (int(round(px)), int(round(py)))
        cv2.circle(image, point, 6, BG_BGR, -1, cv2.LINE_AA)
        cv2.circle(image, point, 4, colour, -1, cv2.LINE_AA)
        tx, ty = point[0] + 8, point[1] - 6
        (tw, th), _ = cv2.getTextSize(obj.class_name, FONT, 0.38, 1)
        box = (tx, ty - th, tx + tw, ty + 4)
        if any(
            box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
            for b in boxes
        ):
            continue
        boxes.append(box)
        cv2.putText(
            image, obj.class_name, (tx + 1, ty + 1), FONT, 0.38, BG_BGR, 1, cv2.LINE_AA
        )
        cv2.putText(
            image, obj.class_name, (tx, ty), FONT, 0.38, colour, 1, cv2.LINE_AA
        )
        labels += 1
    return labels


def _draw_object_legend(
    image: np.ndarray,
    colours: dict[str, tuple[int, int, int]],
    bounds: tuple[int, int, int, int],
) -> None:
    """Draw a colour key for the classes present, inside the map area."""
    names = sorted(colours)
    if not names:
        return
    font_scale = 0.38
    row_h = 15
    widths = [
        cv2.getTextSize(name, FONT, font_scale, 1)[0][0] for name in names
    ]
    box_w = max(widths) + 34
    box_h = 8 + row_h * len(names)
    left, top, right, bottom = bounds
    x1 = right - box_w
    y1 = top + 6
    if x1 < left or y1 + box_h > bottom:
        return
    cv2.rectangle(image, (x1, y1), (x1 + box_w, y1 + box_h), BG_BGR, -1)
    cv2.rectangle(image, (x1, y1), (x1 + box_w, y1 + box_h), GRID_BGR, 1)
    for i, name in enumerate(names):
        y = y1 + 8 + row_h * i + row_h // 2
        cv2.circle(image, (x1 + 12, y), 4, colours[name], -1, cv2.LINE_AA)
        cv2.putText(
            image,
            name,
            (x1 + 22, y + 4),
            FONT,
            font_scale,
            TEXT_BGR,
            1,
            cv2.LINE_AA,
        )


def render_session_map(
    points: Sequence[tuple[float, float]],
    grid: np.ndarray | None,
    cfg: MapConfig | None = None,
    *,
    heading_deg: float = 0.0,
    meta: MapMeta | None = None,
    objects: Sequence[MapObject] | None = None,
    show_legend: bool = True,
    size: tuple[int, int] = (900, 920),
) -> np.ndarray:
    """Render a top-down session map with the estimated trail.

    ``objects`` are world-frame positions of detected objects; they are drawn
    with a class label and a colour that is fixed per class name.
    """
    if cfg is None:
        cfg = MapConfig()
    width, height = int(size[0]), int(size[1])
    header_h = 52
    footer_h = 78
    margin = 18
    image = np.full((height, width, 3), BG_BGR, dtype=np.uint8)

    trail = [(float(x), float(y)) for x, y in points] or [(0.0, 0.0)]
    current = trail[-1]
    seen = [
        (float(obj.x), float(obj.y))
        for obj in (objects or ())
        if math.isfinite(float(obj.x)) and math.isfinite(float(obj.y))
    ]
    xs = [p[0] for p in trail] + [0.0, current[0]] + [p[0] for p in seen]
    ys = [p[1] for p in trail] + [0.0, current[1]] + [p[1] for p in seen]
    span_x = max(max(xs) - min(xs), 8.0)
    span_y = max(max(ys) - min(ys), 8.0)
    pad_m = max(2.5, 0.08 * max(span_x, span_y))
    center_x = 0.5 * (min(xs) + max(xs))
    center_y = 0.5 * (min(ys) + max(ys))
    map_w = width - 2 * margin
    map_h = height - header_h - footer_h - 2 * margin
    scale = min(map_w / (span_x + 2 * pad_m), map_h / (span_y + 2 * pad_m))
    origin = (margin + map_w / 2.0, header_h + margin + map_h / 2.0)

    # Metre grid.
    step_m = 1.0 if max(span_x, span_y) <= 30.0 else 5.0
    half_x = 0.5 * map_w / scale
    half_y = 0.5 * map_h / scale
    x_from = math.floor((center_x - half_x) / step_m) * step_m
    x_to = center_x + half_x
    y_from = math.floor((center_y - half_y) / step_m) * step_m
    y_to = center_y + half_y
    value = x_from
    while value <= x_to:
        px, _ = _world_to_px(value, 0.0, (center_x, center_y), scale, origin)
        cv2.line(
            image,
            (int(round(px)), header_h + margin),
            (int(round(px)), height - footer_h - margin),
            GRID_BGR,
            1,
        )
        value += step_m
    value = y_from
    while value <= y_to:
        _, py = _world_to_px(0.0, value, (center_x, center_y), scale, origin)
        cv2.line(
            image,
            (margin, int(round(py))),
            (width - margin, int(round(py))),
            GRID_BGR,
            1,
        )
        value += step_m

    # World origin cross.
    ox, oy = _world_to_px(0.0, 0.0, (center_x, center_y), scale, origin)
    cv2.drawMarker(image, (int(ox), int(oy)), MUTED_BGR, cv2.MARKER_CROSS, 12, 1)

    if grid is not None:
        obs_x, obs_y = _current_view_points(np.asarray(grid), cfg, current, heading_deg)
        if obs_x.size:
            px, py = _world_to_px(obs_x, obs_y, (center_x, center_y), scale, origin)
            for x, y in zip(np.atleast_1d(px), np.atleast_1d(py)):
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(image, (int(x), int(y)), 2, OBS_BGR, -1)

    if len(trail) >= 2:
        px, py = _world_to_px(
            np.array([p[0] for p in trail]),
            np.array([p[1] for p in trail]),
            (center_x, center_y),
            scale,
            origin,
        )
        poly = np.stack([np.asarray(px), np.asarray(py)], axis=1).astype(np.int32)
        cv2.polylines(image, [poly], False, TRAIL_BGR, 3, cv2.LINE_AA)

    _draw_objects(
        image,
        objects or (),
        (center_x, center_y),
        scale,
        origin,
        (margin, header_h + margin, width - margin, height - footer_h - margin),
    )

    sx, sy = _world_to_px(trail[0][0], trail[0][1], (center_x, center_y), scale, origin)
    cv2.circle(image, (int(sx), int(sy)), 5, START_BGR, -1, cv2.LINE_AA)
    cv2.putText(image, "start", (int(sx) + 8, int(sy) - 6), FONT, 0.40, START_BGR, 1, cv2.LINE_AA)

    heading = math.radians(float(heading_deg))
    forward = np.array([math.sin(heading), math.cos(heading)])
    right = np.array([math.cos(heading), -math.sin(heading)])
    tip = np.array(current) + forward * 0.45
    left = np.array(current) - forward * 0.12 - right * 0.24
    right_pt = np.array(current) - forward * 0.12 + right * 0.24
    triangle_world = np.array([tip, left, right_pt])
    tx, ty = _world_to_px(
        triangle_world[:, 0], triangle_world[:, 1], (center_x, center_y), scale, origin
    )
    triangle = np.stack([np.asarray(tx), np.asarray(ty)], axis=1).astype(np.int32)
    cv2.fillConvexPoly(image, triangle, CAMERA_BGR, cv2.LINE_AA)

    if show_legend:
        _draw_object_legend(
            image,
            object_colors([obj.class_name for obj in (objects or ())]),
            (margin, header_h + margin, width - margin, height - footer_h - margin),
        )

    cv2.putText(image, "Visiona3D session map", (margin, 24), FONT, 0.62, TEXT_BGR, 1, cv2.LINE_AA)
    subtitle = "estimated trail; detected objects coloured and labelled by class"
    cv2.putText(image, subtitle, (margin, 43), FONT, 0.40, MUTED_BGR, 1, cv2.LINE_AA)

    footer_y = height - footer_h + 8
    cv2.putText(
        image,
        "TRAIL IS ESTIMATED from floor-plane optical flow (display only).",
        (margin, footer_y + 12),
        FONT,
        0.45,
        WARN_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "It drifts over distance. Display only - never used for guidance or safety.",
        (margin, footer_y + 30),
        FONT,
        0.42,
        WARN_BGR,
        1,
        cv2.LINE_AA,
    )
    if meta is not None:
        line = (
            f"frame {meta.frame_index}   {meta.decision or 'NO_PERCEPTION'}   "
            f"heading {meta.heading_deg:+.0f} deg   travelled {meta.distance_m:.1f} m   "
            f"steps {meta.steps}   gaps {meta.misses}"
        )
        cv2.putText(image, line, (margin, footer_y + 50), FONT, 0.42, TEXT_BGR, 1, cv2.LINE_AA)
        extra = []
        if meta.nearest_m is not None:
            extra.append(f"nearest obstacle {meta.nearest_m:.1f} m")
        if meta.session_id:
            extra.append(f"session {meta.session_id}")
        if meta.uncalibrated:
            extra.append("UNCALIBRATED: distances are estimates")
        if extra:
            cv2.putText(
                image,
                "   ".join(extra),
                (margin, footer_y + 67),
                FONT,
                0.38,
                WARN_BGR if meta.uncalibrated else MUTED_BGR,
                1,
                cv2.LINE_AA,
            )

    _draw_scale_bar(image, scale, (width - 150, height - footer_h - 24))

    cv2.putText(image, "X right / Y forward (metres)", (margin, header_h + 14), FONT, 0.38, MUTED_BGR, 1, cv2.LINE_AA)
    return image


def save_session_map(path: str | Path, image: np.ndarray) -> Path:
    """Write the map image atomically so a reader never sees a partial PNG."""
    out = Path(path).expanduser()
    if out.suffix == "":
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.tmp{out.suffix}")
    try:
        ok = cv2.imwrite(str(tmp), image)
        if not ok:
            raise OSError(f"could not encode map image: {out}")
        os.replace(tmp, out)
    finally:
        if tmp.exists() and tmp != out:
            try:
                tmp.unlink()
            except OSError:
                pass
    return out


__all__ = [
    "MapMeta",
    "MapObject",
    "OBJECT_PALETTE_BGR",
    "object_colors",
    "render_session_map",
    "save_session_map",
]
