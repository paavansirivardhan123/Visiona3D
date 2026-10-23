"""Live HUD: segmentation, 3D object labels, BEV inset and guidance strip.

All drawing happens on the exact frame that was processed, never on a newer
capture frame. That keeps masks, boxes and anchors aligned with the pixels
they were computed from; the HUD shows the frame age so latency is visible
instead of hidden.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np

from core.config import MASK_FLOOR, MASK_OBSTACLE, MASK_OTHER, MASK_WALL
from geometry.camera import project
from geometry.frames import ground_to_camera
from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNCERTAIN,
    CELL_UNKNOWN,
    MapConfig,
    world_to_cell,
)

# BGR palette, matching the offline segmentation colours.
CLASS_COLORS_BGR = {
    MASK_FLOOR: (70, 180, 40),
    MASK_WALL: (220, 90, 60),
    MASK_OBSTACLE: (50, 60, 220),
    MASK_OTHER: (140, 140, 140),
}

BEV_COLORS_BGR = {
    CELL_UNKNOWN: (85, 80, 80),
    CELL_FREE: (210, 245, 210),
    CELL_OCCUPIED: (50, 50, 220),
    CELL_UNCERTAIN: (50, 190, 240),
}

DECISION_COLORS_BGR = {
    "GUIDE": (80, 200, 80),
    "CAUTION": (40, 170, 255),
    "STOP_UNCLEAR": (40, 40, 230),
    "STOP_BLOCKED": (40, 40, 230),
    "NO_PERCEPTION": (40, 40, 230),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (245, 245, 245)
BLACK = (18, 18, 18)
MUTED = (185, 185, 185)


def decision_color(decision: str) -> tuple[int, int, int]:
    return DECISION_COLORS_BGR.get(str(decision), (160, 160, 160))


def blend_segmentation(
    frame_bgr: np.ndarray,
    labels: np.ndarray | None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend floor/wall/obstacle masks onto a copy of the frame."""
    out = frame_bgr.copy()
    if labels is None:
        return out
    arr = np.asarray(labels)
    if arr.shape != frame_bgr.shape[:2]:
        return out
    blend = float(np.clip(alpha, 0.0, 1.0))
    for class_id, color in CLASS_COLORS_BGR.items():
        mask = arr == class_id
        if not np.any(mask):
            continue
        out[mask] = (
            (1.0 - blend) * out[mask] + blend * np.asarray(color, dtype=np.float32)
        ).astype(np.uint8)
    return out


def _clip_text(frame: np.ndarray, x: int, y: int, text: str) -> tuple[int, int]:
    h, w = frame.shape[:2]
    x = max(2, min(int(x), w - 4))
    y = max(12, min(int(y), h - 4))
    return x, y


def draw_objects(
    frame_bgr: np.ndarray,
    observations: Sequence[Any],
    K: Any = None,
    plane_normal: np.ndarray | None = None,
    mount_height_m: float | None = None,
) -> np.ndarray:
    """Draw boxes, track IDs, distances and projected 3D anchor points."""
    out = frame_bgr
    h, w = out.shape[:2]
    if plane_normal is not None and mount_height_m is not None:
        normal = np.asarray(plane_normal, dtype=np.float64)
        if normal.shape == (3,):
            normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        else:
            normal = None
    else:
        normal = None

    ordered = sorted(
        observations,
        key=lambda obs: float(getattr(obs, "distance_m", float("inf"))),
    )
    for obs in ordered[:20]:
        box = np.asarray(obs.box_xyxy, dtype=np.float64).reshape(-1)
        if box.size != 4:
            continue
        x1, y1, x2, y2 = box.tolist()
        distance = float(getattr(obs, "distance_m", float("nan")))
        if distance < 1.5:
            color = (40, 40, 230)
        elif distance < 3.0:
            color = (40, 170, 255)
        else:
            color = (0, 220, 255)
        x1i = max(0, min(int(round(x1)), w - 1))
        y1i = max(0, min(int(round(y1)), h - 1))
        x2i = max(0, min(int(round(x2)), w - 1))
        y2i = max(0, min(int(round(y2)), h - 1))
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)

        lateral = float(getattr(obs, "lateral_m", float("nan")))
        if math.isfinite(lateral):
            side = "R" if lateral >= 0.0 else "L"
            side_text = f"{side}{abs(lateral):.1f}m"
        else:
            side_text = "--"
        label = (
            f"#{int(getattr(obs, 'track_id', -1))} "
            f"{str(getattr(obs, 'class_name', '?'))} "
            f"{float(getattr(obs, 'conf', 0.0)):.2f} "
            f"{distance:.1f}m {side_text}"
        )
        tx, ty = _clip_text(out, x1i, y1i - 6, label)
        cv2.putText(out, label, (tx, ty), FONT, 0.42, BLACK, 3, cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), FONT, 0.42, color, 1, cv2.LINE_AA)

        if K is not None and normal is not None:
            position = np.asarray(getattr(obs, "position_g", (0.0, 0.0, 0.0)))
            if position.shape == (3,) and np.isfinite(position).all():
                point_c = ground_to_camera(
                    position.reshape(1, 3), normal, float(mount_height_m)
                )
                if float(point_c[0, 2]) > 0.05:
                    uv = project(point_c, K)[0]
                    u, v = int(round(uv[0])), int(round(uv[1]))
                    if 0 <= u < w and 0 <= v < h:
                        cv2.circle(out, (u, v), 5, color, -1, cv2.LINE_AA)
                        cv2.circle(out, (u, v), 8, WHITE, 1, cv2.LINE_AA)
    return out


def render_bev_panel(
    grid: np.ndarray | None,
    path_cells: Sequence[tuple[int, int]] = (),
    target_cell: tuple[int, int] | None = None,
    cfg: MapConfig | None = None,
    size: tuple[int, int] = (320, 360),
    title: str = "BEV 2.5D  grid 0.1 m",
) -> np.ndarray:
    """Render a BGR BEV panel with the robot, path and target."""
    panel_w, panel_h = int(size[0]), int(size[1])
    if cfg is None:
        cfg = MapConfig()
    panel = np.full((panel_h, panel_w, 3), 28, dtype=np.uint8)
    header_h = 24
    footer_h = 26
    map_h = max(1, panel_h - header_h - footer_h)
    map_w = max(1, panel_w - 8)
    cv2.putText(panel, title, (8, 17), FONT, 0.44, WHITE, 1, cv2.LINE_AA)
    if grid is None:
        cv2.putText(
            panel,
            "waiting for first map",
            (12, header_h + 30),
            FONT,
            0.48,
            MUTED,
            1,
            cv2.LINE_AA,
        )
        return panel

    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.size == 0:
        return panel
    n_rows, n_cols = arr.shape
    img = np.full((n_rows, n_cols, 3), BEV_COLORS_BGR[CELL_UNKNOWN], dtype=np.uint8)
    img[arr == CELL_FREE] = BEV_COLORS_BGR[CELL_FREE]
    img[arr == CELL_OCCUPIED] = BEV_COLORS_BGR[CELL_OCCUPIED]
    img[arr == CELL_UNCERTAIN] = BEV_COLORS_BGR[CELL_UNCERTAIN]

    for c, r in path_cells:
        if 0 <= c < n_cols and 0 <= r < n_rows:
            img[r, c] = (255, 120, 30)
    if target_cell is not None:
        tc, tr = target_cell
        if 0 <= tc < n_cols and 0 <= tr < n_rows:
            img[max(0, tr - 1) : tr + 2, max(0, tc - 1) : tc + 2] = (255, 0, 255)

    # Forward is up, camera origin near the bottom.
    img = np.flipud(img)
    scale_x = map_w / float(n_cols)
    scale_y = map_h / float(n_rows)
    resized = cv2.resize(
        img, (map_w, map_h), interpolation=cv2.INTER_NEAREST
    )

    origin_c, origin_r = world_to_cell(0.0, 0.0, cfg)
    ox = int((origin_c + 0.5) * scale_x)
    oy = int((n_rows - 1 - origin_r + 0.5) * scale_y)
    ox = max(4, min(map_w - 4, ox))
    oy = max(6, min(map_h - 4, oy))
    triangle = np.array(
        [[ox, oy - 7], [ox - 5, oy + 5], [ox + 5, oy + 5]], dtype=np.int32
    )
    cv2.fillConvexPoly(resized, triangle, (255, 255, 0), cv2.LINE_AA)
    if target_cell is not None:
        tc, tr = target_cell
        tx = int((tc + 0.5) * scale_x)
        ty = int((n_rows - 1 - tr + 0.5) * scale_y)
        if 0 <= tx < map_w and 0 <= ty < map_h:
            cv2.drawMarker(
                resized, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 10, 2
            )

    panel[header_h : header_h + map_h, 4 : 4 + map_w] = resized
    legend_y = panel_h - 8
    cv2.putText(panel, "FREE", (8, legend_y), FONT, 0.38, BEV_COLORS_BGR[CELL_FREE], 1, cv2.LINE_AA)
    cv2.putText(panel, "OCC", (78, legend_y), FONT, 0.38, BEV_COLORS_BGR[CELL_OCCUPIED], 1, cv2.LINE_AA)
    cv2.putText(panel, "UNC", (132, legend_y), FONT, 0.38, BEV_COLORS_BGR[CELL_UNCERTAIN], 1, cv2.LINE_AA)
    cv2.putText(panel, "PATH", (186, legend_y), FONT, 0.38, (255, 120, 30), 1, cv2.LINE_AA)
    return panel


def _bar(
    canvas: np.ndarray,
    y: int,
    height: int,
    color: tuple[int, int, int],
    alpha: float = 0.82,
) -> np.ndarray:
    """Draw a translucent horizontal bar and return it as the text target."""
    region = canvas[y : y + height, :, :]
    overlay = np.full_like(region, color)
    canvas[y : y + height, :, :] = cv2.addWeighted(
        region, 1.0 - alpha, overlay, alpha, 0.0
    )
    return canvas


def compose_hud(
    frame_bgr: np.ndarray,
    *,
    guidance: Any = None,
    observations: Sequence[Any] = (),
    nearest: dict | None = None,
    bev_panel: np.ndarray | None = None,
    status_line: str = "",
    frame_age_s: float | None = None,
    process_ms: float | None = None,
    fps: float | None = None,
    db_line: str = "",
    uncalibrated: bool = False,
    profile: str = "",
) -> np.ndarray:
    """Compose camera view + BEV panel + status strips into one BGR image."""
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    panel_w = 0 if bev_panel is None else int(bev_panel.shape[1])
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    if bev_panel is not None:
        canvas[: min(h, bev_panel.shape[0]), w : w + panel_w] = bev_panel[
            : min(h, bev_panel.shape[0])
        ]

    top_h = 44 if uncalibrated else 24
    _bar(canvas, 0, top_h, (20, 20, 20), alpha=0.70)
    top_text = status_line or "Visiona-Walk live"
    cv2.putText(canvas, top_text, (8, 16), FONT, 0.44, WHITE, 1, cv2.LINE_AA)
    if uncalibrated:
        warn = "UNCALIBRATED INTRINSICS - DISTANCES ARE ESTIMATES"
        cv2.putText(canvas, warn, (8, 38), FONT, 0.42, (60, 60, 255), 1, cv2.LINE_AA)

    bottom_h = 52
    decision = str(getattr(guidance, "decision", "NO_PERCEPTION"))
    color = decision_color(decision)
    _bar(canvas, h - bottom_h, bottom_h, color, alpha=0.85)

    heading = float(getattr(guidance, "heading_deg", 0.0) or 0.0)
    cv2.putText(
        canvas,
        f"{decision}   heading {heading:+.0f} deg",
        (8, h - bottom_h + 18),
        FONT,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    nearest_text = "nearest obstacle: none"
    if nearest:
        nominal = nearest.get("distance_nominal_m")
        lower = nearest.get("distance_lower_m")
        bearing = nearest.get("bearing_deg")
        if nominal is not None and math.isfinite(float(nominal)):
            nearest_text = f"nearest obstacle {float(nominal):.1f} m"
            if bearing is not None and math.isfinite(float(bearing)):
                nearest_text += f" at {float(bearing):+.0f} deg"
            if lower is not None and math.isfinite(float(lower)):
                nearest_text += f" (lower bound {float(lower):.1f} m)"
    cv2.putText(
        canvas, nearest_text, (8, h - bottom_h + 36), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA
    )

    meta_parts = []
    if frame_age_s is not None:
        meta_parts.append(f"frame age {float(frame_age_s):.1f}s")
    if process_ms is not None:
        meta_parts.append(f"process {float(process_ms):.0f}ms")
    if fps is not None and math.isfinite(float(fps)):
        meta_parts.append(f"{float(fps):.1f} fps")
    if profile:
        meta_parts.append(profile)
    if db_line:
        meta_parts.append(db_line)
    if observations:
        meta_parts.append(f"{len(observations)} objects")
    meta = "  |  ".join(meta_parts)
    if meta:
        tw = cv2.getTextSize(meta, FONT, 0.42, 1)[0][0]
        cv2.putText(
            canvas,
            meta,
            (max(8, w + panel_w - tw - 10), h - 8),
            FONT,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return canvas


def loading_frame(width: int, height: int, message: str) -> np.ndarray:
    canvas = np.full((max(120, int(height)), max(320, int(width)), 3), 24, dtype=np.uint8)
    cv2.putText(
        canvas,
        message,
        (16, int(canvas.shape[0]) // 2),
        FONT,
        0.62,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    return canvas


__all__ = [
    "BEV_COLORS_BGR",
    "CLASS_COLORS_BGR",
    "blend_segmentation",
    "compose_hud",
    "decision_color",
    "draw_objects",
    "loading_frame",
    "render_bev_panel",
]
