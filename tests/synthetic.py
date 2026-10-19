"""Analytic synthetic scene renderer for camera and ground geometry validation.

Computes exact depth maps analytically via ray-plane and ray-box intersections
without drawing or rasterization.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np

from core.config import Intrinsics
from geometry.camera import _extract_intrinsics
from geometry.frames import ground_frame_basis


def render_scene(
    K: Union[Intrinsics, np.ndarray],
    H: int,
    W: int,
    h: float,
    pitch_deg: float,
    wall_distance: Optional[float] = None,
    boxes: Sequence[Union[Sequence[float], Mapping[str, float]]] = (),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Render an analytic scene with floor, optional wall, and optional 3D boxes.

    Computes exact depth per pixel via ray-surface intersection in camera frame C
    and ground frame G.

    Args:
        K: Camera Intrinsics or 3x3 matrix.
        H: Image height in pixels.
        W: Image width in pixels.
        h: Camera height above ground plane in metres (h > 0).
        pitch_deg: Camera pitch down in degrees.
        wall_distance: Distance along ground frame Y to vertical wall in metres.
        boxes: List of axis-aligned boxes in G. Each box can be:
            - tuple (x_min, x_max, y_min, y_max, z_min, z_max)
            - dict with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        seed: Random seed (unused for purely analytic rendering, preserved for API).

    Returns:
        Tuple of:
        - depth: (H, W) array of analytic depths (z coordinate in C).
        - floor_mask: (H, W) boolean mask for floor.
        - wall_mask: (H, W) boolean mask for vertical wall.
        - per_box_masks: list of (H, W) boolean masks for each box.
        - gt: dict of ground truth parameters.
    """
    fx, fy, cx, cy = _extract_intrinsics(K)

    # Unit ray direction representation in C: p(t) = t * v_c where v_c = [dx, dy, 1.0]
    # Because v_c[2] == 1.0, the parameter t is exactly the camera depth z!
    v_grid, u_grid = np.indices((H, W), dtype=np.float64)
    dx = (u_grid - cx) / fx
    dy = (v_grid - cy) / fy
    dz = np.ones_like(dx)
    v_c = np.stack([dx, dy, dz], axis=-1)  # (H, W, 3)

    # Floor normal in C: n = (0, cos(theta), sin(theta))
    theta_rad = math.radians(pitch_deg)
    n = np.array([0.0, math.cos(theta_rad), math.sin(theta_rad)], dtype=np.float64)
    e_right, e_fwd, e_up = ground_frame_basis(n)

    # 1. Floor plane intersection: n . (t * v_c) = h => t = h / (n . v_c)
    n_dot_v = np.sum(v_c * n, axis=-1)
    t_floor = np.full((H, W), np.inf, dtype=np.float64)
    valid_floor = n_dot_v > 1e-8
    t_floor[valid_floor] = h / n_dot_v[valid_floor]
    # Floor hits must be in front of camera
    t_floor[t_floor <= 0.0] = np.inf

    # 2. Vertical wall intersection at Ground frame Y = wall_distance
    # In G: Y(t) = t * (e_fwd . v_c)
    t_wall = np.full((H, W), np.inf, dtype=np.float64)
    v_dot_fwd = np.sum(v_c * e_fwd, axis=-1)
    if wall_distance is not None and wall_distance > 0.0:
        valid_wall_dir = v_dot_fwd > 1e-8
        t_wall_cand = wall_distance / np.where(valid_wall_dir, v_dot_fwd, 1.0)
        # Wall is only above ground (Z >= 0 in G): Z = h - n . (t * v_c) >= 0
        z_at_wall = h - t_wall_cand * n_dot_v
        valid_wall = valid_wall_dir & (t_wall_cand > 0.0) & (z_at_wall >= -1e-6)
        t_wall[valid_wall] = t_wall_cand[valid_wall]

    # 3. Axis-aligned boxes in G
    # Ray origin in G: O_G = (0, 0, h)
    # Ray direction in G: V_G = (e_right . v_c, e_fwd . v_c, e_up . v_c)
    v_dot_right = np.sum(v_c * e_right, axis=-1)
    v_dot_up = np.sum(v_c * e_up, axis=-1)

    t_boxes = []
    box_positions = []
    for box in boxes:
        if isinstance(box, dict):
            bx0, bx1 = float(box["x_min"]), float(box["x_max"])
            by0, by1 = float(box["y_min"]), float(box["y_max"])
            bz0, bz1 = float(box["z_min"]), float(box["z_max"])
        else:
            bx0, bx1, by0, by1, bz0, bz1 = [float(v) for v in box]

        box_positions.append(((bx0 + bx1) / 2.0, (by0 + by1) / 2.0, (bz0 + bz1) / 2.0))

        # Slab intersection in G
        # X slab
        with np.errstate(divide="ignore", invalid="ignore"):
            tx0 = np.where(np.abs(v_dot_right) > 1e-8, bx0 / v_dot_right, np.where(bx0 <= 0 <= bx1, -np.inf, np.inf))
            tx1 = np.where(np.abs(v_dot_right) > 1e-8, bx1 / v_dot_right, np.where(bx0 <= 0 <= bx1, np.inf, -np.inf))
            t_min_x = np.minimum(tx0, tx1)
            t_max_x = np.maximum(tx0, tx1)

            # Y slab
            ty0 = np.where(np.abs(v_dot_fwd) > 1e-8, by0 / v_dot_fwd, np.where(by0 <= 0 <= by1, -np.inf, np.inf))
            ty1 = np.where(np.abs(v_dot_fwd) > 1e-8, by1 / v_dot_fwd, np.where(by0 <= 0 <= by1, np.inf, -np.inf))
            t_min_y = np.minimum(ty0, ty1)
            t_max_y = np.maximum(ty0, ty1)

            # Z slab (origin at h)
            tz0 = np.where(np.abs(v_dot_up) > 1e-8, (bz0 - h) / v_dot_up, np.where(bz0 <= h <= bz1, -np.inf, np.inf))
            tz1 = np.where(np.abs(v_dot_up) > 1e-8, (bz1 - h) / v_dot_up, np.where(bz0 <= h <= bz1, np.inf, -np.inf))
            t_min_z = np.minimum(tz0, tz1)
            t_max_z = np.maximum(tz0, tz1)

            t_enter = np.maximum(np.maximum(t_min_x, t_min_y), t_min_z)
            t_exit = np.minimum(np.minimum(t_max_x, t_max_y), t_max_z)

        valid_box = (t_enter <= t_exit) & (t_exit > 0.0)
        t_box = np.full((H, W), np.inf, dtype=np.float64)
        hit_t = np.where(t_enter > 0.0, t_enter, t_exit)
        valid_box = valid_box & (hit_t > 0.0)
        t_box[valid_box] = hit_t[valid_box]
        t_boxes.append(t_box)

    # 4. Nearest hit per pixel
    all_hits = [t_floor, t_wall] + t_boxes
    hit_stack = np.stack(all_hits, axis=-1)  # (H, W, num_surfaces)
    min_hit_idx = np.argmin(hit_stack, axis=-1)
    min_t = np.take_along_axis(hit_stack, min_hit_idx[..., None], axis=-1)[..., 0]

    has_hit = np.isfinite(min_t) & (min_t > 0.0)
    depth = np.where(has_hit, min_t, 0.0)

    floor_mask = has_hit & (min_hit_idx == 0)
    wall_mask = has_hit & (min_hit_idx == 1) if wall_distance is not None else np.zeros((H, W), dtype=bool)

    per_box_masks = []
    for idx in range(len(boxes)):
        per_box_masks.append(has_hit & (min_hit_idx == (2 + idx)))

    gt_dict: dict[str, Any] = {
        "h": h,
        "n": n,
        "pitch_deg": pitch_deg,
        "wall_distance": wall_distance,
        "boxes": list(boxes),
        "box_positions": box_positions,
    }

    return depth, floor_mask, wall_mask, per_box_masks, gt_dict


def add_noise(
    depth: np.ndarray,
    rel_sigma: float = 0.02,
    outlier_frac: float = 0.10,
    outlier_scale: tuple[float, float] = (0.3, 3.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian relative depth noise and uniform outlier multiplier noise.

    Args:
        depth: Clean depth map (H, W).
        rel_sigma: Relative Gaussian standard deviation (e.g. 0.02 for 2%).
        outlier_frac: Fraction of valid pixels corrupted with outliers.
        outlier_scale: Range [min, max] of multiplicative outlier scale factors.
        rng: Seeded numpy Generator.

    Returns:
        Corrupted depth map of shape (H, W).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    depth_out = depth.copy()
    valid = depth > 0.0
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return depth_out

    # Add Gaussian relative noise
    noise = rng.normal(0.0, rel_sigma, size=depth.shape)
    depth_out[valid] = np.maximum(0.05, depth_out[valid] * (1.0 + noise[valid]))

    # Add outliers
    if outlier_frac > 0.0:
        valid_indices = np.where(valid)
        n_outliers = int(outlier_frac * n_valid)
        if n_outliers > 0:
            outlier_pick = rng.choice(n_valid, size=n_outliers, replace=False)
            chosen_u = valid_indices[0][outlier_pick]
            chosen_v = valid_indices[1][outlier_pick]
            factors = rng.uniform(outlier_scale[0], outlier_scale[1], size=n_outliers)
            depth_out[chosen_u, chosen_v] = np.maximum(0.05, depth_out[chosen_u, chosen_v] * factors)

    return depth_out


def scale_depth(depth: np.ndarray, k: float) -> np.ndarray:
    """Scale depth map by constant factor k."""
    return depth * k


def save_scene(
    out_dir: Union[str, Path],
    frame_id: str,
    depth: np.ndarray,
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    box_masks: Sequence[np.ndarray],
    gt: dict[str, Any],
    K: Union[Intrinsics, np.ndarray],
) -> dict[str, str]:
    """Save a rendered scene to disk as NPZ + JSON metadata.

    Creates:
    - {out_dir}/{frame_id}.npz: depth, floor_mask, wall_mask, box_masks (if any)
    - {out_dir}/{frame_id}_meta.json: ground truth parameters and intrinsics

    Args:
        out_dir: Output directory path.
        frame_id: Frame identifier (e.g., "route_001_frame_0005").
        depth: (H, W) depth map in metres (z in camera frame C).
        floor_mask: (H, W) boolean floor mask.
        wall_mask: (H, W) boolean wall mask.
        box_masks: List of (H, W) boolean masks for boxes.
        gt: Ground truth dict from render_scene.
        K: Camera intrinsics.

    Returns:
        Dict with keys 'npz_path' and 'json_path'.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Prepare masks array for NPZ (stack box masks if any)
    masks_dict = {
        "depth": depth.astype(np.float32),
        "floor_mask": floor_mask,
        "wall_mask": wall_mask,
    }
    for i, bm in enumerate(box_masks):
        masks_dict[f"box_mask_{i}"] = bm

    npz_path = out_path / f"{frame_id}.npz"
    np.savez_compressed(npz_path, **masks_dict)

    # Save metadata as JSON
    import json
    fx, fy, cx, cy = _extract_intrinsics(K)
    meta = {
        "frame_id": frame_id,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "ground_truth": {
            "h": gt.get("h"),
            "pitch_deg": gt.get("pitch_deg"),
            "wall_distance": gt.get("wall_distance"),
            "boxes": gt.get("boxes", []),
            "box_positions": gt.get("box_positions", []),
        },
        "mask_classes": {
            "unlabeled": 0,
            "floor": 1,
            "wall": 2,
            "obstacle": 3,
            "other": 4,
            "ignore": 255,
        },
    }
    json_path = out_path / f"{frame_id}_meta.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"npz_path": str(npz_path), "json_path": str(json_path)}
