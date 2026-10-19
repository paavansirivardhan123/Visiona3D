"""Lifting 2D pixel observations into 3D metric ground frame coordinates.

Combines ground plane fitting, metric scale estimation, backprojection,
and frame transformation into a unified interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Optional, Union
import numpy as np

from core.config import Intrinsics, MountConfig, RansacConfig
from geometry.camera import _extract_intrinsics, backproject
from geometry.frames import camera_to_ground
from geometry.ground_plane import FitStatus, PlaneFit, fit_ground_plane
from geometry.scale import ScaleEstimator, ScaleResult


class AnchorStatus(str, Enum):
    """Status outcomes for object anchor extraction."""

    OK = "OK"
    EMPTY_MASK = "EMPTY_MASK"
    TINY_MASK = "TINY_MASK"


@dataclass(frozen=True)
class Anchor:
    """Estimated 3D spatial anchor for an object observation.

    Attributes:
        distance_m: Representative distance to object (q-th percentile depth) in metres.
        position_g: 3D position [X, Y, Z] in ground frame G in metres.
        status: Anchor computation status.
    """

    distance_m: float
    position_g: np.ndarray
    status: AnchorStatus


@dataclass(frozen=True)
class LiftResult:
    """Result of lifting 2D observations to 3D metric space.

    Attributes:
        points_c: Mapping of label name to 3D points (N, 3) in camera frame C.
        points_g: Mapping of label name to 3D points (N, 3) in ground frame G.
        plane_fit: Estimated ground plane parameters.
        scale_result: Estimated metric scale correction factor.
        scaled_depth: Metric-corrected depth map (H, W).
    """

    points_c: dict[str, np.ndarray]
    points_g: dict[str, np.ndarray]
    plane_fit: PlaneFit
    scale_result: ScaleResult
    scaled_depth: np.ndarray


def lift_frame(
    depth: np.ndarray,
    floor_mask: np.ndarray,
    other_masks: Mapping[str, np.ndarray],
    K: Union[Intrinsics, np.ndarray],
    mount: MountConfig,
    estimator: ScaleEstimator,
    rng: Optional[np.random.Generator] = None,
    ransac_cfg: Optional[RansacConfig] = None,
) -> LiftResult:
    """Lift 2D depth and masks into 3D metric camera and ground frames.

    1. Backprojects raw floor depth to camera frame C.
    2. Fits ground plane to estimate camera height h_hat and normal n.
    3. Updates metric scale estimator s = h_known / h_hat.
    4. Applies scale s to depth map.
    5. Backprojects scaled depth and transforms to ground frame G.

    Args:
        depth: Raw predicted depth map of shape (H, W).
        floor_mask: Boolean mask of floor pixels of shape (H, W).
        other_masks: Mapping of label names to boolean masks of shape (H, W).
        K: Intrinsics object or 3x3 camera matrix.
        mount: Camera mounting configuration.
        estimator: State of temporal scale estimator.
        rng: Seeded numpy Generator for deterministic plane fitting.
        ransac_cfg: Optional RANSAC configuration.

    Returns:
        LiftResult containing labelled points in C and G, plane fit, and scale.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # 1. Backproject floor points using raw depth
    raw_points_c = backproject(depth, K)
    floor_pts_c = raw_points_c[floor_mask]

    # 2. Fit ground plane
    plane_fit = fit_ground_plane(floor_pts_c, cfg=ransac_cfg, rng=rng)

    # 3. Update scale estimator
    scale_result = estimator.update(plane_fit.h_hat, plane_fit.status)
    s = scale_result.s

    # 4. Apply scale correction to depth
    scaled_depth = depth * s

    # 5. Backproject scaled depth into camera frame C
    scaled_points_c = backproject(scaled_depth, K)

    # 6. Transform to ground frame G
    if plane_fit.status == FitStatus.OK:
        n = plane_fit.n
    else:
        pitch_rad = math.radians(mount.pitch_deg)
        n = np.array([0.0, math.cos(pitch_rad), math.sin(pitch_rad)], dtype=np.float64)

    h_ground = mount.h_known
    scaled_points_g = camera_to_ground(scaled_points_c, n, h_ground)

    # 7. Collect labelled points
    points_c_dict: dict[str, np.ndarray] = {
        "floor": scaled_points_c[floor_mask],
    }
    points_g_dict: dict[str, np.ndarray] = {
        "floor": scaled_points_g[floor_mask],
    }

    for label, mask in other_masks.items():
        points_c_dict[label] = scaled_points_c[mask]
        points_g_dict[label] = scaled_points_g[mask]

    return LiftResult(
        points_c=points_c_dict,
        points_g=points_g_dict,
        plane_fit=plane_fit,
        scale_result=scale_result,
        scaled_depth=scaled_depth,
    )


def object_anchor(
    depth: np.ndarray,
    mask: np.ndarray,
    K: Union[Intrinsics, np.ndarray],
    mount: MountConfig,
    plane: Optional[PlaneFit] = None,
    q: float = 0.10,
    min_pixels: int = 10,
) -> Anchor:
    """Compute 3D ground-frame spatial anchor for an object observation.

    The obstacle distance is computed as the q-th percentile (default 10th)
    of scaled depth inside the mask. The 3D position is the backprojection
    of the 2D mask centroid at that depth, expressed in Ground frame G.

    Args:
        depth: Scaled depth map of shape (H, W).
        mask: Boolean mask of object pixels of shape (H, W).
        K: Intrinsics object or 3x3 camera matrix.
        mount: Camera mounting configuration.
        plane: Optional estimated PlaneFit for ground normal alignment.
        q: Quantile for distance calculation (default: 0.10 for 10th percentile).
        min_pixels: Minimum pixel count to compute an anchor.

    Returns:
        Anchor instance containing distance_m, position_g, and status.
    """
    valid_mask = mask & np.isfinite(depth) & (depth > 0.0)
    pixel_count = int(np.sum(valid_mask))

    if pixel_count == 0:
        return Anchor(
            distance_m=0.0,
            position_g=np.zeros(3, dtype=np.float64),
            status=AnchorStatus.EMPTY_MASK,
        )

    if pixel_count < min_pixels:
        return Anchor(
            distance_m=0.0,
            position_g=np.zeros(3, dtype=np.float64),
            status=AnchorStatus.TINY_MASK,
        )

    depth_values = depth[valid_mask]
    distance_m = float(np.percentile(depth_values, q * 100.0))

    # Mask centroid in image coordinates (u right, v down)
    v_coords, u_coords = np.where(valid_mask)
    u_c = float(np.mean(u_coords))
    v_c = float(np.mean(v_coords))

    # Backproject centroid at distance_m
    fx, fy, cx, cy = _extract_intrinsics(K)
    x_c = (u_c - cx) * distance_m / fx
    y_c = (v_c - cy) * distance_m / fy
    z_c = distance_m
    p_c = np.array([x_c, y_c, z_c], dtype=np.float64)

    # Transform to ground frame G
    if plane is not None and plane.status == FitStatus.OK:
        n = plane.n
    else:
        pitch_rad = math.radians(mount.pitch_deg)
        n = np.array([0.0, math.cos(pitch_rad), math.sin(pitch_rad)], dtype=np.float64)

    position_g = camera_to_ground(p_c, n, mount.h_known)

    return Anchor(
        distance_m=distance_m,
        position_g=position_g,
        status=AnchorStatus.OK,
    )
