"""RANSAC ground plane fitting with depth-dependent inlier thresholds and SVD refinement.

Follows Visiona3D geometry notation:
- Floor plane in C: { p : n.p = h }, where n is the unit normal pointing DOWN
  toward the floor (n_y > 0) and h > 0 is camera height.
- For pitch theta (positive = tilted down): n = (0, cos(theta), sin(theta)).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional
import numpy as np

from core.config import RansacConfig


class FitStatus(str, Enum):
    """Status outcomes for ground plane fitting."""

    OK = "OK"
    TOO_FEW_POINTS = "TOO_FEW_POINTS"
    LOW_INLIER_RATIO = "LOW_INLIER_RATIO"
    IMPLAUSIBLE_GEOMETRY = "IMPLAUSIBLE_GEOMETRY"


@dataclass(frozen=True)
class PlaneFit:
    """Result of ground plane fitting in camera frame C.

    Attributes:
        n: Unit plane normal vector pointing DOWN toward floor (shape (3,)).
        h_hat: Estimated camera height above floor in metres (h_hat > 0).
        inlier_ratio: Fraction of evaluated points that are inliers.
        n_inliers: Count of inlier points.
        rms_residual: Root-mean-square residual of inliers in metres.
        status: Plane fitting status enum.
    """

    n: np.ndarray
    h_hat: float
    inlier_ratio: float
    n_inliers: int
    rms_residual: float
    status: FitStatus


def fit_ground_plane(
    points_c: np.ndarray,
    cfg: Optional[RansacConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> PlaneFit:
    """Fit ground plane equation n.p = h to 3D points in camera frame C.

    Performs RANSAC over 3-point samples followed by total least squares (SVD)
    refinement on inliers.

    Args:
        points_c: Array of shape (N, 3) or (H, W, 3) of points in camera frame C.
        cfg: RansacConfig parameters. Defaults used if None.
        rng: Seeded numpy Generator for deterministic sampling.

    Returns:
        PlaneFit instance containing estimated normal, height, and status.
    """
    if cfg is None:
        cfg = RansacConfig()
    if rng is None:
        rng = np.random.default_rng(0)

    default_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    pts = np.asarray(points_c, dtype=np.float64)
    if pts.ndim > 2:
        pts = pts.reshape(-1, 3)
    elif pts.ndim == 1 and pts.shape[0] == 3:
        pts = pts.reshape(1, 3)

    # Filter invalid/non-finite points and points behind camera
    valid = np.isfinite(pts).all(axis=-1) & (pts[:, 2] > 0.0)
    pts = pts[valid]

    n_total = len(pts)
    if n_total < cfg.min_points:
        return PlaneFit(
            n=default_normal,
            h_hat=0.0,
            inlier_ratio=0.0,
            n_inliers=n_total,
            rms_residual=0.0,
            status=FitStatus.TOO_FEW_POINTS,
        )

    # Subsample to at most max_points for speed and consistency
    if n_total > cfg.max_points:
        sub_idx = rng.choice(n_total, size=cfg.max_points, replace=False)
        pts_eval = pts[sub_idx]
    else:
        pts_eval = pts
    n_eval = len(pts_eval)

    best_inlier_count = -1
    best_inlier_mask: Optional[np.ndarray] = None
    best_normal = default_normal
    best_h = 0.0

    z_eval = pts_eval[:, 2]
    tau = cfg.tau0 + cfg.tau1 * z_eval

    # RANSAC loop
    for _ in range(cfg.max_iterations):
        sample_indices = rng.choice(n_eval, size=3, replace=False)
        p1, p2, p3 = pts_eval[sample_indices]

        v1 = p2 - p1
        v2 = p3 - p1
        cand_n = np.cross(v1, v2)
        norm = np.linalg.norm(cand_n)
        if norm < 1e-8:
            continue
        cand_n = cand_n / norm

        # Orient normal to point DOWN toward floor (y positive)
        if cand_n[1] < 0.0:
            cand_n = -cand_n

        cand_h = float(np.mean(np.dot(pts_eval[sample_indices], cand_n)))
        if cand_h <= 0.0:
            continue

        res = np.abs(np.dot(pts_eval, cand_n) - cand_h)
        inlier_mask = res < tau
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_normal = cand_n
            best_h = cand_h

    if best_inlier_mask is None or best_inlier_count < 3:
        return PlaneFit(
            n=default_normal,
            h_hat=0.0,
            inlier_ratio=0.0,
            n_inliers=0,
            rms_residual=0.0,
            status=FitStatus.LOW_INLIER_RATIO,
        )

    # Refine using SVD on inliers
    inlier_pts = pts_eval[best_inlier_mask]
    centroid = np.mean(inlier_pts, axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    refined_n = vh[-1, :]
    refined_norm = np.linalg.norm(refined_n)
    if refined_norm > 1e-8:
        refined_n = refined_n / refined_norm
    else:
        refined_n = best_normal

    # Orient normal down (y > 0)
    if refined_n[1] < 0.0:
        refined_n = -refined_n

    refined_h = float(np.dot(centroid, refined_n))

    # Evaluate refined plane on evaluation points
    final_res = np.abs(np.dot(pts_eval, refined_n) - refined_h)
    final_inliers = final_res < tau
    n_inliers = int(np.sum(final_inliers))
    inlier_ratio = float(n_inliers / n_eval)
    rms_residual = (
        float(np.sqrt(np.mean(final_res[final_inliers] ** 2)))
        if n_inliers > 0
        else 0.0
    )

    # Check plausibility: pitch angle and positive height
    pitch_rad = math.atan2(refined_n[2], refined_n[1])
    pitch_deg = math.degrees(pitch_rad)

    if refined_h <= 0.0 or abs(pitch_deg) > cfg.max_pitch_deg or refined_n[1] <= 0.0:
        status = FitStatus.IMPLAUSIBLE_GEOMETRY
    elif inlier_ratio < cfg.min_inlier_ratio:
        status = FitStatus.LOW_INLIER_RATIO
    else:
        status = FitStatus.OK

    return PlaneFit(
        n=refined_n,
        h_hat=refined_h,
        inlier_ratio=inlier_ratio,
        n_inliers=n_inliers,
        rms_residual=rms_residual,
        status=status,
    )
