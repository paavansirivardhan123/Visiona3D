"""Estimated session trail from floor-plane optical flow.

DISPLAY ONLY. Nothing in this module feeds guidance, certification or
planning. The map picture is a convenience view, not a navigation input.

This project has no IMU or wheel odometry (see ``mapping/rolling.py``).
Camera motion is estimated from the frames the pipeline already processes.
The camera walks on the ground, so the dominant plane in view is the floor and
its distance is known from the lift's ground-plane fit. Between two frames the
floor induces a homography::

    H = R_y(-yaw) @ (I - C n^T / h),    C = [dx, 0, dz]

with ``n`` the floor normal (pointing down), ``h`` the perpendicular camera
height and ``C`` the camera displacement in the previous camera frame (``dx``
right, ``dz`` forward). For a fixed yaw the homography is linear in ``dx`` and
``dz``, so the step is recovered by robustly weighted least squares over tracked
floor features, with yaw scanned and refined. The metric scale anchor is ``h``
alone; monocular depth is not used for translation. Because ``dx`` is fitted
rather than folded into yaw, the trail follows the direction actually travelled
even while the camera pans. The estimate still drifts, assumes a flat floor and
ignores lens distortion, so it is an estimate and not a measurement.

Conventions match ``Odometry2D`` in the ground frame G: X right, Y forward,
positive yaw = clockwise. Camera coordinates follow OpenCV: X right, Y down,
Z forward. A step is rejected when tracking is weak, the fit is poor, or the
motion would exceed a plausible walking speed or turn rate for the elapsed time
between the two frames. The trail then keeps its last known point and records a
gap instead of inventing motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

from core.config import Intrinsics
from mapping.rolling import Odometry2D

# Starting values only; none of these is validated on real data.
MAX_FEATURES: int = 900  # TO TUNE ON REAL DATA
FEATURE_QUALITY: float = 0.005  # TO TUNE ON REAL DATA
FEATURE_MIN_DISTANCE_PX: float = 5.0  # TO TUNE ON REAL DATA
FLOOR_MASK_DILATE_PX: int = 9  # TO TUNE ON REAL DATA
LK_WINDOW_PX: int = 31  # TO TUNE ON REAL DATA
LK_MAX_LEVEL: int = 5  # TO TUNE ON REAL DATA
MIN_TRACKED: int = 60  # TO TUNE ON REAL DATA
MIN_PLANE_HEIGHT_M: float = 0.2  # TO TUNE ON REAL DATA
MAX_PLANE_HEIGHT_M: float = 3.0  # TO TUNE ON REAL DATA
MAX_MEDIAN_ERR_PX: float = 10.0  # TO TUNE ON REAL DATA
MIN_ERR_GAIN_PX: float = 1.0  # TO TUNE ON REAL DATA
MAX_STEP_M: float = 2.0  # TO TUNE ON REAL DATA: fixed fallback when elapsed time is unknown
MAX_STEP_YAW_DEG: float = 20.0  # TO TUNE ON REAL DATA: fixed fallback when elapsed time is unknown
MAX_SPEED_MPS: float = 2.5  # TO TUNE ON REAL DATA: plausible walking speed for the time-scaled step gate
MAX_YAW_RATE_DPS: float = 60.0  # TO TUNE ON REAL DATA: plausible turn rate for the time-scaled yaw gate
SEARCH_YAW_DEG: float = 12.0  # TO TUNE ON REAL DATA: fixed fallback yaw search half-range
SEARCH_YAW_CAP_DEG: float = 45.0  # TO TUNE ON REAL DATA: widest yaw search as elapsed time grows
COARSE_YAW_DEG: float = 1.0  # TO TUNE ON REAL DATA
FINE_YAW_DEG: float = 0.1  # TO TUNE ON REAL DATA
IRLS_C_PX: float = 4.0  # TO TUNE ON REAL DATA
IRLS_C_START_PX: float = 40.0  # TO TUNE ON REAL DATA
IRLS_STEPS: int = 4  # TO TUNE ON REAL DATA
MAX_TRACE_GAP_S: float = 8.0  # TO TUNE ON REAL DATA: longer gaps restart the trail reference
TRACE_MIN_BASELINE_S: float = 0.5  # TO TUNE ON REAL DATA: short baselines are unmeasurable
MAX_MISSES_BEFORE_RESET: int = 3  # TO TUNE ON REAL DATA


@dataclass(frozen=True)
class FloorPlane:
    """Ground plane in the camera frame: ``normal . p = height_m``.

    ``normal`` is a unit vector pointing down toward the floor, matching
    ``geometry.ground_plane.PlaneFit``. ``height_m`` is the metric
    perpendicular distance from the camera, i.e. the fitted ``h_hat`` times the
    metric scale factor.
    """

    normal: np.ndarray
    height_m: float


def _camera_matrix(K: Intrinsics) -> np.ndarray:
    return np.array(
        [
            [float(K.fx), 0.0, float(K.cx)],
            [0.0, float(K.fy), float(K.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rot_y(deg: float) -> np.ndarray:
    angle = math.radians(float(deg))
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64
    )


def _valid_plane(plane: FloorPlane) -> bool:
    normal = np.asarray(plane.normal, dtype=np.float64)
    height = float(plane.height_m)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        return False
    if abs(float(np.linalg.norm(normal)) - 1.0) > 1e-3:
        return False
    if not math.isfinite(height):
        return False
    return MIN_PLANE_HEIGHT_M <= height <= MAX_PLANE_HEIGHT_M


def _rays(pixels: np.ndarray, Kinv: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [pixels, np.ones((pixels.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (Kinv @ homogeneous.T).T


def _errors(
    rays: np.ndarray, dst: np.ndarray, H: np.ndarray, Km: np.ndarray
) -> np.ndarray:
    """Per-point pixel distance between ``H`` applied to rays and ``dst``."""
    projected = (rays @ H.T) @ Km.T
    z = projected[:, 2]
    uv = np.zeros((projected.shape[0], 2), dtype=np.float64)
    ok = z > 1e-6
    uv[ok] = projected[ok, :2] / z[ok, None]
    err = np.linalg.norm(uv - dst, axis=1)
    err[~ok] = np.inf  # a ray behind the camera contradicts the model
    return err


def _median_error(
    rays: np.ndarray, dst: np.ndarray, H: np.ndarray, Km: np.ndarray
) -> float:
    err = _errors(rays, dst, H, Km)
    if not np.isfinite(err).any():
        return float("inf")
    return float(np.median(err))


def _floor_homography(
    plane: FloorPlane, dx: float, dz: float, yaw_deg: float
) -> np.ndarray:
    n = np.asarray(plane.normal, dtype=np.float64)
    h = float(plane.height_m)
    C = np.array([float(dx), 0.0, float(dz)], dtype=np.float64)
    return _rot_y(-float(yaw_deg)) @ (np.eye(3) - np.outer(C, n) / h)


def _solve_translation(
    rays: np.ndarray,
    dst: np.ndarray,
    plane: FloorPlane,
    Km: np.ndarray,
    yaw_deg: float,
    weights: np.ndarray,
) -> Optional[tuple[float, float]]:
    """Least-squares (dx, dz) at a fixed yaw.

    With ``H = R (I - C n^T / h)`` and ``C = [dx, 0, dz]``, write ``u = R r``,
    ``a = n.r / h`` and ``w = R C``. Then ``H r = u - a w`` and every tracked
    point gives two equations that are linear in ``w_x`` and ``w_z``::

        a p_x w_z - a fx w_x = p_x u_z - fx u_x
                       a p_y w_z = p_y u_z - fy u_y

    with ``p`` the destination pixel minus the principal point. Solving them
    weights every point, unlike a median, which is nearly flat along the
    yaw/lateral trade-off and cannot separate a sideways step from a pan.
    """
    n = np.asarray(plane.normal, dtype=np.float64)
    h = float(plane.height_m)
    fx, fy = float(Km[0, 0]), float(Km[1, 1])
    cx, cy = float(Km[0, 2]), float(Km[1, 2])
    R = _rot_y(-float(yaw_deg))
    u = rays @ R.T
    a = (rays @ n) / h
    px = dst[:, 0] - cx
    py = dst[:, 1] - cy

    ax = -a * fx
    az = a * px
    b1 = px * u[:, 2] - fx * u[:, 0]
    az2 = a * py
    b2 = py * u[:, 2] - fy * u[:, 1]

    m00 = float(np.sum(weights * ax * ax))
    m01 = float(np.sum(weights * ax * az))
    m11 = float(np.sum(weights * (az * az + az2 * az2)))
    r0 = float(np.sum(weights * ax * b1))
    r1 = float(np.sum(weights * (az * b1 + az2 * b2)))
    det = m00 * m11 - m01 * m01
    if not math.isfinite(det) or abs(det) < 1e-12:
        return None
    wx = (m11 * r0 - m01 * r1) / det
    wz = (m00 * r1 - m01 * r0) / det
    C = R.T @ np.array([wx, 0.0, wz], dtype=np.float64)
    if not np.isfinite(C).all():
        return None
    return float(C[0]), float(C[2])


def _fit_at_yaw(
    rays: np.ndarray,
    dst: np.ndarray,
    plane: FloorPlane,
    Km: np.ndarray,
    yaw_deg: float,
) -> Optional[tuple[float, float, float, np.ndarray]]:
    """Robustly weighted (dx, dz) at a fixed yaw.

    The rejection threshold starts wide and narrows to ``IRLS_C_PX`` so the
    fit can escape the bias that gross tracking errors put on the plain
    least-squares start; a hard threshold applied straight away would keep only
    the points that happen to agree with that biased start.

    Returns ``(cost, dx, dz, per_point_errors)`` where ``cost`` is the sum of
    squared errors truncated at ``IRLS_C_PX``, so it is comparable across yaws.
    """
    weights = np.ones(rays.shape[0], dtype=np.float64)
    solution = _solve_translation(rays, dst, plane, Km, yaw_deg, weights)
    if solution is None:
        return None
    for c in np.geomspace(float(IRLS_C_START_PX), float(IRLS_C_PX), int(IRLS_STEPS)):
        err = _errors(
            rays, dst, _floor_homography(plane, solution[0], solution[1], yaw_deg), Km
        )
        weights = np.where(err < c, (1.0 - (err / c) ** 2) ** 2, 0.0)
        refined = _solve_translation(rays, dst, plane, Km, yaw_deg, weights)
        if refined is None:
            break
        solution = refined
    err = _errors(
        rays, dst, _floor_homography(plane, solution[0], solution[1], yaw_deg), Km
    )
    cost = float(np.sum(np.minimum(err * err, IRLS_C_PX * IRLS_C_PX)))
    return cost, solution[0], solution[1], err


def _motion_gates(
    dt_s: Optional[float],
    max_step_m: float,
    max_step_yaw_deg: float,
) -> tuple[float, float, float, float]:
    """Return (step limit, yaw limit, search half-range, coarse yaw step).

    With an elapsed time the gates grow at a plausible walking speed and turn
    rate, so slow processing (large gaps between handled frames) does not look
    like an impossible jump. Without a usable time the fixed starting values
    apply. The yaw gate never exceeds the range the scan actually searches, and
    the coarse step is widened with the range so the scan keeps its cost.
    """
    limit_m = float(max_step_m)
    limit_yaw = float(max_step_yaw_deg)
    search_yaw = float(SEARCH_YAW_DEG)
    coarse = float(COARSE_YAW_DEG)
    if dt_s is None:
        return limit_m, limit_yaw, search_yaw, coarse
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        return limit_m, limit_yaw, search_yaw, coarse
    limit_m = max(limit_m, float(MAX_SPEED_MPS) * dt)
    rate_yaw = min(float(MAX_YAW_RATE_DPS) * dt, float(SEARCH_YAW_CAP_DEG))
    limit_yaw = max(limit_yaw, rate_yaw)
    search_yaw = min(max(search_yaw, rate_yaw), float(SEARCH_YAW_CAP_DEG))
    coarse = max(coarse, search_yaw / (float(SEARCH_YAW_DEG) / float(COARSE_YAW_DEG)))
    return limit_m, limit_yaw, search_yaw, coarse


def fit_floor_motion(
    src_px: np.ndarray,
    dst_px: np.ndarray,
    plane: FloorPlane,
    K: Intrinsics,
    *,
    min_tracked: int = MIN_TRACKED,
    max_median_err_px: float = MAX_MEDIAN_ERR_PX,
    min_err_gain_px: float = MIN_ERR_GAIN_PX,
    max_step_m: float = MAX_STEP_M,
    max_step_yaw_deg: float = MAX_STEP_YAW_DEG,
    dt_s: Optional[float] = None,
) -> Optional[Odometry2D]:
    """Fit lateral step, forward step and yaw that explain tracked floor motion.

    Returns the accepted step in ground-frame axes (X right, Y forward,
    clockwise yaw), or None when the correspondences are too few, the plane is
    implausible, or the fitted motion does not beat the standing-still model.

    ``dt_s`` is the elapsed time between the two frames. When it is usable the
    motion gates and the yaw search widen with it; when it is None the fixed
    fallback limits apply.
    """
    src = np.asarray(src_px, dtype=np.float64)
    dst = np.asarray(dst_px, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or dst.ndim != 2 or dst.shape[1] != 2:
        return None
    if src.shape[0] != dst.shape[0] or src.shape[0] < min_tracked:
        return None
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        return None
    if not _valid_plane(plane):
        return None

    limit_m, limit_yaw, search_yaw, coarse = _motion_gates(
        dt_s, max_step_m, max_step_yaw_deg
    )
    Km = _camera_matrix(K)
    rays = _rays(src, np.linalg.inv(Km))
    still_err = _median_error(rays, dst, np.eye(3), Km)

    best: Optional[tuple[float, float, float, float, np.ndarray]] = None
    for yaw in np.arange(-search_yaw, search_yaw + 1e-9, coarse):
        fit = _fit_at_yaw(rays, dst, plane, Km, float(yaw))
        if fit is not None and (best is None or fit[0] < best[0]):
            best = (fit[0], float(yaw), fit[1], fit[2], fit[3])
    if best is None:
        return None

    step = float(coarse)
    while step > float(FINE_YAW_DEG) + 1e-9:
        for yaw in (best[1] - step, best[1] + step):
            if abs(yaw) > float(search_yaw) + 1e-9:
                continue
            fit = _fit_at_yaw(rays, dst, plane, Km, yaw)
            if fit is not None and fit[0] < best[0]:
                best = (fit[0], yaw, fit[1], fit[2], fit[3])
        step *= 0.5

    _, yaw, dx, dz, per_point = best
    err = float(np.median(per_point))

    if not math.isfinite(err) or err > float(max_median_err_px):
        return None
    if still_err - err < float(min_err_gain_px):
        return None
    if math.hypot(float(dx), float(dz)) > float(limit_m):
        return None
    if abs(float(yaw)) > float(limit_yaw):
        return None
    return Odometry2D(dx=float(dx), dy=float(dz), dyaw_deg=float(yaw))


def track_floor_features(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    floor_mask: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Track floor features from the previous frame into the current one."""
    prev = np.asarray(prev_gray)
    curr = np.asarray(curr_gray)
    mask = np.asarray(floor_mask)
    if prev.ndim != 2 or curr.ndim != 2 or prev.shape != curr.shape:
        return None
    if mask.shape != prev.shape:
        return None
    binary = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if not binary.any():
        return None
    size = int(FLOOR_MASK_DILATE_PX)
    if size > 0:
        binary = cv2.dilate(binary, np.ones((size, size), dtype=np.uint8))

    points = cv2.goodFeaturesToTrack(
        prev,
        maxCorners=int(MAX_FEATURES),
        qualityLevel=float(FEATURE_QUALITY),
        minDistance=float(FEATURE_MIN_DISTANCE_PX),
        mask=binary * 255,
    )
    if points is None or len(points) < MIN_TRACKED:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        prev,
        curr,
        points,
        None,
        winSize=(int(LK_WINDOW_PX), int(LK_WINDOW_PX)),
        maxLevel=int(LK_MAX_LEVEL),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return None
    keep = status.reshape(-1) == 1
    src = points.reshape(-1, 2)[keep]
    dst = tracked.reshape(-1, 2)[keep]
    height, width = prev.shape
    inside = (
        (dst[:, 0] > 0.0)
        & (dst[:, 0] < float(width))
        & (dst[:, 1] > 0.0)
        & (dst[:, 1] < float(height))
    )
    src, dst = src[inside], dst[inside]
    if len(src) < MIN_TRACKED:
        return None
    return src, dst


def estimate_floor_step(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_floor_mask: np.ndarray,
    plane: FloorPlane,
    K: Intrinsics,
    *,
    dt_s: Optional[float] = None,
) -> Optional[Odometry2D]:
    """Track floor features and fit the metric floor-plane step."""
    tracked = track_floor_features(prev_gray, curr_gray, prev_floor_mask)
    if tracked is None:
        return None
    return fit_floor_motion(tracked[0], tracked[1], plane, K, dt_s=dt_s)


@dataclass
class SessionTrace:
    """Accumulate estimated camera positions in a frame fixed at session start."""

    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0
    distance_m: float = 0.0
    steps: int = 0
    misses: int = 0
    gaps: int = 0

    def __post_init__(self) -> None:
        self.points: list[tuple[float, float]] = [(0.0, 0.0)]
        self._prev_gray: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._prev_plane: FloorPlane | None = None
        self._prev_ts: float | None = None

    @property
    def position(self) -> tuple[float, float]:
        return (float(self.x), float(self.y))

    def reset_reference(self) -> None:
        self._prev_gray = None
        self._prev_mask = None
        self._prev_plane = None
        self._prev_ts = None

    def _store(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        plane: FloorPlane,
        ts: float | None,
    ) -> None:
        self._prev_gray = np.asarray(gray).copy()
        self._prev_mask = np.asarray(mask).astype(bool, copy=True)
        self._prev_plane = FloorPlane(
            normal=np.asarray(plane.normal, dtype=np.float64).copy(),
            height_m=float(plane.height_m),
        )
        self._prev_ts = None if ts is None else float(ts)

    def _drop_reference(self) -> None:
        self._prev_gray = None
        self._prev_mask = None
        self._prev_plane = None
        self._prev_ts = None
        self.misses = 0

    def update_frame(
        self,
        gray: np.ndarray,
        plane: Optional[FloorPlane],
        floor_mask: Optional[np.ndarray],
        K: Intrinsics,
        *,
        ts: float | None = None,
    ) -> Optional[Odometry2D]:
        """Add one processed frame. Returns the accepted step, or None."""
        if plane is None or floor_mask is None or not _valid_plane(plane):
            self.gaps += 1
            self.misses += 1
            if self.misses >= MAX_MISSES_BEFORE_RESET:
                self._drop_reference()
            return None
        if self._prev_gray is None or self._prev_mask is None or self._prev_plane is None:
            self._store(gray, floor_mask, plane, ts)
            return None

        dt: float | None = None
        if ts is not None and self._prev_ts is not None:
            elapsed = float(ts) - self._prev_ts
            if elapsed > float(MAX_TRACE_GAP_S):
                self.gaps += 1
                self.misses += 1
                self._store(gray, floor_mask, plane, ts)
                return None
            if math.isfinite(elapsed) and elapsed > 0.0:
                dt = elapsed

        if dt is not None and dt < float(TRACE_MIN_BASELINE_S):
            # Adjacent high-rate frames move only a pixel or two; the standing
            # still model wins and no step would ever be accepted. Keep the
            # older reference and wait for a baseline long enough to measure.
            return None

        step = estimate_floor_step(
            self._prev_gray,
            gray,
            self._prev_mask,
            self._prev_plane,
            K,
            dt_s=dt,
        )
        self._store(gray, floor_mask, plane, ts)
        if step is None:
            self.gaps += 1
            self.misses += 1
            if self.misses >= MAX_MISSES_BEFORE_RESET:
                self._drop_reference()
            return None

        heading = math.radians(self.heading_deg)
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        # The step is expressed in the previous camera frame; rotate it into the world.
        world_dx = cos_h * float(step.dx) + sin_h * float(step.dy)
        world_dy = -sin_h * float(step.dx) + cos_h * float(step.dy)
        self.x += world_dx
        self.y += world_dy
        self.heading_deg = (
            self.heading_deg + math.degrees(float(step.get_yaw_rad()))
        ) % 360.0
        self.distance_m += math.hypot(float(step.dx), float(step.dy))
        self.steps += 1
        self.misses = 0
        self.points.append((float(self.x), float(self.y)))
        return step


__all__ = [
    "FloorPlane",
    "SessionTrace",
    "estimate_floor_step",
    "fit_floor_motion",
    "track_floor_features",
]
