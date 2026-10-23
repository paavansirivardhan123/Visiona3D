"""Pinhole camera projection and backprojection operations.

Follows OpenCV convention:
- Camera frame C: x right, y down, z forward.
- Depth d is the z coordinate in C (not Euclidean range).
- Intrinsics K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
"""

from __future__ import annotations

from typing import Union
import numpy as np

from core.config import Intrinsics


def _extract_intrinsics(K: Union[Intrinsics, np.ndarray]) -> tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from Intrinsics instance or 3x3 ndarray."""
    if isinstance(K, Intrinsics):
        return float(K.fx), float(K.fy), float(K.cx), float(K.cy)
    if isinstance(K, np.ndarray) and K.shape == (3, 3):
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    raise TypeError(f"Expected Intrinsics or (3, 3) np.ndarray, got {type(K)}")


def scale_intrinsics(
    K: Union[Intrinsics, np.ndarray],
    sx: float,
    sy: float,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> Intrinsics:
    """Scale intrinsics for a resize and optional letterbox padding.

    Pinhole intrinsics transform under an axis-aligned resize as
    ``fx' = sx * fx``, ``cx' = sx * cx + pad_x`` and likewise for y. The
    letterbox pads are the left/top offsets added by the pad operation.
    The original K remains the K of record; this helper returns the K to use
    with the resized image.

    Args:
        K: Original Intrinsics or 3x3 camera matrix.
        sx: Horizontal scale factor applied to the source image.
        sy: Vertical scale factor applied to the source image.
        pad_x: Left padding in output pixels (0 for a plain resize).
        pad_y: Top padding in output pixels (0 for a plain resize).

    Returns:
        Intrinsics with scaled focal lengths and principal point. When K is an
        Intrinsics instance, width/height describe the padded output image.
    """
    fx, fy, cx, cy = _extract_intrinsics(K)

    if isinstance(K, Intrinsics):
        width = int(round(K.width * sx + 2.0 * pad_x))
        height = int(round(K.height * sy + 2.0 * pad_y))
    else:
        width = int(round(2.0 * (cx * sx + pad_x)))
        height = int(round(2.0 * (cy * sy + pad_y)))

    return Intrinsics(
        fx=fx * sx,
        fy=fy * sy,
        cx=cx * sx + pad_x,
        cy=cy * sy + pad_y,
        width=width,
        height=height,
    )


def project(points_c: np.ndarray, K: Union[Intrinsics, np.ndarray]) -> np.ndarray:
    """Project 3D points in camera frame C to pixel coordinates (u, v).

    Args:
        points_c: Array of shape (..., 3) containing 3D coordinates in C.
        K: Intrinsics object or 3x3 projection matrix.

    Returns:
        Array of shape (..., 2) containing pixel coordinates (u right, v down).
    """
    points = np.asarray(points_c, dtype=np.float64)
    fx, fy, cx, cy = _extract_intrinsics(K)

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # Handle division by zero / negative depth safely
    z_safe = np.where(z != 0.0, z, np.nan)
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy

    return np.stack([u, v], axis=-1)


def backproject(depth: np.ndarray, K: Union[Intrinsics, np.ndarray]) -> np.ndarray:
    """Backproject a 2D depth map (H, W) into 3D points in camera frame C.

    Note: Depth d is the z coordinate in C, not Euclidean range.

    Args:
        depth: Array of shape (H, W) representing depth in metres.
        K: Intrinsics object or 3x3 projection matrix.

    Returns:
        Array of shape (H, W, 3) containing 3D points (x, y, z) in C.
    """
    depth_arr = np.asarray(depth, dtype=np.float64)
    if depth_arr.ndim != 2:
        raise ValueError(f"Depth array must be 2D of shape (H, W), got {depth_arr.shape}")

    h, w = depth_arr.shape
    fx, fy, cx, cy = _extract_intrinsics(K)

    # Broadcast 1-D pixel grids instead of materialising two full H x W
    # index arrays.
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)[:, None]
    x = (u - cx) * depth_arr / fx
    y = (v - cy) * depth_arr / fy
    z = depth_arr

    return np.stack([x, y, z], axis=-1)


def ray_directions(K: Union[Intrinsics, np.ndarray], H: int, W: int) -> np.ndarray:
    """Compute unit ray directions in camera frame C for each pixel.

    Args:
        K: Intrinsics object or 3x3 projection matrix.
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        Array of shape (H, W, 3) with unit direction vectors pointing from camera center.
    """
    fx, fy, cx, cy = _extract_intrinsics(K)
    v_grid, u_grid = np.indices((H, W), dtype=np.float64)

    dx = (u_grid - cx) / fx
    dy = (v_grid - cy) / fy
    dz = np.ones_like(dx)

    rays = np.stack([dx, dy, dz], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / norms


def range_to_depth(range_m: np.ndarray, u: np.ndarray, v: np.ndarray, K: Union[Intrinsics, np.ndarray]) -> np.ndarray:
    """Convert laser rangefinder Euclidean range to camera-frame z-depth.

    A laser rangefinder measures Euclidean range along the pixel ray:
    The 3D point is `range * r / |r|` with `r = K^-1 @ [u, v, 1]`.
    The camera-frame depth is the z-coordinate: `depth = range * r_z / |r|`.

    Args:
        range_m: Euclidean range measurement(s) in metres. Scalar or array broadcastable to u,v.
        u: Pixel x coordinate(s) (u right).
        v: Pixel y coordinate(s) (v down).
        K: Intrinsics object or 3x3 projection matrix.

    Returns:
        Camera-frame z-depth in metres. Same shape as broadcasted inputs.

    Examples:
        >>> import numpy as np
        >>> from core.config import Intrinsics
        >>> K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        >>> # At principal point, ray is (0,0,1), so range == depth
        >>> float(range_to_depth(5.0, 320.0, 240.0, K))
        5.0
        >>> # Off-axis: u=420, v=240 (100px right of center)
        >>> d = range_to_depth(5.0, 420.0, 240.0, K)
        >>> # Ray direction r = [(420-320)/500, 0, 1] = [0.2, 0, 1], |r| = sqrt(1.04)
        >>> # depth = 5.0 * 1.0 / sqrt(1.04) ≈ 4.90196
        >>> bool(np.isclose(d, 5.0 / np.sqrt(1.04), rtol=1e-6))
        True
        >>> # Array inputs
        >>> ranges = np.array([3.0, 4.0, 5.0])
        >>> us = np.array([320.0, 320.0, 320.0])
        >>> vs = np.array([240.0, 240.0, 240.0])
        >>> depths = range_to_depth(ranges, us, vs, K)
        >>> bool(np.allclose(depths, ranges))
        True
    """
    range_arr = np.asarray(range_m, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    fx, fy, cx, cy = _extract_intrinsics(K)

    # Ray vector r = K^-1 @ [u, v, 1] = [(u-cx)/fx, (v-cy)/fy, 1]
    dx = (u_arr - cx) / fx
    dy = (v_arr - cy) / fy
    dz = np.ones_like(dx)

    # Norm of ray vector
    r_norm = np.sqrt(dx * dx + dy * dy + dz * dz)

    # Depth is range * (r_z / |r|) = range * (1 / |r|)
    return range_arr * (dz / r_norm)


def depth_to_range(depth: np.ndarray, u: np.ndarray, v: np.ndarray, K: Union[Intrinsics, np.ndarray]) -> np.ndarray:
    """Convert camera-frame z-depth to laser rangefinder Euclidean range.

    Inverse of range_to_depth: `range = depth * |r| / r_z = depth * |r|`.

    Args:
        depth: Camera-frame z-depth in metres. Scalar or array broadcastable to u,v.
        u: Pixel x coordinate(s) (u right).
        v: Pixel y coordinate(s) (v down).
        K: Intrinsics object or 3x3 projection matrix.

    Returns:
        Euclidean range in metres. Same shape as broadcasted inputs.

    Examples:
        >>> import numpy as np
        >>> from core.config import Intrinsics
        >>> K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        >>> # Round-trip at principal point
        >>> d = 4.0
        >>> r = depth_to_range(d, 320.0, 240.0, K)
        >>> d2 = range_to_depth(r, 320.0, 240.0, K)
        >>> bool(np.isclose(d, d2, rtol=1e-9))
        True
        >>> # Round-trip off-axis
        >>> d = 5.0
        >>> r = depth_to_range(d, 420.0, 240.0, K)
        >>> d2 = range_to_depth(r, 420.0, 240.0, K)
        >>> bool(np.isclose(d, d2, rtol=1e-9))
        True
    """
    depth_arr = np.asarray(depth, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    fx, fy, cx, cy = _extract_intrinsics(K)

    dx = (u_arr - cx) / fx
    dy = (v_arr - cy) / fy
    dz = np.ones_like(dx)

    r_norm = np.sqrt(dx * dx + dy * dy + dz * dz)

    # Range = depth * |r| / r_z = depth * |r| (since r_z = 1)
    return depth_arr * r_norm
