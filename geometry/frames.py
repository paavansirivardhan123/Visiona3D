"""Coordinate frame transformations between Camera frame C and Ground frame G.

Follows Visiona3D geometry notation:
- Camera frame C (OpenCV convention): x right, y down, z forward.
- Floor plane in C: { p : n.p = h }, where n is unit normal pointing DOWN toward floor
  and h > 0 is camera height.
- Ground frame G (gravity-aligned, origin on floor directly below camera):
  X right, Y forward (horizontal), Z up (height above floor).
  Basis vectors in C:
    e_up = -n
    e_fwd = normalize(z_c - (z_c.e_up) e_up) with z_c = (0, 0, 1)
    e_right = e_fwd x e_up
  Coordinates in G:
    X = e_right.p
    Y = e_fwd.p
    Z = h - n.p
"""

from __future__ import annotations

import numpy as np


def ground_frame_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the orthonormal basis (e_right, e_fwd, e_up) of Ground frame G in C.

    Args:
        n: Unit normal vector pointing DOWN toward the floor (shape (3,)).

    Returns:
        Tuple of (e_right, e_fwd, e_up), each shape (3,).
    """
    n_unit = np.asarray(n, dtype=np.float64) / np.linalg.norm(n)
    e_up = -n_unit

    z_c = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fwd_proj = z_c - np.dot(z_c, e_up) * e_up
    fwd_norm = np.linalg.norm(fwd_proj)
    if fwd_norm < 1e-8:
        # Singularity: camera pointing straight up or down; fallback to standard horizontal
        e_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        e_fwd = fwd_proj / fwd_norm

    e_right = np.cross(e_fwd, e_up)
    e_right = e_right / np.linalg.norm(e_right)

    return e_right, e_fwd, e_up


def camera_to_ground(points_c: np.ndarray, n: np.ndarray, h: float) -> np.ndarray:
    """Transform points from camera frame C to gravity-aligned ground frame G.

    Args:
        points_c: Array of shape (..., 3) containing 3D points in C.
        n: Unit normal vector pointing DOWN toward the floor (shape (3,)).
        h: Camera height above ground in metres (h > 0).

    Returns:
        Array of shape (..., 3) containing coordinates [X, Y, Z] in G.
    """
    pts = np.asarray(points_c, dtype=np.float64)
    e_right, e_fwd, e_up = ground_frame_basis(n)
    n_unit = -e_up

    # Single matrix product instead of three full-array dot products.  Rows
    # are the ground axes expressed in C, so [X, Y, Z - h] = R @ p.
    basis = np.stack([e_right, e_fwd, -n_unit], axis=0)
    flat = pts.reshape(-1, 3)
    out = flat @ basis.T
    out[:, 2] += h
    return out.reshape(pts.shape)


def ground_to_camera(points_g: np.ndarray, n: np.ndarray, h: float) -> np.ndarray:
    """Transform points from ground frame G back to camera frame C.

    Inverse of camera_to_ground.

    Args:
        points_g: Array of shape (..., 3) containing coordinates [X, Y, Z] in G.
        n: Unit normal vector pointing DOWN toward the floor (shape (3,)).
        h: Camera height above ground in metres (h > 0).

    Returns:
        Array of shape (..., 3) containing coordinates in C.
    """
    pts = np.asarray(points_g, dtype=np.float64)
    e_right, e_fwd, e_up = ground_frame_basis(n)

    x = pts[..., 0:1]
    y = pts[..., 1:2]
    z = pts[..., 2:3]

    # p = h * n + X * e_right + Y * e_fwd + Z * e_up
    origin_c = h * (-e_up)
    pts_c = origin_c + x * e_right + y * e_fwd + z * e_up
    return pts_c
