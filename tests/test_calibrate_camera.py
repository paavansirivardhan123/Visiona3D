"""P10: checkerboard calibration from synthetic correspondences (no images)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.calibrate_camera import calibrate_from_correspondences


TRUE_K = np.array(
    [[800.0, 0.0, 320.0], [0.0, 780.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
DIST = np.array([-0.12, 0.04, 0.001, -0.0005, 0.0], dtype=np.float64)
IMAGE_SIZE = (640, 480)
BOARD = (9, 6)
SQUARE = 0.025


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def _project(object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    points_c = object_points @ _rodrigues(rvec).T + tvec
    x = points_c[:, 0] / points_c[:, 2]
    y = points_c[:, 1] / points_c[:, 2]
    r2 = x * x + y * y
    radial = 1.0 + DIST[0] * r2 + DIST[1] * r2**2 + DIST[4] * r2**3
    xd = x * radial + 2.0 * DIST[2] * x * y + DIST[3] * (r2 + 2.0 * x * x)
    yd = y * radial + DIST[2] * (r2 + 2.0 * y * y) + 2.0 * DIST[3] * x * y
    u = TRUE_K[0, 0] * xd + TRUE_K[0, 2]
    v = TRUE_K[1, 1] * yd + TRUE_K[1, 2]
    return np.stack([u, v], axis=-1)


def _synthetic_views() -> tuple[list[np.ndarray], list[np.ndarray]]:
    grid = np.mgrid[0 : BOARD[0], 0 : BOARD[1]].T.reshape(-1, 2).astype(np.float64)
    object_points = np.concatenate(
        [grid * SQUARE, np.zeros((grid.shape[0], 1))], axis=1
    )

    rvecs = [
        (0.10, 0.05, 0.02),
        (-0.12, 0.08, -0.03),
        (0.20, -0.10, 0.05),
        (-0.05, 0.15, 0.08),
        (0.08, -0.18, -0.06),
        (0.25, 0.12, 0.10),
        (-0.22, -0.06, 0.03),
        (0.03, 0.22, -0.09),
        (-0.15, 0.02, 0.11),
        (0.18, 0.18, 0.04),
        (-0.08, -0.20, -0.02),
        (0.12, 0.03, -0.12),
    ]
    translations = [
        (-0.18, -0.12, 0.75),
        (0.10, -0.08, 0.70),
        (-0.05, 0.10, 0.80),
        (0.15, 0.05, 0.65),
        (0.00, -0.15, 0.85),
        (-0.20, 0.08, 0.72),
        (0.18, -0.02, 0.78),
        (0.05, 0.18, 0.68),
        (-0.10, 0.00, 0.82),
        (0.12, 0.12, 0.74),
        (-0.16, -0.10, 0.76),
        (0.02, 0.06, 0.90),
    ]
    image_points = [
        _project(object_points, np.asarray(rvec), np.asarray(tvec))
        for rvec, tvec in zip(rvecs, translations)
    ]
    return [object_points.copy() for _ in range(len(image_points))], image_points


def test_p10_recovers_intrinsics_within_half_percent():
    object_points, image_points = _synthetic_views()
    result = calibrate_from_correspondences(
        object_points,
        image_points,
        IMAGE_SIZE,
        board_size=BOARD,
        square_size_m=SQUARE,
        device_label="synthetic",
    )

    assert abs(result["fx"] - TRUE_K[0, 0]) / TRUE_K[0, 0] < 0.005
    assert abs(result["fy"] - TRUE_K[1, 1]) / TRUE_K[1, 1] < 0.005
    assert abs(result["cx"] - TRUE_K[0, 2]) / TRUE_K[0, 2] < 0.005
    assert abs(result["cy"] - TRUE_K[1, 2]) / TRUE_K[1, 2] < 0.005
    assert result["rms_reprojection_error"] >= 0.0
    assert len(result["per_view_errors"]) == len(image_points)
    assert result["board"]["views_used"] == len(image_points)
    assert result["image_size"] == list(IMAGE_SIZE)


def test_p10_fewer_than_ten_views_clear_error():
    object_points, image_points = _synthetic_views()
    with pytest.raises(ValueError, match="at least 10 usable views"):
        calibrate_from_correspondences(
            object_points[:9], image_points[:9], IMAGE_SIZE, board_size=BOARD
        )
