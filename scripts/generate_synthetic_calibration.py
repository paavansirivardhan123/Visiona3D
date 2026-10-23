"""Generate synthetic checkerboard views and a matching synthetic intrinsics file.

This exists to exercise the calibration path and unblock offline demos when no
physical checkerboard capture is available. It is NOT a measurement of a real
camera: the focal length comes from ``--hfov-deg``, and every value written to
``camera_intrinsics.json`` is recovered from rendered views of that assumed
camera. The JSON is tagged ``provenance: synthetic`` and the loader warns when
it is used, because metric distances on real frames are not trustworthy.

Usage::

    python scripts/generate_synthetic_calibration.py \
        --width 1042 --height 500 --hfov-deg 70 \
        --out reports/synthetic_calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import Intrinsics  # noqa: E402
from scripts.calibrate_camera import (  # noqa: E402
    MIN_VIEWS,
    calibrate_from_correspondences,
)

DEFAULT_HFOV_DEG = 70.0  # ASSUMED synthetic field of view, not a measurement.
DEFAULT_SQUARE_M = 0.02
DEFAULT_NOISE_PX = 0.1
DEFAULT_VIEWS = 20
SYNTHETIC_WARNING = (
    "SYNTHETIC calibration of a generated camera; this is NOT a measurement of a "
    "physical camera. Metric distances computed from real frames are not trustworthy."
)


def _parse_board(board: str) -> tuple[int, int]:
    try:
        cols, rows = board.lower().split("x")
        cols_i, rows_i = int(cols), int(rows)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"board must look like '9x6', got {board!r}") from exc
    if cols_i < 2 or rows_i < 2:
        raise ValueError("board dimensions must both be at least 2")
    return cols_i, rows_i


def _object_points(board: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = board
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
    points = (grid + 0.5) * float(square_size_m)
    return np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)


def _pose_for_view(
    K: np.ndarray,
    image_size: tuple[int, int],
    board: tuple[int, int],
    square_size_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a pose whose full board stays inside the image."""
    import cv2  # lazy

    width, height = image_size
    cols, rows = board
    full_w = (cols + 1) * float(square_size_m)
    full_h = (rows + 1) * float(square_size_m)
    tan_x = 0.5 * width / float(K[0, 0])
    tan_y = 0.5 * height / float(K[1, 1])
    z0 = 1.15 * max(0.5 * full_w / tan_x, 0.5 * full_h / tan_y)

    board_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [full_w, 0.0, 0.0],
            [full_w, full_h, 0.0],
            [0.0, full_h, 0.0],
        ],
        dtype=np.float64,
    )

    rvec = rng.uniform(-0.25, 0.25, size=3)
    tvec = np.array(
        [rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), z0 * rng.uniform(1.0, 1.18)],
        dtype=np.float64,
    )
    margin = 6.0
    for _ in range(40):
        projected, _ = cv2.projectPoints(board_corners, rvec, tvec, K, None)
        xy = projected.reshape(-1, 2)
        inside = (
            np.all(xy[:, 0] > margin)
            and np.all(xy[:, 0] < width - margin)
            and np.all(xy[:, 1] > margin)
            and np.all(xy[:, 1] < height - margin)
        )
        if inside:
            return rvec, tvec, board_corners
        tvec[2] *= 1.12
    raise RuntimeError("could not place the synthetic board fully inside the image")


def _render_view(
    K: np.ndarray,
    image_size: tuple[int, int],
    board: tuple[int, int],
    square_size_m: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    import cv2  # lazy

    width, height = image_size
    image = np.full((height, width), 255, dtype=np.uint8)
    cols, rows = board
    square = float(square_size_m)
    for j in range(rows + 1):
        for i in range(cols + 1):
            if (i + j) % 2:
                continue
            quad = np.array(
                [
                    [i * square, j * square, 0.0],
                    [(i + 1) * square, j * square, 0.0],
                    [(i + 1) * square, (j + 1) * square, 0.0],
                    [i * square, (j + 1) * square, 0.0],
                ],
                dtype=np.float64,
            )
            projected, _ = cv2.projectPoints(quad, rvec, tvec, K, None)
            corners = np.round(projected.reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(image, [corners], 0, lineType=cv2.LINE_8)
    return image


def generate_synthetic_calibration(
    width: int,
    height: int,
    *,
    hfov_deg: float = DEFAULT_HFOV_DEG,
    board: tuple[int, int] = (9, 6),
    square_size_m: float = DEFAULT_SQUARE_M,
    views: int = DEFAULT_VIEWS,
    seed: int = 0,
    noise_px: float = DEFAULT_NOISE_PX,
) -> tuple[dict, list[np.ndarray], np.ndarray]:
    """Render synthetic checkerboard views and calibrate the generated camera.

    Returns ``(result, images, reference_K)``. ``result`` is what
    ``calibrate_from_correspondences`` recovered plus synthetic provenance;
    ``reference_K`` is the camera used to render the views.
    """
    import cv2  # lazy

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 10.0 <= float(hfov_deg) <= 170.0:
        raise ValueError("hfov_deg must be between 10 and 170")
    if float(square_size_m) <= 0.0:
        raise ValueError("square_size_m must be positive")
    if int(views) < MIN_VIEWS:
        raise ValueError(
            f"Need at least {MIN_VIEWS} synthetic views; got {views}"
        )
    if float(noise_px) < 0.0:
        raise ValueError("noise_px must be non-negative")

    image_size = (int(width), int(height))
    reference = Intrinsics.from_fov(width, height, float(hfov_deg))
    K = reference.as_matrix()
    object_points = _object_points(board, float(square_size_m))

    rng = np.random.default_rng(int(seed))
    images: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for _ in range(int(views)):
        rvec, tvec, _ = _pose_for_view(
            K, image_size, board, float(square_size_m), rng
        )
        images.append(
            _render_view(K, image_size, board, float(square_size_m), rvec, tvec)
        )
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
        xy = projected.reshape(-1, 2)
        xy = xy + rng.normal(0.0, float(noise_px), size=xy.shape)
        image_points.append(xy)

    result = calibrate_from_correspondences(
        [object_points.copy() for _ in range(int(views))],
        image_points,
        image_size,
        board_size=board,
        square_size_m=float(square_size_m),
        device_label="synthetic",
        min_views=MIN_VIEWS,
    )
    result.update(
        {
            "provenance": "synthetic",
            "synthetic": True,
            "assumed_hfov_deg": float(hfov_deg),
            "pixel_noise_px": float(noise_px),
            "seed": int(seed),
            "warning": SYNTHETIC_WARNING,
            "generator": "scripts/generate_synthetic_calibration.py",
        }
    )
    return result, images, K


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a synthetic checkerboard and write a synthetic intrinsics JSON"
    )
    parser.add_argument("--width", type=int, required=True, help="image width in pixels")
    parser.add_argument("--height", type=int, required=True, help="image height in pixels")
    parser.add_argument(
        "--hfov-deg",
        type=float,
        default=DEFAULT_HFOV_DEG,
        help=f"ASSUMED horizontal field of view (default {DEFAULT_HFOV_DEG:g}; not measured)",
    )
    parser.add_argument("--board", default="9x6", help="inner corners, e.g. 9x6")
    parser.add_argument(
        "--square",
        type=float,
        default=DEFAULT_SQUARE_M,
        help=f"synthetic square size in metres (default {DEFAULT_SQUARE_M:g})",
    )
    parser.add_argument(
        "--views", type=int, default=DEFAULT_VIEWS, help="number of rendered views"
    )
    parser.add_argument("--noise-px", type=float, default=DEFAULT_NOISE_PX)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/synthetic_calibration")
    args = parser.parse_args(argv)

    try:
        board = _parse_board(args.board)
        result, images, _ = generate_synthetic_calibration(
            args.width,
            args.height,
            hfov_deg=args.hfov_deg,
            board=board,
            square_size_m=args.square,
            views=args.views,
            seed=args.seed,
            noise_px=args.noise_px,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for index, image in enumerate(images):
        name = f"checkerboard_{index:02d}.png"
        if not cv2_imwrite(images_dir / name, image):
            print(f"ERROR: could not write {images_dir / name}")
            return 2
        names.append(name)
    result["images_used"] = names
    json_path = out_dir / "camera_intrinsics.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"WARNING: {SYNTHETIC_WARNING}")
    print(
        f"Synthetic calibration for {args.width}x{args.height} at assumed HFOV "
        f"{args.hfov_deg:g} deg"
    )
    print(
        f"fx={result['fx']:.3f} fy={result['fy']:.3f} "
        f"cx={result['cx']:.3f} cy={result['cy']:.3f}"
    )
    print(f"RMS reprojection error: {result['rms_reprojection_error']:.4f} px")
    print(f"Wrote {json_path} and {len(names)} checkerboard view(s) in {images_dir}")
    return 0


def cv2_imwrite(path: Path, image: np.ndarray) -> bool:
    import cv2  # lazy

    return bool(cv2.imwrite(str(path), image))


if __name__ == "__main__":
    sys.exit(main())
