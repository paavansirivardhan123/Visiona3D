"""Checkerboard camera calibration tool.

CLI usage:
    python scripts/calibrate_camera.py --images calib/ --board 9x6 --square 0.025 \
        --out reports/camera_intrinsics.json

OpenCV is imported lazily and only by this script. ``calibrate_from_correspondences``
accepts ready-made 2D-3D correspondences so calibration can be tested without
any image files.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Sequence
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIN_VIEWS = 10
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm"}


def _parse_board(board: str) -> tuple[int, int]:
    try:
        cols, rows = board.lower().split("x")
        cols_i, rows_i = int(cols), int(rows)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"board must look like '9x6', got {board!r}") from exc
    if cols_i < 2 or rows_i < 2:
        raise ValueError("board dimensions must both be at least 2")
    return cols_i, rows_i


def calibrate_from_correspondences(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    board_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
    device_label: str = "unknown",
    min_views: int = MIN_VIEWS,
    flags: int | None = None,
) -> dict[str, Any]:
    """Calibrate intrinsics from ready-made checkerboard correspondences.

    Args:
        object_points: Sequence of (N, 3) board coordinates in metres.
        image_points: Sequence of (N, 1, 2) or (N, 2) detected pixel corners.
        image_size: (width, height) of the source images.
        board_size: Checkerboard inner corners (columns, rows).
        square_size_m: Square edge length in metres.
        device_label: Human-readable camera/device label.
        min_views: Minimum usable views required.
        flags: cv2.calibrateCamera flags (default 0).

    Returns:
        Dict with fx, fy, cx, cy, distortion, image_size, RMS reprojection
        error, per-view errors, board config, date and device label.

    Raises:
        ValueError: If fewer than ``min_views`` views are supplied.
    """
    import cv2  # lazy

    n_views = len(object_points)
    if n_views != len(image_points):
        raise ValueError(
            f"object_points and image_points must have equal length; got {n_views} and {len(image_points)}"
        )
    if n_views < min_views:
        raise ValueError(
            f"Need at least {min_views} usable views for a stable calibration; got {n_views}. "
            "Capture more checkerboard views with varied orientation and distance."
        )

    obj = [np.asarray(p, dtype=np.float32).reshape(-1, 3) for p in object_points]
    img = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points]
    if any(p.shape[0] < 4 for p in obj):
        raise ValueError("Each view needs at least 4 correspondences")

    width, height = int(image_size[0]), int(image_size[1])
    calib_flags = 0 if flags is None else int(flags)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, (width, height), None, None, flags=calib_flags
    )

    per_view_errors: list[float] = []
    for view_obj, view_img, rvec, tvec in zip(obj, img, rvecs, tvecs):
        projected, _ = cv2.projectPoints(view_obj, rvec, tvec, K, dist)
        error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - view_img.reshape(-1, 2), axis=1)))
        per_view_errors.append(error)

    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion": np.asarray(dist, dtype=np.float64).reshape(-1).tolist(),
        "image_size": [width, height],
        "rms_reprojection_error": float(rms),
        "per_view_errors": per_view_errors,
        "board": {
            "inner_corners": [int(board_size[0]), int(board_size[1])],
            "square_size_m": float(square_size_m),
            "views_used": int(n_views),
        },
        "date": datetime.datetime.now().isoformat(),
        "device": str(device_label),
        "camera_matrix": np.asarray(K, dtype=np.float64).tolist(),
    }


def find_correspondences(
    image_dir: str | Path,
    board_size: tuple[int, int],
    square_size_m: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
    """Detect checkerboard corners in every image of a folder."""
    import cv2  # lazy

    folder = Path(image_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"image folder not found: {folder}")

    pattern = (int(board_size[0]), int(board_size[1]))
    objp = np.zeros((pattern[0] * pattern[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * float(square_size_m)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    used: list[str] = []
    rejected: list[str] = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected.append(f"{path.name}: could not read")
            continue
        found, corners = cv2.findChessboardCorners(image, pattern, None)
        if not found:
            rejected.append(f"{path.name}: board not found")
            continue
        corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp.copy())
        img_points.append(corners)
        used.append(path.name)

    return obj_points, img_points, used, rejected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkerboard camera calibration")
    parser.add_argument("--images", required=True, help="folder of checkerboard images")
    parser.add_argument("--board", default="9x6", help="inner corners, e.g. 9x6")
    parser.add_argument("--square", type=float, default=0.025, help="square size in metres")
    parser.add_argument("--out", default="camera_intrinsics.json", help="output JSON path")
    parser.add_argument("--device", default="unknown", help="camera/device label")
    parser.add_argument("--min-views", type=int, default=MIN_VIEWS)
    args = parser.parse_args(argv)

    try:
        board_size = _parse_board(args.board)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        obj_points, img_points, used, rejected = find_correspondences(
            args.images, board_size, args.square
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Usable views: {len(used)} (rejected {len(rejected)})")
    for name in rejected:
        print(f"  rejected: {name}")
    if len(used) < args.min_views:
        print(
            f"ERROR: Need at least {args.min_views} usable views; got {len(used)}. "
            "Capture more views with varied checkerboard orientation and distance."
        )
        return 2

    import cv2  # lazy

    first = cv2.imread(str(Path(args.images) / used[0]))
    height, width = first.shape[:2]
    try:
        result = calibrate_from_correspondences(
            obj_points,
            img_points,
            (width, height),
            board_size=board_size,
            square_size_m=args.square,
            device_label=args.device,
            min_views=args.min_views,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    result["images_used"] = used
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Calibration written to {out_path}")
    print(f"fx={result['fx']:.3f} fy={result['fy']:.3f} cx={result['cx']:.3f} cy={result['cy']:.3f}")
    print(f"RMS reprojection error: {result['rms_reprojection_error']:.4f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
