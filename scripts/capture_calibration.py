"""Capture checkerboard views from a webcam for camera calibration.

The bundled target is calib_target/visiona_checkerboard_A4_18mm.pdf:
9x6 inner corners (10x7 squares), 18 mm squares, printed at 100% scale.
Measure one square with a ruler after printing; do not trust the nominal
size if the printer scaled the page.

Usage:
    python scripts/capture_calibration.py --camera 0 --out calib_webcam
    python scripts/calibrate_camera.py --images calib_webcam --board 9x6 \
        --square 0.018 --out reports/camera_intrinsics.json

Controls: SPACE saves the current view when the board is detected, Q quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


def _parse_board(text: str) -> tuple[int, int]:
    parts = str(text).lower().split("x")
    if len(parts) != 2:
        raise ValueError("--board must look like 9x6")
    cols, rows = int(parts[0]), int(parts[1])
    if cols < 2 or rows < 2:
        raise ValueError("board corners must be at least 2x2")
    return cols, rows


def _find_corners(gray: np.ndarray, board: tuple[int, int]) -> np.ndarray | None:
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, board, flags=flags)
        if ok:
            return corners
    ok, corners = cv2.findChessboardCorners(
        gray,
        board,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    return corners if ok else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture checkerboard views from a webcam")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--out", default="calib_webcam", help="output image folder")
    parser.add_argument("--board", default="9x6", help="inner corners, e.g. 9x6")
    parser.add_argument("--views", type=int, default=15, help="target number of saved views")
    parser.add_argument("--width", type=int, default=None, help="requested capture width")
    parser.add_argument("--height", type=int, default=None, help="requested capture height")
    args = parser.parse_args(argv)

    try:
        board = _parse_board(args.board)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            f"ERROR: could not open camera {args.camera}. Close other apps that "
            "may be using it and try again.",
            file=sys.stderr,
        )
        return 2
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    saved = 0
    print(f"Capture target: {args.views} views at {board[0]}x{board[1]} inner corners")
    print("Move the board to different distances, angles and image corners.")
    print("SPACE = save detected view, Q = finish.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("ERROR: camera stopped returning frames.", file=sys.stderr)
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = _find_corners(gray, board)
            view = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, board, corners, True)
                status = f"board found - SPACE to save ({saved}/{args.views})"
                color = (80, 220, 80)
            else:
                status = f"searching for {board[0]}x{board[1]} board ({saved}/{args.views})"
                color = (60, 60, 230)
            cv2.putText(
                view, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )
            cv2.imshow("Visiona calibration capture", view)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" ") and corners is not None:
                path = out_dir / f"view_{saved:02d}.png"
                cv2.imwrite(str(path), frame)
                saved += 1
                print(f"  saved {path}")
                if saved >= args.views:
                    print("  target reached; press Q to finish or keep capturing.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if saved == 0:
        print("No views saved; calibration needs at least 10 usable views.", file=sys.stderr)
        return 1
    print(
        "\nNow run:\n"
        f"  python scripts/calibrate_camera.py --images {out_dir} "
        f"--board {board[0]}x{board[1]} --square 0.018 "
        "--out reports/camera_intrinsics.json\n"
        "Use --square equal to one printed square measured in metres."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
