"""Visiona-Walk Bird's-Eye View Mapping, Path Planning, and Guidance Demo.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.

Usage:
    python scripts/plan_demo.py --fake
    python scripts/plan_demo.py image.jpg --intrinsics camera.json --mount-height 1.4 --pitch 15.0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
import zlib
from typing import Optional, Sequence

import numpy as np

# Ensure root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import Intrinsics, MountConfig, RansacConfig, ScaleConfig
from geometry.scale import ScaleEstimator
from mapping.bev import build_bev
from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNCERTAIN,
    CELL_UNKNOWN,
    MapConfig,
    world_to_cell,
)
from perception.pipeline import PerceptionPipeline
from planning.astar import astar_plan
from planning.costmap import build_costmap
from planning.guidance import compute_guidance, Guidance
from planning.smooth import smooth_path
from planning.targets import find_explore_target


def save_png_rgb(filepath: Path, rgb_array: np.ndarray) -> None:
    """Write RGB uint8 array of shape (H, W, 3) to PNG file using standard library only."""
    arr = np.asarray(rgb_array, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) uint8 array, got {arr.shape}")
    height, width, _ = arr.shape

    # Prepend filter byte 0 to each scanline
    raw_bytes = bytearray()
    for row in range(height):
        raw_bytes.append(0)
        raw_bytes.extend(arr[row].tobytes())

    compressed = zlib.compress(bytes(raw_bytes), level=6)

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    png_header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width(4), height(4), bit_depth(1=8), color_type(1=2 for RGB), comp(0), filter(0), interlace(0)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_chunk = _png_chunk(b"IHDR", ihdr_data)
    idat_chunk = _png_chunk(b"IDAT", compressed)
    iend_chunk = _png_chunk(b"IEND", b"")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(png_header)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)


def render_bev_image(
    bev_grid: np.ndarray,
    path_cells: Sequence[tuple[int, int]] = (),
    smoothed_path_world: Sequence[tuple[float, float]] = (),
    cfg: MapConfig | None = None,
) -> np.ndarray:
    """Render BEV occupancy grid to uint8 RGB image with path overlay.

    Grid rows grow with forward distance Y, so display image flips vertically
    such that forward is UP and camera origin is near bottom.
    """
    if cfg is None:
        cfg = MapConfig()
    n_rows, n_cols = bev_grid.shape

    # Base palette:
    # UNKNOWN: dark grey (80, 80, 85)
    # FREE: light green (210, 245, 210)
    # OCCUPIED: red (220, 50, 50)
    # UNCERTAIN: gold/yellow (240, 190, 50)
    img = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
    img[bev_grid == CELL_UNKNOWN] = [80, 80, 85]
    img[bev_grid == CELL_FREE] = [210, 245, 210]
    img[bev_grid == CELL_OCCUPIED] = [220, 50, 50]
    img[bev_grid == CELL_UNCERTAIN] = [240, 190, 50]

    # Draw path cells in blue
    for c, r in path_cells:
        if 0 <= c < n_cols and 0 <= r < n_rows:
            img[r, c] = [30, 100, 255]

    # Draw smoothed path waypoints
    for wx, wy in smoothed_path_world:
        sc, sr = world_to_cell(wx, wy, cfg)
        if 0 <= sc < n_cols and 0 <= sr < n_rows:
            img[sr, sc] = [0, 200, 255]

    # Draw start origin in cyan
    c_start, r_start = world_to_cell(0.0, 0.0, cfg)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r_start + dr, c_start + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                img[nr, nc] = [0, 255, 255]

    # Flip vertically so forward (larger row) is at the top of the image
    img = np.flipud(img)

    # Upscale 4x for visualization clarity
    upscaled = np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)
    return upscaled


def run_demo(
    image_path: Optional[str] = None,
    fake: bool = False,
    intrinsics_json: Optional[str] = None,
    mount_height_m: float = 1.4,  # TO TUNE ON REAL DATA
    pitch_deg: float = 15.0,  # TO TUNE ON REAL DATA
    out_dir: str = "reports/demo_plan",
) -> tuple[int, Guidance]:
    """Execute mapping and planning demo pipeline."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mount = MountConfig(h_known=float(mount_height_m), pitch_deg=float(pitch_deg))
    scale_cfg = ScaleConfig()
    estimator = ScaleEstimator(mount, scale_cfg)
    map_cfg = MapConfig()

    if intrinsics_json and Path(intrinsics_json).exists():
        data = json.loads(Path(intrinsics_json).read_text(encoding="utf-8"))
        K = Intrinsics(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data.get("width", 640)),
            height=int(data.get("height", 480)),
        )
    else:
        K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)

    if fake or image_path is None:
        from perception.fakes import FakeDepthBackend, FakeDetector, FakeSegBackend
        from tests.synthetic import render_scene

        # Create analytic corridor scene with a floor and box obstacle
        box = (-0.4, 0.4, 2.5, 3.0, 0.0, 0.8)
        scene_args = {
            "K": K,
            "H": 480,
            "W": 640,
            "h": mount.h_known,
            "pitch_deg": mount.pitch_deg,
            "wall_distance": 5.0,
            "boxes": [box],
            "seed": 0,
        }
        seg_backend = FakeSegBackend(**scene_args)
        depth_backend = FakeDepthBackend(**scene_args)
        det_backend = FakeDetector(**scene_args)

        pipeline = PerceptionPipeline(
            seg=seg_backend,
            depth=depth_backend,
            detector=det_backend,
            K=K,
            mount=mount,
            estimator=estimator,
        )
        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        outputs = pipeline.run(fake_image)
    else:
        # Real image mode
        # Check if real model weights exist
        seg_weights = REPO_ROOT / "weights" / "segformer-b0-ade"
        depth_weights = REPO_ROOT / "weights" / "depth-anything-v2-metric-indoor-small"
        if not seg_weights.exists() or not depth_weights.exists():
            print(
                "ERROR: Real model weights missing. "
                "Run `python scripts/fetch_models.py` to download weights.",
                file=sys.stderr,
            )
            sys.exit(2)

        # Lazy load real backends if available
        try:
            from PIL import Image
            raw_img = Image.open(image_path).convert("RGB")
            real_image = np.array(raw_img, dtype=np.uint8)
        except Exception as exc:
            print(f"ERROR: Failed to load image {image_path}: {exc}", file=sys.stderr)
            sys.exit(2)

        # If real backends cannot run without weights/dependencies, exit 2
        print(
            "ERROR: Real model backends not available in offline research mode. "
            "Use `--fake` for analytic verification.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Build BEV Map
    if outputs.lift is None or outputs.seg is None:
        print("FAIL: Perception outputs missing lift or seg.", file=sys.stderr)
        sys.exit(1)

    bev = build_bev(
        pointmap_g=outputs.lift.point_map_g,
        labels=outputs.seg.labels,
        certified_final=outputs.certified_final if outputs.certified_final is not None else (outputs.seg.labels == 1),
        vetoed=outputs.vetoed,
        pessimistic_pointmap=outputs.pessimistic_pointmap_g,
        cfg=map_cfg,
    )

    # Costmap & Planning
    costmap = build_costmap(bev.grid, cfg=map_cfg)
    start_c, start_r = world_to_cell(0.0, 0.0, map_cfg)
    if not costmap.is_passable(start_r, start_c):
        # Look forward along centerline out of camera blind zone for first certified cell
        for y_fwd in np.arange(0.1, 3.0, map_cfg.res):
            tc, tr = world_to_cell(0.0, float(y_fwd), map_cfg)
            if costmap.is_passable(tr, tc):
                start_c, start_r = tc, tr
                break

    target = find_explore_target(costmap, (start_c, start_r))

    if target is None:
        path_res = astar_plan(costmap, (start_c, start_r), (start_c, start_r))
    else:
        path_res = astar_plan(costmap, (start_c, start_r), target)

    smoothed_world, _ = smooth_path(path_res.path_world, costmap)

    # Guidance computation
    guidance = compute_guidance(
        costmap=costmap,
        path_result=path_res,
        points_g=outputs.lift.point_map_g,
        certified_floor_mask=outputs.certified_final if outputs.certified_final is not None else (outputs.seg.labels == 1),
        plane_inlier_ratio=outputs.lift.plane_fit.inlier_ratio,
        plane_status=outputs.lift.plane_fit.status.value,
        scale_status=outputs.lift.scale_result.status,
    )

    # Print Guidance output
    print("=" * 60)
    print("VISIONA-WALK GUIDANCE TELEMETRY")
    print("=" * 60)
    print(f"Decision:           {guidance.decision}")
    print(f"Recommended Heading: {guidance.heading_deg:+.1f} deg (positive right)")
    print(f"Lookahead Target:    X={guidance.lookahead_point_g[0]:.2f}m, Y={guidance.lookahead_point_g[1]:.2f}m")
    print(f"Distance to Turn:    {guidance.distance_to_turn_m:.2f} m")
    print(f"Lateral Clearance:   Left={guidance.left_clearance_m:.2f}m, Right={guidance.right_clearance_m:.2f}m")
    print(f"Nearest Obstacle:    {guidance.nearest_obstacle.get('distance_nominal_m', float('nan')):.2f}m "
          f"at {guidance.nearest_obstacle.get('bearing_deg', 0.0):+.1f} deg")
    print(f"Reasons:             {'; '.join(guidance.reasons)}")
    print("=" * 60)

    # Save PNG and JSON artifacts
    bev_image = render_bev_image(bev.grid, path_res.path, smoothed_world, map_cfg)
    png_path = out_path / "bev_path.png"
    save_png_rgb(png_path, bev_image)

    json_path = out_path / "guidance_summary.json"
    json_path.write_text(json.dumps(guidance.to_dict(), indent=2), encoding="utf-8")

    print(f"Saved BEV path overlay to: {png_path}")
    print(f"Saved guidance summary to: {json_path}")

    return 0, guidance


def main() -> None:
    parser = argparse.ArgumentParser(description="Visiona-Walk Bird's-Eye Map & Planner Demo")
    parser.add_argument("image", nargs="?", default=None, help="Path to input RGB image")
    parser.add_argument("--fake", action="store_true", help="Run offline analytic synthetic scene")
    parser.add_argument("--intrinsics", type=str, default=None, help="Camera intrinsics JSON file")
    parser.add_argument("--mount-height", type=float, default=1.4, help="Camera mounting height in metres")
    parser.add_argument("--pitch", type=float, default=15.0, help="Camera downward pitch in degrees")
    parser.add_argument("--out-dir", type=str, default="reports/demo_plan", help="Output directory")

    args = parser.parse_args()

    exit_code, _ = run_demo(
        image_path=args.image,
        fake=args.fake,
        intrinsics_json=args.intrinsics,
        mount_height_m=args.mount_height,
        pitch_deg=args.pitch,
        out_dir=args.out_dir,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
