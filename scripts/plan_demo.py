"""Visiona-Walk Bird's-Eye View Mapping, Path Planning, and Guidance Demo.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.

Usage:
    python scripts/plan_demo.py --fake
    python scripts/plan_demo.py image.jpg --intrinsics camera.json --mount-height 1.4 --pitch 15.0
    python scripts/plan_demo.py image.jpg --intrinsics camera.json \
        --weights-dir weights --depth-variant indoor \
        --conformal-depth reports/calibration/depth_interval.json \
        --fusion-config reports/fusion_tuning/fusion_tuning.json

``--floor-threshold`` (default 0.90) is the probability floor-certification
threshold and is TO TUNE ON REAL DATA. Real-image mode requires calibrated
intrinsics; there is no FOV fallback for a real camera. ``--conformal-depth``
supplies a fitted split-conformal depth interval and activates the pessimistic
lower-bound obstacle layer; without it obstacles use nominal distances only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
import warnings
import zlib
from typing import Optional, Sequence

import numpy as np

# Ensure root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import Intrinsics, MountConfig, RansacConfig, ScaleConfig
from calibration.depth_interval_config import (
    DepthIntervalConfigError,
    RuntimeDepthInterval,
    load_depth_interval,
)
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
from mapping.obstacles import nearest_obstacle_telemetry
from perception.fusion_config import (
    FusionConfigError,
    RuntimeFusionConfig,
    load_tuned_fusion_config,
    wrap_seg_with_fusion,
)
from perception.pipeline import PerceptionPipeline
from planning.astar import astar_plan
from planning.costmap import build_costmap
from planning.guidance import compute_guidance, Guidance
from planning.smooth import smooth_path
from planning.targets import find_explore_target
from scripts.real_models_smoke import check_weights, load_image, make_real_backends

DEFAULT_WEIGHTS = REPO_ROOT / "weights"
DEFAULT_FLOOR_THRESHOLD = 0.90  # TO TUNE ON REAL DATA
DEPTH_VARIANTS = ("indoor", "outdoor")


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


def load_intrinsics_json(path: str | Path) -> Intrinsics:
    """Load calibrated intrinsics from ``calibrate_camera.py`` JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("intrinsics JSON must contain an object")
    required = ("fx", "fy", "cx", "cy")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    if "image_size" in payload:
        size = payload["image_size"]
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("image_size must be [width, height]")
        width, height = int(size[0]), int(size[1])
    elif "width" in payload and "height" in payload:
        width, height = int(payload["width"]), int(payload["height"])
    else:
        raise ValueError("missing image_size [width, height]")
    values = {key: float(payload[key]) for key in required}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("intrinsics values must be finite")
    if values["fx"] <= 0.0 or values["fy"] <= 0.0:
        raise ValueError("fx and fy must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    if payload.get("provenance") == "synthetic" or payload.get("synthetic") is True:
        warnings.warn(
            "Intrinsics JSON is SYNTHETIC, not a measurement of a physical camera; "
            "metric distances on real frames are not trustworthy.",
            UserWarning,
            stacklevel=2,
        )
    return Intrinsics(
        fx=values["fx"],
        fy=values["fy"],
        cx=values["cx"],
        cy=values["cy"],
        width=width,
        height=height,
    )


def _certified_mask_for_free_space(
    labels: np.ndarray, certified_final: np.ndarray | None
) -> tuple[np.ndarray, bool]:
    """Return the only mask allowed to certify FREE votes, plus availability.

    Labels are never promoted to certification: fused labels may contain
    SAM-refined floor pixels whose probabilities were not certified. Missing or
    shape-mismatched certification fails closed to an all-False mask.
    """
    labels_arr = np.asarray(labels)
    if certified_final is None:
        return np.zeros(labels_arr.shape, dtype=bool), False
    mask = np.asarray(certified_final, dtype=bool)
    if mask.shape != labels_arr.shape:
        return np.zeros(labels_arr.shape, dtype=bool), False
    return mask, True


def _fusion_record(backend: object) -> dict | None:
    result = getattr(backend, "last_result", None)
    if result is None:
        return None
    iou = float(result.iou)
    sam_score = float(result.sam_score)
    return {
        "accepted": bool(result.accepted),
        "reason": str(result.reason),
        "prompt_count": int(result.prompt_count),
        "iou": iou if math.isfinite(iou) else None,
        "sam_score": sam_score if math.isfinite(sam_score) else None,
        "added_floor_pixels": int(result.added_floor_pixels),
        "removed_floor_pixels": int(result.removed_floor_pixels),
        "timings_ms": {str(name): float(ms) for name, ms in result.timings_ms.items()},
    }


def _grid_counts(grid: np.ndarray) -> dict[str, int]:
    return {
        "free": int(np.count_nonzero(grid == CELL_FREE)),
        "occupied": int(np.count_nonzero(grid == CELL_OCCUPIED)),
        "uncertain": int(np.count_nonzero(grid == CELL_UNCERTAIN)),
        "unknown": int(np.count_nonzero(grid == CELL_UNKNOWN)),
    }


def run_demo(
    image_path: Optional[str] = None,
    fake: bool = False,
    intrinsics_json: Optional[str] = None,
    mount_height_m: float = 1.4,  # TO TUNE ON REAL DATA
    pitch_deg: float = 15.0,  # TO TUNE ON REAL DATA
    out_dir: str = "reports/demo_plan",
    weights_dir: Optional[str] = None,
    depth_variant: str = "indoor",
    fusion_config_json: Optional[str] = None,
    depth_interval_json: Optional[str] = None,
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD,  # TO TUNE ON REAL DATA
) -> tuple[int, Optional[Guidance]]:
    """Execute mapping and planning demo pipeline.

    Returns ``(exit_code, guidance)``. Real-image mode exits 2 with an
    actionable message when weights, calibrated intrinsics, or MobileSAM are
    missing. Perception failures never raise; if certification is unavailable
    the run continues with no FREE cells and a STOP_UNCLEAR guidance status.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    threshold = float(floor_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        print(
            "ERROR: --floor-threshold must be a finite number in [0, 1], "
            f"got {floor_threshold!r}",
            file=sys.stderr,
        )
        return 2, None
    if depth_variant not in DEPTH_VARIANTS:
        print(
            f"ERROR: --depth-variant must be one of {DEPTH_VARIANTS}, "
            f"got {depth_variant!r}",
            file=sys.stderr,
        )
        return 2, None

    mount = MountConfig(h_known=float(mount_height_m), pitch_deg=float(pitch_deg))
    scale_cfg = ScaleConfig()
    estimator = ScaleEstimator(mount, scale_cfg)
    map_cfg = MapConfig()
    weights = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS

    fusion_config: RuntimeFusionConfig | None = None
    if fusion_config_json:
        try:
            fusion_config = load_tuned_fusion_config(
                fusion_config_json, allow_fake=bool(fake)
            )
        except FusionConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2, None
        print(
            "Fusion config:      "
            f"enabled={fusion_config.enabled} "
            f"reason={fusion_config.selection_reason}"
        )
    fusion_enabled = fusion_config is not None and fusion_config.enabled

    depth_interval: RuntimeDepthInterval | None = None
    if depth_interval_json:
        try:
            depth_interval = load_depth_interval(
                depth_interval_json, allow_fake=bool(fake)
            )
        except DepthIntervalConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2, None
        interval_model = depth_interval.model
        print(
            "Depth interval:     "
            f"mode={depth_interval.mode} score={interval_model.score} "
            f"alpha={interval_model.alpha:.2f} "
            f"q={interval_model.pooled_quantile_:.4f} "
            f"n={interval_model.pooled_count_}"
        )

    if not fake:
        if image_path is None:
            print(
                "ERROR: real-image mode requires an input image; use --fake "
                "for the analytic synthetic demo.",
                file=sys.stderr,
            )
            return 2, None
        missing = check_weights(weights, depth_variant)
        if missing:
            print("ERROR: real model weights are missing:", file=sys.stderr)
            for item in missing:
                print(
                    f"  - {item['model']} (expected at {item['expected_path']})",
                    file=sys.stderr,
                )
            print("\nFetch them with:", file=sys.stderr)
            print("  python scripts/fetch_models.py --dry-run", file=sys.stderr)
            print("  python scripts/fetch_models.py --yes", file=sys.stderr)
            return 2, None
        if not intrinsics_json:
            print(
                "ERROR: real-image mode requires calibrated intrinsics. "
                "Pass --intrinsics <camera.json>.",
                file=sys.stderr,
            )
            print(
                "  Generate the file with scripts/calibrate_camera.py; the "
                "demo will not guess a field of view for a real camera.",
                file=sys.stderr,
            )
            return 2, None

    intrinsics_source = "default_fov_75deg (fake mode only)"
    intrinsics_provenance = "default_fov_75deg (fake mode only)"
    if intrinsics_json:
        try:
            K = load_intrinsics_json(intrinsics_json)
        except (OSError, ValueError) as exc:
            print(
                f"ERROR: failed to load intrinsics {intrinsics_json}: {exc}",
                file=sys.stderr,
            )
            return 2, None
        intrinsics_source = str(intrinsics_json)
        payload = json.loads(Path(intrinsics_json).read_text(encoding="utf-8"))
        intrinsics_provenance = str(payload.get("provenance", "unspecified"))
    else:
        K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)

    if fake:
        image = np.zeros((K.height, K.width, 3), dtype=np.uint8)
    else:
        image_file = Path(image_path)
        if not image_file.is_file():
            print(f"ERROR: input image not found: {image_path}", file=sys.stderr)
            return 2, None
        try:
            image = load_image(image_file)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(
                f"ERROR: failed to load image {image_path}: {exc}",
                file=sys.stderr,
            )
            return 2, None
        if (image.shape[1], image.shape[0]) != (K.width, K.height):
            print(
                "ERROR: image is "
                f"{image.shape[1]}x{image.shape[0]} but intrinsics describe "
                f"{K.width}x{K.height}. Calibrate at the capture resolution.",
                file=sys.stderr,
            )
            return 2, None

    sam_backend = None
    if fake:
        from perception.fakes import (
            FakeDepthBackend,
            FakeDetector,
            FakePromptMaskBackend,
            FakeSegBackend,
        )

        box = (-0.4, 0.4, 2.5, 3.0, 0.0, 0.8)
        scene_args = {
            "K": K,
            "H": K.height,
            "W": K.width,
            "h": mount.h_known,
            "pitch_deg": mount.pitch_deg,
            "wall_distance": 5.0,
            "boxes": [box],
            "seed": 0,
        }
        base_seg = FakeSegBackend(**scene_args)
        depth_backend = FakeDepthBackend(**scene_args)
        det_backend = FakeDetector(**scene_args)
        if fusion_enabled:
            sam_backend = FakePromptMaskBackend(**scene_args, mode="floor")
    else:
        from perception.seg_fusion import MobileSAMBackend

        base_seg, depth_backend, det_backend = make_real_backends(
            weights, depth_variant
        )
        if fusion_enabled:
            sam_backend = MobileSAMBackend(
                checkpoint=weights / "mobile-sam" / "mobile_sam.pt"
            )
            sam_errors = sam_backend.preflight()
            if sam_errors:
                print(
                    "ERROR: tuned fusion is enabled but MobileSAM is not ready:",
                    file=sys.stderr,
                )
                for error in sam_errors:
                    print(f"  - {error}", file=sys.stderr)
                print("\nFetch it with:", file=sys.stderr)
                print(
                    "  python scripts/fetch_models.py --dry-run", file=sys.stderr
                )
                print("  python scripts/fetch_models.py --yes", file=sys.stderr)
                return 2, None

    seg_backend = (
        wrap_seg_with_fusion(base_seg, sam_backend, fusion_config)
        if fusion_enabled
        else base_seg
    )

    pipeline = PerceptionPipeline(
        seg=seg_backend,
        depth=depth_backend,
        detector=det_backend,
        K=K,
        mount=mount,
        estimator=estimator,
        conformal_depth=(
            None if depth_interval is None else depth_interval.model
        ),
        floor_threshold=threshold,
    )
    outputs = pipeline.run(image)

    # Build BEV Map
    if outputs.lift is None or outputs.seg is None:
        print(
            "FAIL: perception produced no lift or segmentation; refusing to "
            f"plan. Statuses: {outputs.statuses}",
            file=sys.stderr,
        )
        return 1, None

    labels = np.asarray(outputs.seg.labels)
    if outputs.pessimistic_pointmap_g is None:
        print(
            "WARNING: no fitted conformal depth interval was supplied; "
            "obstacles use nominal distances only (no lower-bound layer).",
            file=sys.stderr,
        )
    certified_mask, certification_available = _certified_mask_for_free_space(
        labels, outputs.certified_final
    )
    if not certification_available:
        print(
            "WARNING: certified floor mask is unavailable; failing closed: "
            "no cell can be FREE and guidance will not claim clear space.",
            file=sys.stderr,
        )
    vetoed_mask = (
        None if outputs.vetoed is None else np.asarray(outputs.vetoed, dtype=bool)
    )
    fusion_record = _fusion_record(seg_backend) if fusion_enabled else None

    bev = build_bev(
        pointmap_g=outputs.lift.point_map_g,
        labels=labels,
        certified_final=certified_mask,
        vetoed=vetoed_mask,
        pessimistic_pointmap=outputs.pessimistic_pointmap_g,
        cfg=map_cfg,
    )
    nearest_obstacle = nearest_obstacle_telemetry(
        outputs.lift.point_map_g,
        bev.obstacle_mask,
        lower_pointmap_g=outputs.pessimistic_pointmap_g,
        upper_pointmap_g=outputs.conformal_upper_pointmap_g,
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
        certified_floor_mask=certified_mask,
        plane_inlier_ratio=outputs.lift.plane_fit.inlier_ratio,
        plane_status=outputs.lift.plane_fit.status.value,
        scale_status=outputs.lift.scale_result.status,
        nearest_obstacle=nearest_obstacle,
    )

    interval_text = "no conformal bounds"
    if "distance_lower_m" in guidance.nearest_obstacle:
        interval_text = f"lower={guidance.nearest_obstacle['distance_lower_m']:.2f}m"
        if "distance_upper_m" in guidance.nearest_obstacle:
            interval_text += f" upper={guidance.nearest_obstacle['distance_upper_m']:.2f}m"

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
          f"at {guidance.nearest_obstacle.get('bearing_deg', 0.0):+.1f} deg ({interval_text})")
    print(f"Fusion:              {fusion_record['reason'] if fusion_record else 'disabled'}")
    print(
        "Certification:       "
        f"threshold={threshold:.2f} available={certification_available} "
        f"certified_px={int(np.count_nonzero(certified_mask))} "
        f"vetoed_px={0 if vetoed_mask is None else int(np.count_nonzero(vetoed_mask))}"
    )
    print(f"Reasons:             {'; '.join(guidance.reasons)}")
    print("=" * 60)

    # Save PNG and JSON artifacts
    bev_image = render_bev_image(bev.grid, path_res.path, smoothed_world, map_cfg)
    png_path = out_path / "bev_path.png"
    save_png_rgb(png_path, bev_image)

    json_path = out_path / "guidance_summary.json"
    json_path.write_text(json.dumps(guidance.to_dict(), indent=2), encoding="utf-8")

    run_summary = {
        "mode": "fake" if fake else "real",
        "image": None if fake else str(image_path),
        "intrinsics": {
            "source": intrinsics_source,
            "provenance": intrinsics_provenance,
            "fx": float(K.fx),
            "fy": float(K.fy),
            "cx": float(K.cx),
            "cy": float(K.cy),
            "width": int(K.width),
            "height": int(K.height),
        },
        "mount": {
            "height_m": float(mount.h_known),
            "pitch_deg": float(mount.pitch_deg),
        },
        "depth_variant": "fake" if fake else depth_variant,
        "weights_dir": None if fake else str(weights),
        "fusion_config": None if fusion_config is None else fusion_config.to_dict(),
        "fusion": fusion_record,
        "depth_interval": (
            None if depth_interval is None else depth_interval.provenance()
        ),
        "nearest_obstacle": dict(guidance.nearest_obstacle),
        "certification": {
            "floor_threshold": threshold,
            "available": bool(certification_available),
            "certified_pixels": int(np.count_nonzero(certified_mask)),
            "vetoed_pixels": (
                0 if vetoed_mask is None else int(np.count_nonzero(vetoed_mask))
            ),
            "pessimistic_layer": outputs.pessimistic_pointmap_g is not None,
            "note": (
                "floor_threshold is TO TUNE ON REAL DATA; free votes use only "
                "certified probabilities and the geometric veto, never labels"
            ),
        },
        "statuses": dict(outputs.statuses),
        "timings_ms": {
            str(name): float(ms) for name, ms in outputs.timings.items()
        },
        "grid_counts": _grid_counts(bev.grid),
        "start_cell": [int(start_c), int(start_r)],
        "target_cell": (
            None if target is None else [int(target[0]), int(target[1])]
        ),
        "decision": guidance.decision,
    }
    summary_path = out_path / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(f"Saved BEV path overlay to: {png_path}")
    print(f"Saved guidance summary to: {json_path}")
    print(f"Saved run provenance to:   {summary_path}")

    return 0, guidance


def main() -> None:
    parser = argparse.ArgumentParser(description="Visiona-Walk Bird's-Eye Map & Planner Demo")
    parser.add_argument("image", nargs="?", default=None, help="Path to input RGB image")
    parser.add_argument("--fake", action="store_true", help="Run offline analytic synthetic scene")
    parser.add_argument("--intrinsics", type=str, default=None, help="Camera intrinsics JSON file (required for real images)")
    parser.add_argument("--mount-height", type=float, default=1.4, help="Camera mounting height in metres")
    parser.add_argument("--pitch", type=float, default=15.0, help="Camera downward pitch in degrees")
    parser.add_argument("--out-dir", type=str, default="reports/demo_plan", help="Output directory")
    parser.add_argument("--weights-dir", type=str, default=None, help="Local model weights folder (default: weights/)")
    parser.add_argument(
        "--depth-variant",
        choices=DEPTH_VARIANTS,
        default="indoor",
        help="Metric depth checkpoint to use for real images",
    )
    parser.add_argument(
        "--fusion-config",
        type=str,
        default=None,
        help="fusion_tuning.json whose validated selection configures SegFormer+SAM fusion",
    )
    parser.add_argument(
        "--conformal-depth",
        type=str,
        default=None,
        help=(
            "validated depth_interval.json from scripts/run_calibration_real.py; "
            "activates the lower-bound obstacle layer"
        ),
    )
    parser.add_argument(
        "--floor-threshold",
        type=float,
        default=DEFAULT_FLOOR_THRESHOLD,
        help=(
            "Probability threshold for certified floor (default: "
            f"{DEFAULT_FLOOR_THRESHOLD}); TO TUNE ON REAL DATA"
        ),
    )

    args = parser.parse_args()

    exit_code, _ = run_demo(
        image_path=args.image,
        fake=args.fake,
        intrinsics_json=args.intrinsics,
        mount_height_m=args.mount_height,
        pitch_deg=args.pitch,
        out_dir=args.out_dir,
        weights_dir=args.weights_dir,
        depth_variant=args.depth_variant,
        fusion_config_json=args.fusion_config,
        depth_interval_json=args.conformal_depth,
        floor_threshold=args.floor_threshold,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
