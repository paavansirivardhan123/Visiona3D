"""Synthetic dataset generation.

Builds a synthetic dataset on disk using the analytic renderer from tests/synthetic.py.
Creates N routes x M frames with varying height, pitch, wall distance, boxes, and tags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union
import json
import numpy as np

from core.config import Intrinsics, MountConfig
from tests.synthetic import render_scene, save_scene


def build_synthetic_dataset(
    out_dir: Union[str, Path],
    n_routes: int,
    frames_per_route: int,
    seed: int = 42,
    image_size: tuple[int, int] = (640, 480),
    hfov_deg: float = 75.0,
    height_range: tuple[float, float] = (1.2, 1.8),
    pitch_range: tuple[float, float] = (5.0, 25.0),
    wall_distance_range: tuple[float, float] = (2.0, 8.0),
    max_boxes_per_frame: int = 3,
    box_size_range: tuple[float, float] = (0.2, 0.8),
) -> dict[str, Any]:
    """Build a synthetic dataset on disk.

    Args:
        out_dir: Output directory root.
        n_routes: Number of routes to generate.
        frames_per_route: Number of frames per route.
        seed: Random seed for reproducibility.
        image_size: (width, height) of rendered frames.
        hfov_deg: Horizontal field of view in degrees.
        height_range: (min, max) camera height in metres.
        pitch_range: (min, max) camera pitch in degrees.
        wall_distance_range: (min, max) wall distance in metres.
        max_boxes_per_frame: Maximum number of obstacle boxes per frame.
        box_size_range: (min, max) box dimensions in metres.

    Returns:
        Dict with dataset statistics and manifest info.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    W, H = image_size
    K = Intrinsics.from_fov(width=W, height=H, hfov_deg=hfov_deg)

    rng = np.random.default_rng(seed)

    # Tag vocabularies
    env_tags = ["indoor", "outdoor"]
    light_tags = ["day", "dusk", "dark"]
    hazard_tags = ["glass", "mirror", "wet", "none"]
    crowd_tags = ["none", "low", "high"]

    route_manifests = {}
    total_frames = 0

    for route_idx in range(n_routes):
        route_id = f"route_{route_idx:03d}"
        route_dir = out_path / route_id
        route_dir.mkdir(parents=True, exist_ok=True)

        # Sample route-level tags (consistent within route)
        route_tags = {
            "env": rng.choice(env_tags),
            "light": rng.choice(light_tags),
            "hazard": rng.choice(hazard_tags),
            "crowd": rng.choice(crowd_tags),
        }

        # Sample route-level camera parameters (with small per-frame variation)
        base_height = float(rng.uniform(*height_range))
        base_pitch = float(rng.uniform(*pitch_range))
        base_wall_dist = float(rng.uniform(*wall_distance_range))

        frames_info = []

        for frame_idx in range(frames_per_route):
            # Add small per-frame variations
            h = float(np.clip(base_height + rng.normal(0, 0.02), 0.5, 2.5))
            pitch = float(np.clip(base_pitch + rng.normal(0, 2.0), 0.0, 60.0))
            wall_dist = float(np.clip(base_wall_dist + rng.normal(0, 0.2), 0.5, 20.0))

            # Generate random boxes
            n_boxes = rng.integers(0, max_boxes_per_frame + 1)
            boxes = []
            for _ in range(n_boxes):
                # Box center in G frame: X in [-2, 2], Y in [1, wall_dist-0.5], Z in [0.2, 1.5]
                cx = float(rng.uniform(-2.0, 2.0))
                cy = float(rng.uniform(1.0, max(1.1, wall_dist - 0.5)))
                cz = float(rng.uniform(0.2, 1.5))
                sx = float(rng.uniform(*box_size_range))
                sy = float(rng.uniform(*box_size_range))
                sz = float(rng.uniform(*box_size_range))
                boxes.append((
                    cx - sx/2, cx + sx/2,
                    cy - sy/2, cy + sy/2,
                    cz - sz/2, cz + sz/2
                ))

            # Render scene
            depth, floor_mask, wall_mask, box_masks, gt = render_scene(
                K, H=H, W=W, h=h, pitch_deg=pitch, wall_distance=wall_dist, boxes=boxes
            )

            frame_id = f"{route_id}_frame_{frame_idx:05d}"

            # Save frame data
            save_scene(route_dir, frame_id, depth, floor_mask, wall_mask, box_masks, gt, K)

            # Build distance points from ground truth
            # For synthetic data: use floor/wall/box intersections as distance points
            distance_points = []
            # Add a few sample points on floor, wall, boxes
            for mask_name, mask in [("floor", floor_mask), ("wall", wall_mask)] + \
                                    [(f"box_{i}", bm) for i, bm in enumerate(box_masks)]:
                if np.any(mask):
                    # Sample up to 5 points per surface
                    indices = np.where(mask)
                    n_samples = min(5, len(indices[0]))
                    if n_samples > 0:
                        pick = rng.choice(len(indices[0]), size=n_samples, replace=False)
                        for idx in pick:
                            u = float(indices[1][int(idx)])
                            v = float(indices[0][int(idx)])
                            d = float(depth[int(v), int(u)])
                            if d > 0 and np.isfinite(d):
                                distance_points.append({
                                    "u": u,
                                    "v": v,
                                    "distance_m": d,
                                    "method": "synthetic",
                                    "uncertainty_m": 0.0,
                                })

            frame_info = {
                "frame_id": frame_id,
                "route_id": route_id,
                "frame_index": frame_idx,
                "timestamp_ns": frame_idx * 100_000_000,  # 100ms per frame
                "rgb_path": f"{route_id}/{frame_id}.png",  # Placeholder - no actual RGB
                "depth_path": f"{route_id}/{frame_id}.npz",
                "mask_path": f"{route_id}/{frame_id}.npz",
                "intrinsics": {
                    "fx": float(K.fx),
                    "fy": float(K.fy),
                    "cx": float(K.cx),
                    "cy": float(K.cy),
                    "width": K.width,
                    "height": K.height,
                },
                "mount": {
                    "height_m": h,
                    "pitch_deg": pitch,
                    "roll_deg": 0.0,
                    "yaw_deg": 0.0,
                },
                "tags": route_tags,
                "distance_points": distance_points,
            }
            frames_info.append(frame_info)

            total_frames += 1

        # Write route manifest
        manifest = {
            "route_id": route_id,
            "tags": route_tags,
            "frames": frames_info,
        }
        manifest_path = route_dir / "route_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        route_manifests[route_id] = manifest

    # Write dataset-level summary
    summary = {
        "n_routes": n_routes,
        "frames_per_route": frames_per_route,
        "total_frames": total_frames,
        "image_size": {"width": W, "height": H},
        "hfov_deg": hfov_deg,
        "seed": seed,
        "routes": list(route_manifests.keys()),
    }
    summary_path = out_path / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Visiona3D dataset")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--routes", "-r", type=int, default=5, help="Number of routes")
    parser.add_argument("--frames", "-f", type=int, default=20, help="Frames per route")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--hfov", type=float, default=75.0, help="Horizontal FOV (deg)")

    args = parser.parse_args()

    summary = build_synthetic_dataset(
        out_dir=args.output,
        n_routes=args.routes,
        frames_per_route=args.frames,
        seed=args.seed,
        image_size=(args.width, args.height),
        hfov_deg=args.hfov,
    )
    print(f"Generated dataset: {summary['total_frames']} frames across {summary['n_routes']} routes")
    print(f"Summary saved to: {args.output}/dataset_summary.json")