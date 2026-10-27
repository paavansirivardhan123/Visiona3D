"""Synthetic ablation evaluation script for Visiona3D Mapping & Planning.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.

Evaluates 4 ablation variants over seeded synthetic scenes:
(A) Naive: raw floor labels, no certified floor, no veto, no margin
(B) + Conformal floor calibration
(C) + Geometric veto
(D) + Conformal lower-bound margin (full system)

Outputs text and JSON clearly labelled: SYNTHETIC ONLY.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

# Ensure root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.conformal_depth import ConformalDepth
from calibration.conformal_floor import FloorCalibration
from calibration.geometric_veto import veto_floor_mask
from core.config import Intrinsics, MountConfig
from geometry.camera import backproject
from geometry.frames import camera_to_ground
from mapping.bev import build_bev
from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    MapConfig,
    world_to_cell,
)
from planning.astar import astar_plan, PathResult
from planning.costmap import build_costmap
from planning.guidance import compute_guidance
from planning.targets import find_explore_target
from tests.synthetic import render_scene


def _bootstrap_metric(
    values: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute mean and percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan"), float("nan"), float("nan")

    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = clean.size
    draw_idx = rng.integers(0, n, size=(n_resamples, n))
    means = clean[draw_idx].mean(axis=1)

    ci_low = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return float(np.mean(clean)), ci_low, ci_high


def check_path_collision_analytic(
    path_world: Sequence[tuple[float, float]],
    wall_distance: float,
    boxes: Sequence[tuple[float, float, float, float, float, float]],
    user_half_width_m: float = 0.35,  # TO TUNE ON REAL DATA
) -> bool:
    """Check if 2D path in ground frame collides with true wall or boxes."""
    if not path_world:
        return False

    for wx, wy in path_world:
        # Check wall collision: Y + half_width >= wall_distance
        if wy + user_half_width_m >= wall_distance:
            return True

        # Check box collision (including overhangs that extend down into traversable height)
        for bx0, bx1, by0, by1, bz0, bz1 in boxes:
            # Overhangs below 2.0m are physical obstacles for walking user
            if bz0 < 2.0:
                # Expanded box by user half-width
                if (bx0 - user_half_width_m <= wx <= bx1 + user_half_width_m) and (
                    by0 - user_half_width_m <= wy <= by1 + user_half_width_m
                ):
                    return True

    return False


def run_synthetic_ablation(
    n_scenes: int = 200,
    base_seed: int = 42,
    out_dir: str = "reports/planner_eval",
) -> dict[str, Any]:
    """Run synthetic ablation across 200 seeded scenes."""
    if n_scenes < 20:
        print("ERROR: fewer than 20 scenes requested. Exiting non-zero.", file=sys.stderr)
        sys.exit(1)

    t0_start = time.perf_counter()
    H, W = 240, 320
    K = Intrinsics.from_fov(width=W, height=H, hfov_deg=75.0)
    mount_h = 1.4
    pitch_deg = 15.0
    mount = MountConfig(h_known=mount_h, pitch_deg=pitch_deg)
    map_cfg = MapConfig()

    theta = math.radians(pitch_deg)
    n_plane = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)

    # Pre-fit a stub ConformalDepth model for lower-bound testing (stub q = 0.20)
    # Calibrated on synthetic range
    cdepth = ConformalDepth(alpha=0.10, score="overestimation")
    fake_gt = np.linspace(1.0, 8.0, 100)
    # Overestimation errors so quantile ~ 0.20
    fake_pred = fake_gt * 1.20
    cdepth.fit(fake_pred, fake_gt)

    rng = np.random.default_rng(base_seed)

    # Generate 200 random scenes
    scenes_data: list[dict[str, Any]] = []
    for i in range(n_scenes):
        scene_seed = base_seed + i * 13
        s_rng = np.random.default_rng(scene_seed)

        wall_dist = float(s_rng.uniform(3.5, 6.5))
        n_boxes = int(s_rng.integers(0, 3))

        boxes: list[tuple[float, float, float, float, float, float]] = []
        for b_idx in range(n_boxes):
            bx_center = float(s_rng.uniform(-1.0, 1.0))
            bw = float(s_rng.uniform(0.4, 0.7))
            by_start = float(s_rng.uniform(1.8, min(wall_dist - 0.8, 4.5)))
            bd = float(s_rng.uniform(0.4, 0.7))

            # 50% chance ground box, 50% chance floating overhang
            is_overhang = bool(s_rng.random() < 0.5)
            if is_overhang:
                bz0, bz1 = 1.0, 1.7  # Floating overhang
            else:
                bz0, bz1 = 0.0, 0.8  # Ground box

            boxes.append(
                (
                    bx_center - 0.5 * bw,
                    bx_center + 0.5 * bw,
                    by_start,
                    by_start + bd,
                    bz0,
                    bz1,
                )
            )

        # Render analytic scene
        depth_true, floor_mask, wall_mask, box_masks, _ = render_scene(
            K, H, W, h=mount_h, pitch_deg=pitch_deg, wall_distance=wall_dist, boxes=boxes, seed=scene_seed
        )

        # Depth noise: 3% relative Gaussian noise
        noise = s_rng.normal(0.0, 0.03, size=(H, W))
        noisy_depth = np.where(depth_true > 0, depth_true * (1.0 + noise), np.nan).astype(np.float32)

        # Ground frame point cloud
        pts_c = backproject(noisy_depth, K)
        pts_g = camera_to_ground(pts_c, n_plane, mount_h)

        # True labels
        labels = np.zeros((H, W), dtype=np.int64)
        labels[floor_mask] = 1  # FLOOR
        labels[wall_mask] = 2   # WALL
        for bm in box_masks:
            labels[bm] = 3      # OBSTACLE

        # Corruption: 30% wall-as-floor corruption in all odd scenes
        corrupted_labels = labels.copy()
        has_corruption = (i % 2 == 1)
        if has_corruption:
            wall_pixels = np.argwhere(wall_mask)
            if len(wall_pixels) > 0:
                n_corrupt = int(round(0.30 * len(wall_pixels)))
                perm = s_rng.choice(len(wall_pixels), size=n_corrupt, replace=False)
                corrupted_labels[wall_pixels[perm, 0], wall_pixels[perm, 1]] = 1

        # Probabilities
        probs = np.zeros((H, W, 4), dtype=np.float32)
        # Floor channel: high for floor
        probs[corrupted_labels == 1, 0] = 0.95
        probs[corrupted_labels != 1, 0] = 0.05
        # Normalize
        probs[..., 1] = np.where(corrupted_labels == 2, 0.85, 0.05)
        probs[..., 2] = np.where(corrupted_labels == 3, 0.85, 0.05)
        probs[..., 3] = 0.05
        probs /= np.sum(probs, axis=-1, keepdims=True)

        # Veto computation
        final_floor, veto_mask = veto_floor_mask(
            certified_mask=(corrupted_labels == 1),
            points_g_full=pts_g,
            tau0_m=0.10,
            tau1=0.02,
        )

        # Pessimistic lower bound points
        c_pred = cdepth.predict(noisy_depth.reshape(-1))
        lower_d = c_pred.lower.reshape(H, W)
        lower_pts_c = backproject(lower_d, K)
        lower_pts_g = camera_to_ground(lower_pts_c, n_plane, mount_h)

        scenes_data.append(
            {
                "index": i,
                "wall_dist": wall_dist,
                "boxes": boxes,
                "pts_g": pts_g,
                "lower_pts_g": lower_pts_g,
                "raw_labels": corrupted_labels,
                "probs": probs,
                "final_floor": final_floor,
                "veto_mask": veto_mask,
                "has_corruption": has_corruption,
            }
        )

    # Evaluate 4 variants
    variants = ["A_naive", "B_conformal_floor", "C_geometric_veto", "D_conformal_margin"]
    results: dict[str, Any] = {
        "metadata": {
            "title": "Visiona3D Mapping & Planning Synthetic Ablation",
            "tag": "SYNTHETIC ONLY",
            "n_scenes": n_scenes,
            "seed": base_seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "variants": {},
    }

    start_c, start_r = world_to_cell(0.0, 2.0, map_cfg)

    for var in variants:
        unsafe_flags: list[float] = []
        available_flags: list[float] = []
        clearances_list: list[float] = []
        latencies_ms: list[float] = []

        # Also track specifically on corrupted scenes
        corrupted_unsafe_flags: list[float] = []

        for sc in scenes_data:
            t_plan_start = time.perf_counter()

            if var == "A_naive":
                # Raw floor labels, no certified floor, no veto, no pessimistic margin
                bev = build_bev(
                    pointmap_g=sc["pts_g"],
                    labels=sc["raw_labels"],
                    certified_final=(sc["raw_labels"] == 1),
                    vetoed=None,
                    pessimistic_pointmap=None,
                    cfg=map_cfg,
                )
                costmap = build_costmap(
                    bev.grid, cfg=map_cfg, user_half_width_m=0.0, safety_margin_m=0.0
                )

            elif var == "B_conformal_floor":
                # Certified floor via probability threshold, no veto, no margin
                cert_floor = sc["probs"][..., 0] >= 0.50
                bev = build_bev(
                    pointmap_g=sc["pts_g"],
                    labels=sc["raw_labels"],
                    certified_final=cert_floor,
                    vetoed=None,
                    pessimistic_pointmap=None,
                    cfg=map_cfg,
                )
                costmap = build_costmap(bev.grid, cfg=map_cfg, safety_margin_m=0.0)

            elif var == "C_geometric_veto":
                # Conformal floor + geometric veto, standard margin
                bev = build_bev(
                    pointmap_g=sc["pts_g"],
                    labels=sc["raw_labels"],
                    certified_final=sc["final_floor"],
                    vetoed=sc["veto_mask"],
                    pessimistic_pointmap=None,
                    cfg=map_cfg,
                )
                costmap = build_costmap(bev.grid, cfg=map_cfg, safety_margin_m=0.10)

            elif var == "D_conformal_margin":
                # Full system: veto + lower bound pessimistic points + safety margin
                bev = build_bev(
                    pointmap_g=sc["pts_g"],
                    labels=sc["raw_labels"],
                    certified_final=sc["final_floor"],
                    vetoed=sc["veto_mask"],
                    pessimistic_pointmap=sc["lower_pts_g"],
                    cfg=map_cfg,
                )
                costmap = build_costmap(bev.grid, cfg=map_cfg, safety_margin_m=0.10)

            # Target & A* Plan
            target = find_explore_target(costmap, (start_c, start_r))
            if target is None:
                path_res = PathResult("NO_PATH", [], [], math.inf, 0.0, [], 0.0)
            else:
                path_res = astar_plan(costmap, (start_c, start_r), target)

            lat_ms = (time.perf_counter() - t_plan_start) * 1000.0
            latencies_ms.append(lat_ms)

            guidance = compute_guidance(costmap=costmap, path_result=path_res)

            is_available = guidance.decision in ("GUIDE", "CAUTION")
            available_flags.append(1.0 if is_available else 0.0)

            is_unsafe = False
            if is_available and len(path_res.path_world) > 0:
                is_unsafe = check_path_collision_analytic(
                    path_res.path_world,
                    sc["wall_dist"],
                    sc["boxes"],
                    user_half_width_m=0.35,
                )
                clearances_list.append(path_res.min_clearance_m)
            else:
                clearances_list.append(float("nan"))

            unsafe_flags.append(1.0 if is_unsafe else 0.0)

            if sc["has_corruption"]:
                corrupted_unsafe_flags.append(1.0 if is_unsafe else 0.0)

        # Compute bootstrap stats
        u_mean, u_low, u_high = _bootstrap_metric(np.array(unsafe_flags))
        u_corrupt_mean, u_c_low, u_c_high = _bootstrap_metric(np.array(corrupted_unsafe_flags))
        a_mean, a_low, a_high = _bootstrap_metric(np.array(available_flags))
        c_mean, c_low, c_high = _bootstrap_metric(np.array(clearances_list))

        p50_lat = float(np.percentile(latencies_ms, 50))
        p95_lat = float(np.percentile(latencies_ms, 95))

        results["variants"][var] = {
            "unsafe_rate": u_mean,
            "unsafe_rate_ci": [u_low, u_high],
            "unsafe_rate_corrupted_scenes": u_corrupt_mean,
            "unsafe_rate_corrupted_ci": [u_c_low, u_c_high],
            "availability": a_mean,
            "availability_ci": [a_low, a_high],
            "mean_min_clearance_m": c_mean,
            "mean_min_clearance_ci": [c_low, c_high],
            "latency_p50_ms": p50_lat,
            "latency_p95_ms": p95_lat,
        }

    total_time = time.perf_counter() - t0_start
    results["metadata"]["total_elapsed_s"] = total_time

    # Print clean ablation table
    print("\n" + "=" * 105)
    print("SYNTHETIC ONLY: VISIONA-WALK MAPPING & PLANNING ABLATION EVALUATION")
    print(f"{n_scenes} Scenes | Wall 3.5-6.5m | Overhangs & Ground Boxes | Wall-as-Floor Corruption | Runtime: {total_time:.2f}s")
    print("=" * 105)
    print(
        f"{'Variant':<22} | {'Unsafe Rate (All)':<22} | {'Unsafe (Corrupted)':<22} | "
        f"{'Availability':<16} | {'Clearance (m)':<15} | {'Latency p50/p95':<15}"
    )
    print("-" * 120)
    for var, data in results["variants"].items():
        u_str = f"{data['unsafe_rate']:.1%} [{data['unsafe_rate_ci'][0]:.1%}, {data['unsafe_rate_ci'][1]:.1%}]"
        uc_str = f"{data['unsafe_rate_corrupted_scenes']:.1%} [{data['unsafe_rate_corrupted_ci'][0]:.1%}, {data['unsafe_rate_corrupted_ci'][1]:.1%}]"
        av_str = f"{data['availability']:.1%}"
        cl_str = f"{data['mean_min_clearance_m']:.2f} m"
        lat_str = f"{data['latency_p50_ms']:.1f} / {data['latency_p95_ms']:.1f} ms"
        print(f"{var:<22} | {u_str:<22} | {uc_str:<22} | {av_str:<16} | {cl_str:<15} | {lat_str:<15}")
    print("=" * 120)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "ablation_results.json"
    report_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to: {report_file}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Visiona3D Planner Synthetic Ablation")
    parser.add_argument("--scenes", type=int, default=200, help="Number of random scenes (>= 20)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--out-dir", type=str, default="reports/planner_eval", help="Output directory")

    args = parser.parse_args()

    run_synthetic_ablation(
        n_scenes=args.scenes,
        base_seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
