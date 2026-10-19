"""Benchmark script for Visiona3D geometry module lift_frame on a 1280x720 frame.

Times lift_frame on a 1280x720 synthetic frame over 20 runs and reports
the median runtime in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from core.config import Intrinsics, MountConfig, RansacConfig, ScaleConfig
from geometry.lift import lift_frame
from geometry.scale import ScaleEstimator
from tests.synthetic import render_scene


def run_benchmark() -> float:
    """Run benchmark and return median latency in milliseconds."""
    H, W = 720, 1280
    K = Intrinsics.from_fov(width=W, height=H, hfov_deg=80.0)
    h_known = 1.4
    pitch_deg = 15.0

    print("=" * 70)
    print("BENCHMARK: lift_frame on 1280x720 Frame (Visiona3D Geometry)")
    print("=" * 70)
    print(f"Image Resolution: {W}x{H}")
    print(f"Mounted Height:   {h_known:.2f} m")
    print(f"Camera Pitch:     {pitch_deg:.1f} deg")

    # Render clean scene with floor and wall
    depth, floor_mask, wall_mask, _, _ = render_scene(
        K, H=H, W=W, h=h_known, pitch_deg=pitch_deg, wall_distance=4.0
    )
    floor_pixel_count = int(np.sum(floor_mask))
    print(f"Floor Pixels:     {floor_pixel_count} / {W * H} ({floor_pixel_count / (W * H) * 100:.1f}%)")
    print("-" * 70)

    mount = MountConfig(h_known=h_known, pitch_deg=pitch_deg)
    ransac_cfg = RansacConfig(max_iterations=500, max_points=5000)

    # Warmup runs
    for _ in range(2):
        estimator_warmup = ScaleEstimator(mount, ScaleConfig())
        rng_warmup = np.random.default_rng(0)
        lift_frame(
            depth,
            floor_mask,
            {"wall": wall_mask},
            K,
            mount,
            estimator_warmup,
            rng=rng_warmup,
            ransac_cfg=ransac_cfg,
        )

    # Timed runs
    num_runs = 20
    latencies_ms: list[float] = []

    for run_idx in range(num_runs):
        estimator = ScaleEstimator(mount, ScaleConfig())
        rng = np.random.default_rng(run_idx + 1)

        t0 = time.perf_counter()
        res = lift_frame(
            depth,
            floor_mask,
            {"wall": wall_mask},
            K,
            mount,
            estimator,
            rng=rng,
            ransac_cfg=ransac_cfg,
        )
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    median_ms = float(np.median(latencies_ms))
    mean_ms = float(np.mean(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))

    print(f"Runs:             {num_runs}")
    print(f"Median Latency:   {median_ms:.2f} ms ({1000.0 / median_ms:.1f} FPS)")
    print(f"Mean Latency:     {mean_ms:.2f} ms")
    print(f"Min Latency:      {min_ms:.2f} ms")
    print(f"Max Latency:      {max_ms:.2f} ms")
    print(f"95th Percentile:  {p95_ms:.2f} ms")
    print("=" * 70)

    return median_ms


if __name__ == "__main__":
    run_benchmark()
