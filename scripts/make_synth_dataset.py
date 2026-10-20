#!/usr/bin/env python3
"""CLI script to generate synthetic Visiona3D dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.synth_dataset import build_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Visiona3D dataset for evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for dataset"
    )
    parser.add_argument(
        "--routes", "-r", type=int, default=5, help="Number of routes to generate"
    )
    parser.add_argument(
        "--frames", "-f", type=int, default=20, help="Frames per route"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Image width in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Image height in pixels"
    )
    parser.add_argument(
        "--hfov", type=float, default=75.0, help="Horizontal field of view (degrees)"
    )

    args = parser.parse_args()

    out_dir = Path(args.output)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Warning: Output directory {out_dir} is not empty", file=sys.stderr)
        response = input("Continue anyway? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(1)

    print(f"Generating synthetic dataset...")
    print(f"  Output: {out_dir}")
    print(f"  Routes: {args.routes}")
    print(f"  Frames per route: {args.frames}")
    print(f"  Seed: {args.seed}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  HFOV: {args.hfov} deg")

    summary = build_synthetic_dataset(
        out_dir=args.output,
        n_routes=args.routes,
        frames_per_route=args.frames,
        seed=args.seed,
        image_size=(args.width, args.height),
        hfov_deg=args.hfov,
    )

    print(f"\nDone! Generated {summary['total_frames']} frames across {summary['n_routes']} routes")
    print(f"Summary saved to: {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()