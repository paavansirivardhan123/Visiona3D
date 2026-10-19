"""Test D11: Synthetic round trip - make_synth_dataset, load, evaluate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dataset.loader import iterate_dataset
from dataset.synth_dataset import build_synthetic_dataset
from evaluation.metrics_depth import compute_depth_metrics
from evaluation.metrics_seg import compute_segmentation_metrics
from core.config import MASK_FLOOR, MASK_WALL, MASK_OBSTACLE


def test_d11_synth_roundtrip_perfect():
    """D11: make_synth_dataset -> load -> evaluate with GT predictions gives AbsRel 0, IoU 1, FW 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Generate synthetic dataset
        summary = build_synthetic_dataset(
            out_dir=tmpdir_path,
            n_routes=2,
            frames_per_route=3,
            seed=42,
            image_size=(320, 240),  # Small for speed
        )

        # Iterate and evaluate with perfect predictions
        all_absrel = []
        all_iou = []
        all_fw = []

        for route_id, frame_data in iterate_dataset(tmpdir_path):
            depth = frame_data["depth"]
            mask = frame_data["mask"]
            valid_mask = frame_data["valid_mask"]

            # Perfect predictions = GT
            pred_depth = depth.copy()
            pred_mask = mask.copy()

            # Depth metrics
            depth_metrics = compute_depth_metrics(
                depth[valid_mask], pred_depth[valid_mask], valid_mask=valid_mask[valid_mask]
            )
            all_absrel.append(depth_metrics["AbsRel"])

            # Segmentation metrics
            seg_metrics = compute_segmentation_metrics(pred_mask, mask)
            all_iou.append(seg_metrics["mean_iou"])
            all_fw.append(seg_metrics["false_walkable_rate"])

        # With perfect predictions: AbsRel = 0, IoU = 1, FW = 0
        assert np.mean(all_absrel) == pytest.approx(0.0, abs=1e-6), f"AbsRel = {np.mean(all_absrel)}"
        assert np.mean(all_iou) == pytest.approx(1.0, abs=1e-6), f"IoU = {np.mean(all_iou)}"
        assert np.mean(all_fw) == pytest.approx(0.0, abs=1e-6), f"FW = {np.mean(all_fw)}"


def test_d11_synth_roundtrip_scaled_depth():
    """D11: With predictions 1.1x GT depth gives AbsRel 0.1 within 1e-6."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        summary = build_synthetic_dataset(
            out_dir=tmpdir_path,
            n_routes=2,
            frames_per_route=3,
            seed=42,
            image_size=(320, 240),
        )

        all_absrel = []

        for route_id, frame_data in iterate_dataset(tmpdir_path):
            depth = frame_data["depth"]
            valid_mask = frame_data["valid_mask"]

            # Prediction = 1.1 * GT
            pred_depth = depth * 1.1

            depth_metrics = compute_depth_metrics(
                depth[valid_mask], pred_depth[valid_mask], valid_mask=valid_mask[valid_mask]
            )
            all_absrel.append(depth_metrics["AbsRel"])

        mean_absrel = np.mean(all_absrel)
        assert mean_absrel == pytest.approx(0.1, abs=1e-6), f"AbsRel = {mean_absrel}, expected 0.1"