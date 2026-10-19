"""Test D3: Loader exact round trip of synthetic record."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dataset.loader import load_frame, load_mask, load_depth
from dataset.schema import FrameRecord, DistancePoint, MountInfo, Tags
from tests.synthetic import render_scene, save_scene
from core.config import Intrinsics


def test_d3_exact_round_trip():
    """D3: Exact round trip of a synthetic record through save/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a synthetic frame
        K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        depth, floor_mask, wall_mask, box_masks, gt = render_scene(
            K, H=480, W=640, h=1.4, pitch_deg=15.0, wall_distance=4.0
        )

        frame_id = "test_route_frame_00001"
        route_dir = tmpdir_path / "test_route"
        route_dir.mkdir()

        # Save scene
        save_scene(route_dir, frame_id, depth, floor_mask, wall_mask, box_masks, gt, K)

        # Create FrameRecord - paths relative to dataset root
        record = FrameRecord(
            frame_id=frame_id,
            route_id="test_route",
            frame_index=1,
            timestamp_ns=100_000_000,
            rgb_path=f"test_route/{frame_id}.jpg",
            depth_path=f"test_route/{frame_id}.npz",
            mask_path=f"test_route/{frame_id}.npz",
            intrinsics={
                "fx": float(K.fx),
                "fy": float(K.fy),
                "cx": float(K.cx),
                "cy": float(K.cy),
                "width": K.width,
                "height": K.height,
            },
            mount=MountInfo(height_m=1.4, pitch_deg=15.0),
            tags=Tags(env="indoor", light="day", hazard="none", crowd="low"),
            distance_points=[
                DistancePoint(u=320.0, v=300.0, distance_m=2.0),
            ],
        )

        # Load frame
        loaded = load_frame(tmpdir_path, record)

        # Verify depth matches (float32 precision from NPZ)
        assert np.allclose(loaded["depth"], depth, rtol=1e-6), "Depth should match within float32 precision"

        # Verify mask matches exactly
        # Note: mask file contains multiple masks; load_mask returns floor_mask
        loaded_mask = loaded["mask"]
        with np.load(route_dir / f"{frame_id}.npz") as data:
            saved_floor_mask = data["floor_mask"]
        assert np.array_equal(loaded_mask, saved_floor_mask), "Loaded floor mask should match saved"

        # Verify intrinsics
        assert loaded["intrinsics"]["fx"] == K.fx
        assert loaded["intrinsics"]["fy"] == K.fy

        # Verify mount
        assert loaded["mount"].height_m == 1.4
        assert loaded["mount"].pitch_deg == 15.0

        # Verify valid_mask
        expected_valid = np.isfinite(depth) & (depth > 0.0)
        assert np.array_equal(loaded["valid_mask"], expected_valid)


def test_d3_missing_files_clear_errors():
    """D3: Missing files give clear errors, not crashes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        record = FrameRecord(
            frame_id="missing_frame",
            route_id="test_route",
            frame_index=0,
            timestamp_ns=0,
            rgb_path="missing.jpg",
            depth_path="missing_depth.npz",
            mask_path="missing_mask.npz",
            intrinsics={"fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480},
            mount=MountInfo(height_m=1.4, pitch_deg=15.0),
            tags=Tags(env="indoor", light="day", hazard="none", crowd="low"),
            distance_points=[],
        )

        with pytest.raises(FileNotFoundError, match="missing_depth.npz"):
            load_frame(tmpdir_path, record)