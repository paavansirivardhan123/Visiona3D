"""Test D1: Schema validation for FrameRecord."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.config import Intrinsics
from dataset.loader import load_frame
from dataset.schema import (
    FrameRecord,
    DistancePoint,
    MountInfo,
    Tags,
    validate,
)


def make_valid_record() -> FrameRecord:
    """Create a valid FrameRecord for testing."""
    return FrameRecord(
        frame_id="route_001_frame_00005",
        route_id="route_001",
        frame_index=5,
        timestamp_ns=500_000_000,
        rgb_path="route_001/frame_00005.jpg",
        depth_path="route_001/frame_00005_depth.npz",
        mask_path="route_001/frame_00005_mask.npz",
        intrinsics={
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640,
            "height": 480,
        },
        mount=MountInfo(height_m=1.4, pitch_deg=15.0),
        tags=Tags(env="indoor", light="day", hazard="none", crowd="low"),
        distance_points=[
            DistancePoint(u=320.0, v=300.0, distance_m=2.0, method="rangefinder", uncertainty_m=0.02),
        ],
    )


def test_d1_valid_record_passes():
    """D1: A valid record passes validation with no errors."""
    record = make_valid_record()
    errors = validate(record)
    assert errors == [], f"Valid record should have no errors, got: {errors}"


@pytest.mark.parametrize("field,corrupt_fn,expected_msg", [
    ("missing_key", lambda r: setattr(r, "frame_id", ""), "frame_id: must be non-empty"),
    ("negative_height", lambda r: setattr(r.mount, "height_m", -1.0), "height_m: must be > 0"),
    ("class_index_oob", lambda r: r.distance_points.append(
        type(r.distance_points[0])(u=100, v=100, distance_m=1.0, method="synthetic", uncertainty_m=0.0)
    ) or setattr(r.distance_points[-1], "method", "invalid"), "method must be one of"),
    ("bad_tag_value", lambda r: setattr(r.tags, "env", "invalid"), "tags.env: must be one of"),
    ("wrong_dtype", lambda r: setattr(r, "frame_index", "not_an_int"), "frame_index: must be an integer"),
    ("empty_route_id", lambda r: setattr(r, "route_id", ""), "route_id: must be non-empty"),
])
def test_d1_corruptions(field: str, corrupt_fn, expected_msg: str):
    """D1: Each of 6 corruptions gives a specific error message in validate()."""
    record = make_valid_record()
    corrupt_fn(record)
    errors = validate(record)
    assert any(expected_msg in e for e in errors), (
        f"Corruption '{field}': expected message containing '{expected_msg}', got errors: {errors}"
    )


def test_d1_loader_catches_bounds_and_existence():
    """D1: Loader catches distance point out of bounds and nonexistent files."""
    from dataset.loader import load_frame
    import tempfile
    from pathlib import Path
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a valid NPZ file with depth and mask
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        mask = np.zeros((480, 640), dtype=np.uint8)
        np.savez_compressed(tmpdir_path / "dummy.npz", depth=depth, floor_mask=mask)

        # Test distance point outside image bounds
        record = make_valid_record()
        record.distance_points[0].u = 1000.0  # Outside 640 width
        record.depth_path = "dummy.npz"
        record.mask_path = "dummy.npz"

        with pytest.raises(ValueError, match="outside image bounds"):
            load_frame(tmpdir_path, record)

        # Test nonexistent mask path
        record2 = make_valid_record()
        record2.mask_path = "nonexistent.npz"
        record2.depth_path = "dummy.npz"

        with pytest.raises(FileNotFoundError):
            load_frame(tmpdir_path, record2)