"""Test D10: JSONL logging."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from evaluation.logging_jsonl import JSONLLogger, FrameLogRecord, read_jsonl, validate_jsonl_schema


def test_d10_write_and_parse():
    """D10: Written JSONL parses back correctly."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        logger = JSONLLogger(tmp_path, mode="w")

        record = FrameLogRecord(
            frame_id="route_001_frame_00005",
            route_id="route_001",
            timestamp_ns=500_000_000,
            wall_time_s=1234567890.123,
            module_latencies={"depth": 45.2, "lift": 12.5},
            prediction_summary={"depth_mean": 3.5, "depth_std": 0.8},
            gt_summary={"depth_mean": 3.4, "num_points": 10},
            status_codes={"plane_fit": "OK", "scale": "OK", "anchor": "OK"},
        )

        logger.write(record)
        logger.close()

        # Read back
        records = read_jsonl(tmp_path)
        assert len(records) == 1

        r = records[0]
        assert r["frame_id"] == "route_001_frame_00005"
        assert r["route_id"] == "route_001"
        assert r["timestamp_ns"] == 500_000_000
        assert r["wall_time_s"] == 1234567890.123
        assert r["module_latencies"]["depth"] == 45.2
        assert r["status_codes"]["plane_fit"] == "OK"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_d10_schema_violation_rejected():
    """D10: Schema-violating record is rejected on write."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        logger = JSONLLogger(tmp_path, mode="w")

        # Missing required field
        bad_record = {
            "frame_id": "test",
            "route_id": "route_001",
            "timestamp_ns": 0,
            "wall_time_s": 0.0,
            "module_latencies": {},
            "prediction_summary": {},
            "gt_summary": {},
            # Missing status_codes
        }

        with pytest.raises(ValueError, match="Missing required field"):
            logger.write_dict(bad_record)

        # Wrong type
        bad_record2 = {
            "frame_id": "test",
            "route_id": "route_001",
            "timestamp_ns": "not_an_int",  # Wrong type
            "wall_time_s": 0.0,
            "module_latencies": {},
            "prediction_summary": {},
            "gt_summary": {},
            "status_codes": {},
        }

        with pytest.raises(ValueError, match="timestamp_ns must be non-negative int"):
            logger.write_dict(bad_record2)

        logger.close()
    finally:
        # On Windows, ensure file is closed before unlink
        import gc
        gc.collect()
        try:
            tmp_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # Ignore on Windows if still locked


def test_d10_validate_jsonl_schema():
    """D10: validate_jsonl_schema catches violations."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        # Write valid record
        logger = JSONLLogger(tmp_path, mode="w")
        logger.write(FrameLogRecord(
            frame_id="f1", route_id="r1", timestamp_ns=0, wall_time_s=0.0,
            module_latencies={}, prediction_summary={}, gt_summary={}, status_codes={}
        ))
        logger.close()

        # Validate
        errors = validate_jsonl_schema(tmp_path)
        assert errors == []

        # Append invalid record
        with open(tmp_path, "a") as f:
            f.write('{"frame_id": "f2"}\n')  # Missing most fields

        errors = validate_jsonl_schema(tmp_path)
        assert len(errors) > 0
        assert any("missing field" in e for e in errors)
    finally:
        tmp_path.unlink(missing_ok=True)