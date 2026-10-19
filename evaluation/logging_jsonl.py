"""JSONL logging for Visiona3D evaluation.

Schema-checked per-frame JSONL writer with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Union
import json
import time


@dataclass
class FrameLogRecord:
    """Schema for per-frame evaluation log entry.

    Attributes:
        frame_id: Unique frame identifier.
        route_id: Route identifier.
        timestamp_ns: Frame timestamp in nanoseconds.
        wall_time_s: Wall-clock time when logged (seconds since epoch).
        module_latencies: Dict of module_name -> latency_ms.
        prediction_summary: Dict with prediction summary statistics.
        gt_summary: Dict with ground truth summary statistics.
        status_codes: Dict of status codes (e.g., plane_fit, scale, anchor).
    """
    frame_id: str
    route_id: str
    timestamp_ns: int
    wall_time_s: float
    module_latencies: dict[str, float]
    prediction_summary: dict[str, Any]
    gt_summary: dict[str, Any]
    status_codes: dict[str, str]


class JSONLLogger:
    """JSONL writer with schema validation.

    Each line is a valid JSON object matching FrameLogRecord schema.
    """

    REQUIRED_FIELDS = [
        "frame_id", "route_id", "timestamp_ns", "wall_time_s",
        "module_latencies", "prediction_summary", "gt_summary", "status_codes"
    ]

    def __init__(self, file_path: Union[str, Path], mode: str = "a"):
        """Initialize logger.

        Args:
            file_path: Path to JSONL file.
            mode: File mode ('a' for append, 'w' for overwrite).
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.file_path.open(mode, encoding="utf-8")
        self._count = 0

    def write(self, record: FrameLogRecord) -> None:
        """Write a validated record to JSONL.

        Args:
            record: FrameLogRecord to write.

        Raises:
            ValueError: If record fails schema validation.
        """
        # Convert to dict
        data = asdict(record)

        # Validate required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate types
        if not isinstance(data["frame_id"], str) or not data["frame_id"]:
            raise ValueError("frame_id must be non-empty string")
        if not isinstance(data["route_id"], str) or not data["route_id"]:
            raise ValueError("route_id must be non-empty string")
        if not isinstance(data["timestamp_ns"], int) or data["timestamp_ns"] < 0:
            raise ValueError("timestamp_ns must be non-negative int")
        if not isinstance(data["wall_time_s"], (int, float)) or data["wall_time_s"] < 0:
            raise ValueError("wall_time_s must be non-negative number")
        if not isinstance(data["module_latencies"], dict):
            raise ValueError("module_latencies must be dict")
        if not isinstance(data["prediction_summary"], dict):
            raise ValueError("prediction_summary must be dict")
        if not isinstance(data["gt_summary"], dict):
            raise ValueError("gt_summary must be dict")
        if not isinstance(data["status_codes"], dict):
            raise ValueError("status_codes must be dict")

        # Write as single JSON line
        self._file.write(json.dumps(data, separators=(",", ":")) + "\n")
        self._count += 1

    def write_dict(self, data: dict[str, Any]) -> None:
        """Write a raw dict (validated) to JSONL."""
        try:
            record = FrameLogRecord(**data)
        except TypeError as e:
            # Convert dataclass missing field errors to ValueError
            raise ValueError(f"Missing required field: {e}") from e
        self.write(record)

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self._count


def read_jsonl(file_path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read all records from a JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
    return records


def validate_jsonl_schema(file_path: Union[str, Path]) -> list[str]:
    """Validate all records in a JSONL file against schema.

    Returns:
        List of error messages (empty if all valid).
    """
    errors = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Check required fields
                for field in JSONLLogger.REQUIRED_FIELDS:
                    if field not in data:
                        errors.append(f"Line {line_num}: missing field '{field}'")
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON - {e}")
    return errors