"""Perception backend interfaces and output dataclasses.

All backends exchange numpy arrays with a fixed, checkable contract:

- images: RGB uint8 (H, W, 3)
- labels: uint8 (H, W) using core.config mask indices
- probabilities: float32 (H, W, 4) in [floor, wall, obstacle, other] channel order
- depth: float32 (H, W) metres, z-depth in camera frame C, NaN where invalid
- sky: boolean (H, W) mask, excluded from depth and label metrics

Each dataclass exposes ``validate()`` returning a list of human-readable
errors. An empty list means the object satisfies the interface contract.
Backends are duck-typed Protocols so real and fake implementations are
interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable
import numpy as np

from core.config import MASK_OTHER


_FRAME_LABEL_IDS = (0, 1, 2, 3, 4)


def _as_float_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray([np.nan], dtype=np.float64)


@dataclass
class Det2D:
    """One 2D object detection in original-image pixel coordinates."""

    class_name: str
    box_xyxy: np.ndarray
    conf: float

    def validate(self, image_hw: tuple[int, int] | None = None) -> list[str]:
        """Return contract violations for this detection."""
        errors: list[str] = []

        if not isinstance(self.class_name, str) or not self.class_name.strip():
            errors.append("Det2D.class_name: must be a non-empty string")

        try:
            conf = float(self.conf)
        except (TypeError, ValueError):
            errors.append("Det2D.conf: must be a finite number in [0, 1]")
            conf = float("nan")
        if not np.isfinite(conf) or not (0.0 <= conf <= 1.0):
            errors.append("Det2D.conf: must be a finite number in [0, 1]")

        box = _as_float_array(self.box_xyxy)
        if box.shape != (4,):
            errors.append(f"Det2D.box_xyxy: expected shape (4,), got {box.shape}")
        elif not np.isfinite(box).all():
            errors.append("Det2D.box_xyxy: coordinates must be finite")
        else:
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                errors.append(
                    f"Det2D.box_xyxy: expected x2 > x1 and y2 > y1, got {tuple(box.tolist())}"
                )
            elif image_hw is not None:
                h, w = int(image_hw[0]), int(image_hw[1])
                if x1 < 0.0 or y1 < 0.0 or x2 > w or y2 > h:
                    errors.append(
                        f"Det2D.box_xyxy: box {tuple(box.tolist())} outside image bounds "
                        f"{w}x{h}"
                    )
        return errors


@dataclass
class SegOutput:
    """Semantic segmentation output at the original image resolution."""

    labels: np.ndarray
    probs: np.ndarray
    sky_mask: np.ndarray

    def validate(self) -> list[str]:
        """Return contract violations for labels, probabilities and sky mask."""
        errors: list[str] = []

        labels = np.asarray(self.labels)
        probs = np.asarray(self.probs)
        sky = np.asarray(self.sky_mask)

        if labels.ndim != 2:
            errors.append(f"SegOutput.labels: expected 2D (H, W), got {labels.shape}")
        if labels.dtype != np.uint8:
            errors.append(f"SegOutput.labels: expected uint8, got {labels.dtype}")
        if labels.size:
            bad = np.setdiff1d(np.unique(labels), np.asarray(_FRAME_LABEL_IDS, dtype=labels.dtype))
            if bad.size:
                errors.append(
                    f"SegOutput.labels: class id {int(bad[0])} outside valid range 0..4"
                )

        if probs.ndim != 3 or probs.shape[-1] != 4:
            errors.append(
                f"SegOutput.probs: expected (H, W, 4), got {probs.shape}"
            )
        if probs.dtype != np.float32:
            errors.append(f"SegOutput.probs: expected float32, got {probs.dtype}")
        if probs.ndim == 3 and probs.shape[-1] == 4 and probs.size:
            if not np.isfinite(probs).all():
                errors.append("SegOutput.probs: contains NaN or inf")
            elif float(probs.min()) < -1e-6 or float(probs.max()) > 1.0 + 1e-6:
                errors.append("SegOutput.probs: values must lie in [0, 1]")
            else:
                deviation = float(np.max(np.abs(probs.sum(axis=-1) - 1.0)))
                if deviation > 1e-4:
                    errors.append(
                        "SegOutput.probs: probabilities must sum to 1 along the channel "
                        f"axis; max deviation {deviation:.6g}"
                    )

        if sky.ndim != 2:
            errors.append(f"SegOutput.sky_mask: expected 2D (H, W), got {sky.shape}")
        if sky.dtype != np.bool_:
            errors.append(f"SegOutput.sky_mask: expected bool, got {sky.dtype}")
        if labels.ndim == 2 and probs.ndim == 3 and sky.ndim == 2:
            if not (labels.shape == sky.shape == probs.shape[:2]):
                errors.append(
                    "SegOutput: labels, probs and sky_mask must share (H, W); got "
                    f"{labels.shape}, {probs.shape[:2]}, {sky.shape}"
                )
        return errors


@dataclass
class DepthOutput:
    """Metric z-depth output at the original image resolution."""

    depth_m: np.ndarray
    max_range_m: float
    model_id: str

    def validate(self) -> list[str]:
        """Return contract violations for the depth map and metadata."""
        errors: list[str] = []

        depth = np.asarray(self.depth_m)
        if depth.ndim != 2:
            errors.append(f"DepthOutput.depth_m: expected 2D (H, W), got {depth.shape}")
        if depth.dtype != np.float32:
            errors.append(f"DepthOutput.depth_m: expected float32, got {depth.dtype}")

        try:
            max_range = float(self.max_range_m)
        except (TypeError, ValueError):
            errors.append("DepthOutput.max_range_m: must be a positive finite number")
            max_range = float("nan")
        if not np.isfinite(max_range) or max_range <= 0.0:
            errors.append("DepthOutput.max_range_m: must be a positive finite number")

        if not isinstance(self.model_id, str) or not self.model_id.strip():
            errors.append("DepthOutput.model_id: must be a non-empty string")

        if depth.ndim == 2 and depth.size:
            if np.isinf(depth).any():
                errors.append("DepthOutput.depth_m: inf is invalid; use NaN for invalid pixels")
            finite = np.isfinite(depth)
            if finite.any() and float(np.min(depth[finite])) < 0.0:
                errors.append(
                    "DepthOutput.depth_m: negative depth values must be NaN "
                    f"(found {float(np.min(depth[finite])):.3f})"
                )
        return errors


@dataclass
class StageTiming:
    """One measured pipeline stage latency."""

    stage: str
    ms: float

    def validate(self) -> list[str]:
        """Return contract violations for a stage timing."""
        errors: list[str] = []
        if not isinstance(self.stage, str) or not self.stage.strip():
            errors.append("StageTiming.stage: must be a non-empty string")
        try:
            ms = float(self.ms)
        except (TypeError, ValueError):
            errors.append("StageTiming.ms: must be a non-negative finite number")
            ms = float("nan")
        if not np.isfinite(ms) or ms < 0.0:
            errors.append("StageTiming.ms: must be a non-negative finite number")
        return errors


@dataclass
class FrameOutputs:
    """Complete per-frame perception output plus lift result and timings.

    ``statuses`` maps stage names to short codes:
    ``"OK"``, a plane-fit status value, ``"error:<stage>:<Type>:<message>"``
    or ``"skipped:<reason>"``. The pipeline never raises on backend failure;
    the error is reported here instead.
    """

    image_size: tuple[int, int]
    seg: SegOutput | None
    depth: DepthOutput | None
    detections: list[Det2D]
    lift: Any | None
    timings: dict[str, float] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    certified_final: np.ndarray | None = None
    vetoed: np.ndarray | None = None
    pessimistic_pointmap_g: np.ndarray | None = None

    def validate(self) -> list[str]:
        """Return contract violations for the frame container."""
        errors: list[str] = []
        try:
            h, w = int(self.image_size[0]), int(self.image_size[1])
            if h <= 0 or w <= 0:
                raise ValueError
        except (TypeError, ValueError, IndexError):
            errors.append("FrameOutputs.image_size: must be a positive (H, W) pair")
            h, w = 0, 0

        if self.seg is not None:
            errors.extend(self.seg.validate())
        if self.depth is not None:
            errors.extend(self.depth.validate())
        for det in self.detections:
            errors.extend(det.validate((h, w) if h and w else None))
        for name, ms in self.timings.items():
            if not isinstance(name, str) or not name:
                errors.append("FrameOutputs.timings: keys must be non-empty stage names")
            if not np.isfinite(float(ms)) or float(ms) < 0.0:
                errors.append(f"FrameOutputs.timings[{name!r}]: must be non-negative")
        for name, code in self.statuses.items():
            if not isinstance(name, str) or not isinstance(code, str):
                errors.append("FrameOutputs.statuses: keys and values must be strings")
        return errors

    def to_log_record(
        self,
        frame_id: str,
        route_id: str,
        timestamp_ns: int | None = None,
        wall_time_s: float | None = None,
    ) -> dict[str, Any]:
        """Convert to the dict schema accepted by evaluation.logging_jsonl."""
        import time as _time

        class_histogram = {name: 0 for name in ("floor", "wall", "obstacle", "other")}
        depth_valid_fraction = 0.0
        depth_min_m: float | None = None
        depth_max_m: float | None = None
        wall_distance_m: float | None = None

        if self.seg is not None:
            labels = np.asarray(self.seg.labels)
            for label_id, name in ((1, "floor"), (2, "wall"), (3, "obstacle"), (4, "other")):
                class_histogram[name] = int(np.sum(labels == label_id))

        if self.depth is not None:
            depth = np.asarray(self.depth.depth_m)
            finite = depth[np.isfinite(depth)]
            depth_valid_fraction = float(finite.size / depth.size) if depth.size else 0.0
            if finite.size:
                depth_min_m = float(np.min(finite))
                depth_max_m = float(np.max(finite))

        if self.lift is not None:
            wall_points = getattr(self.lift, "points_g", {}).get("wall")
            if wall_points is not None and np.asarray(wall_points).size:
                wall_distance_m = float(np.median(np.asarray(wall_points)[:, 1]))

        prediction_summary: dict[str, Any] = {
            "n_detections": len(self.detections),
            "class_histogram": class_histogram,
            "depth_valid_fraction": depth_valid_fraction,
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
            "wall_distance_m": wall_distance_m,
        }

        return {
            "frame_id": str(frame_id),
            "route_id": str(route_id),
            "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else 0),
            "wall_time_s": float(wall_time_s if wall_time_s is not None else _time.time()),
            "module_latencies": {str(k): float(v) for k, v in self.timings.items()},
            "prediction_summary": prediction_summary,
            "gt_summary": {},
            "status_codes": {str(k): str(v) for k, v in self.statuses.items()},
        }


@runtime_checkable
class SegBackend(Protocol):
    """Semantic segmentation backend contract."""

    def predict(self, image_rgb: np.ndarray) -> SegOutput:
        """Return segmentation at the input image resolution."""


@runtime_checkable
class DepthBackend(Protocol):
    """Metric depth backend contract."""

    def predict(self, image_rgb: np.ndarray) -> DepthOutput:
        """Return metric z-depth at the input image resolution."""


@runtime_checkable
class DetectorBackend(Protocol):
    """2D object detector backend contract."""

    def predict(self, image_rgb: np.ndarray) -> Sequence[Det2D]:
        """Return 2D detections in input-image pixel coordinates."""


__all__ = [
    "Det2D",
    "SegOutput",
    "DepthOutput",
    "StageTiming",
    "FrameOutputs",
    "SegBackend",
    "DepthBackend",
    "DetectorBackend",
    "MASK_OTHER",
]
