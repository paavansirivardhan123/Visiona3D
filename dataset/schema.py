"""Dataset schema dataclasses and validation.

Defines the core data structures for the Visiona3D dataset:
- FrameRecord: per-frame annotations and metadata
- DistancePoint: ground-truth distance measurements
- MountInfo: camera mounting parameters
- Tags: scene/condition tags
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np


@dataclass
class DistancePoint:
    """A single ground-truth distance measurement.

    Attributes:
        u: Pixel x coordinate (u right).
        v: Pixel y coordinate (v down).
        distance_m: Ground-truth z-depth in camera frame C (metres).
        method: How the distance was obtained ("synthetic", "rangefinder", "lidar", "manual").
        uncertainty_m: Estimated uncertainty in metres (0 for synthetic).
    """
    u: float
    v: float
    distance_m: float
    method: str = "synthetic"
    uncertainty_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "u": self.u,
            "v": self.v,
            "distance_m": self.distance_m,
            "method": self.method,
            "uncertainty_m": self.uncertainty_m,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistancePoint":
        return cls(
            u=float(d["u"]),
            v=float(d["v"]),
            distance_m=float(d["distance_m"]),
            method=str(d.get("method", "synthetic")),
            uncertainty_m=float(d.get("uncertainty_m", 0.0)),
        )


@dataclass
class MountInfo:
    """Camera mounting configuration as recorded during data collection.

    Attributes:
        height_m: Camera height above ground in metres.
        pitch_deg: Camera pitch angle in degrees (positive = tilted down).
        roll_deg: Camera roll angle in degrees (positive = right side down).
        yaw_deg: Camera yaw angle in degrees (not used in current geometry).
    """
    height_m: float
    pitch_deg: float
    roll_deg: float = 0.0
    yaw_deg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "height_m": self.height_m,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "yaw_deg": self.yaw_deg,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MountInfo":
        return cls(
            height_m=float(d["height_m"]),
            pitch_deg=float(d["pitch_deg"]),
            roll_deg=float(d.get("roll_deg", 0.0)),
            yaw_deg=float(d.get("yaw_deg", 0.0)),
        )


@dataclass
class Tags:
    """Scene and condition tags for a route or frame.

    Attributes:
        env: Environment type ("indoor", "outdoor").
        light: Lighting condition ("day", "dusk", "dark").
        hazard: Hazard type ("glass", "mirror", "wet", "none").
        crowd: Crowd level ("none", "low", "high").
    """
    env: str
    light: str
    hazard: str
    crowd: str

    def to_dict(self) -> dict[str, str]:
        return {
            "env": self.env,
            "light": self.light,
            "hazard": self.hazard,
            "crowd": self.crowd,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Tags":
        return cls(
            env=str(d["env"]),
            light=str(d["light"]),
            hazard=str(d["hazard"]),
            crowd=str(d["crowd"]),
        )


@dataclass
class FrameRecord:
    """Complete annotation record for a single frame.

    Attributes:
        frame_id: Unique frame identifier (e.g., "route_001_frame_0005").
        route_id: Route identifier this frame belongs to.
        frame_index: Sequential index within the route (0-based).
        timestamp_ns: Timestamp in nanoseconds (monotonic within route).
        rgb_path: Path to RGB image file (relative to dataset root).
        depth_path: Path to depth map file (.npz or .npy, relative to dataset root).
        mask_path: Path to segmentation mask file (.npz or .npy, relative to dataset root).
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy, width, height.
        mount: Camera mounting info.
        tags: Scene/condition tags.
        distance_points: List of ground-truth distance measurements.
    """
    frame_id: str
    route_id: str
    frame_index: int
    timestamp_ns: int
    rgb_path: str
    depth_path: str
    mask_path: str
    intrinsics: dict[str, Any]
    mount: MountInfo
    tags: Tags
    distance_points: list[DistancePoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "route_id": self.route_id,
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "mask_path": self.mask_path,
            "intrinsics": self.intrinsics,
            "mount": self.mount.to_dict(),
            "tags": self.tags.to_dict(),
            "distance_points": [dp.to_dict() for dp in self.distance_points],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrameRecord":
        return cls(
            frame_id=str(d["frame_id"]),
            route_id=str(d["route_id"]),
            frame_index=int(d["frame_index"]),
            timestamp_ns=int(d["timestamp_ns"]),
            rgb_path=str(d["rgb_path"]),
            depth_path=str(d["depth_path"]),
            mask_path=str(d["mask_path"]),
            intrinsics=dict(d["intrinsics"]),
            mount=MountInfo.from_dict(d["mount"]),
            tags=Tags.from_dict(d["tags"]),
            distance_points=[DistancePoint.from_dict(dp) for dp in d.get("distance_points", [])],
        )


def validate(record: FrameRecord) -> list[str]:
    """Validate a FrameRecord and return list of human-readable error messages.

    Returns empty list if valid. Each error message describes a specific violation.
    """
    errors = []

    # Required string fields non-empty
    if not record.frame_id or not record.frame_id.strip():
        errors.append("frame_id: must be non-empty string")
    if not record.route_id or not record.route_id.strip():
        errors.append("route_id: must be non-empty string")
    if not record.rgb_path or not record.rgb_path.strip():
        errors.append("rgb_path: must be non-empty string")
    if not record.depth_path or not record.depth_path.strip():
        errors.append("depth_path: must be non-empty string")
    if not record.mask_path or not record.mask_path.strip():
        errors.append("mask_path: must be non-empty string")

    # frame_index >= 0
    try:
        if record.frame_index < 0:
            errors.append("frame_index: must be >= 0")
    except TypeError:
        errors.append("frame_index: must be an integer")

    # timestamp_ns >= 0
    try:
        if record.timestamp_ns < 0:
            errors.append("timestamp_ns: must be >= 0")
    except TypeError:
        errors.append("timestamp_ns: must be an integer")

    # intrinsics required keys
    required_intrinsics = ["fx", "fy", "cx", "cy", "width", "height"]
    for key in required_intrinsics:
        if key not in record.intrinsics:
            errors.append(f"intrinsics: missing required key '{key}'")
        else:
            val = record.intrinsics[key]
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append(f"intrinsics['{key}']: must be positive number")

    # mount validation
    if record.mount.height_m <= 0:
        errors.append("mount.height_m: must be > 0")
    if not (-90 <= record.mount.pitch_deg <= 90):
        errors.append("mount.pitch_deg: must be in [-90, 90]")
    if not (-180 <= record.mount.roll_deg <= 180):
        errors.append("mount.roll_deg: must be in [-180, 180]")
    if not (-180 <= record.mount.yaw_deg <= 180):
        errors.append("mount.yaw_deg: must be in [-180, 180]")

    # tags validation
    valid_env = {"indoor", "outdoor"}
    valid_light = {"day", "dusk", "dark"}
    valid_hazard = {"glass", "mirror", "wet", "none"}
    valid_crowd = {"none", "low", "high"}

    if record.tags.env not in valid_env:
        errors.append(f"tags.env: must be one of {sorted(valid_env)}, got '{record.tags.env}'")
    if record.tags.light not in valid_light:
        errors.append(f"tags.light: must be one of {sorted(valid_light)}, got '{record.tags.light}'")
    if record.tags.hazard not in valid_hazard:
        errors.append(f"tags.hazard: must be one of {sorted(valid_hazard)}, got '{record.tags.hazard}'")
    if record.tags.crowd not in valid_crowd:
        errors.append(f"tags.crowd: must be one of {sorted(valid_crowd)}, got '{record.tags.crowd}'")

    # distance_points validation
    for i, dp in enumerate(record.distance_points):
        if not np.isfinite(dp.u) or not np.isfinite(dp.v):
            errors.append(f"distance_points[{i}]: u,v must be finite")
        if dp.u < 0 or dp.v < 0:
            errors.append(f"distance_points[{i}]: u,v must be >= 0")
        # Image bounds checked by loader (requires intrinsics width/height)
        if dp.distance_m <= 0 or not np.isfinite(dp.distance_m):
            errors.append(f"distance_points[{i}]: distance_m must be > 0 and finite")
        if dp.method not in {"synthetic", "rangefinder", "lidar", "manual"}:
            errors.append(f"distance_points[{i}]: method must be one of 'synthetic','rangefinder','lidar','manual'")
        if dp.uncertainty_m < 0:
            errors.append(f"distance_points[{i}]: uncertainty_m must be >= 0")

    return errors