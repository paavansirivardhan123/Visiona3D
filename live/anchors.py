"""Per-detection 3D anchors and stable track IDs for live mode.

The safety stack's ``Det2D`` has no identity, and ``object_anchor`` has no
caller outside tests. This module bridges both gaps without touching legacy
app code: detections are converted to the legacy ``Detection`` shape for the
existing IoU tracker, then grouped into 3D ground-frame anchors.

The anchor distance is the 10th-percentile depth inside the box
(``geometry.lift.object_anchor`` default), so a partially occluded object is
placed at its nearest visible surface rather than its centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from core.config import Intrinsics, MountConfig
from core.detection import Detection
from geometry.lift import AnchorStatus, object_anchor
from perception.interfaces import Det2D
from perception.tracker import ObjectTracker

DEFAULT_ANCHOR_QUANTILE = 0.10  # TO TUNE ON REAL DATA
DEFAULT_MIN_ANCHOR_PIXELS = 10  # TO TUNE ON REAL DATA


@dataclass(frozen=True)
class TrackedDetection:
    """A detection paired with the tracker's stable identity."""

    track_id: int
    det: Det2D


@dataclass(frozen=True)
class ObjectAnchor:
    """3D object observation in the ground frame, ready for HUD and DB."""

    track_id: int
    class_name: str
    conf: float
    box_xyxy: tuple[float, float, float, float]
    distance_m: float
    position_g: tuple[float, float, float]
    status: str

    @property
    def lateral_m(self) -> float:
        """Signed X offset in metres (positive = right)."""
        return float(self.position_g[0])

    @property
    def forward_m(self) -> float:
        """Forward Y offset in metres."""
        return float(self.position_g[1])

    @property
    def bearing_deg(self) -> float:
        """Bearing in the ground frame, positive to the right."""
        return math.degrees(math.atan2(self.position_g[0], self.position_g[1]))

    def to_row(self) -> dict:
        """Plain types for SQLite binding."""
        return {
            "track_id": int(self.track_id),
            "class_name": str(self.class_name),
            "conf": float(self.conf),
            "distance_m": float(self.distance_m),
            "anchor_x": float(self.position_g[0]),
            "anchor_y": float(self.position_g[1]),
            "anchor_z": float(self.position_g[2]),
            "anchor_status": str(self.status),
            "box_x1": float(self.box_xyxy[0]),
            "box_y1": float(self.box_xyxy[1]),
            "box_x2": float(self.box_xyxy[2]),
            "box_y2": float(self.box_xyxy[3]),
        }


def box_mask(
    shape: tuple[int, int],
    box_xyxy: tuple[float, float, float, float] | np.ndarray,
    margin_px: int = 0,
) -> np.ndarray:
    """Return a boolean mask for one clipped detection box."""
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"image shape must be positive, got {shape}")
    box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.isfinite(box).all():
        raise ValueError(f"box_xyxy must be four finite values, got {box_xyxy!r}")
    x1, y1, x2, y2 = box.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box must have positive size, got {box_xyxy!r}")
    m = max(0, int(margin_px))
    x1 = max(0, int(math.floor(x1)) - m)
    y1 = max(0, int(math.floor(y1)) - m)
    x2 = min(w, int(math.ceil(x2)) + m)
    y2 = min(h, int(math.ceil(y2)) + m)
    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def assign_track_ids(
    detections: list[Det2D],
    frame: np.ndarray,
    tracker: ObjectTracker | None = None,
) -> list[TrackedDetection]:
    """Assign stable IDs with the repository's existing IoU tracker.

    The tracker expects legacy ``Detection`` objects. Depth is left unknown
    here; the anchor step supplies the metric distance afterwards.
    """
    if not detections:
        return []
    track = tracker if tracker is not None else ObjectTracker()
    legacy: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = np.asarray(det.box_xyxy, dtype=np.float64).tolist()
        legacy.append(
            Detection(
                label=str(det.class_name),
                direction="FRONT",
                confidence=float(det.conf),
                box=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            )
        )
    updated = track.update(legacy, frame)
    paired: list[TrackedDetection] = []
    for det, old in zip(detections, updated):
        track_id = old.track_id
        if track_id is None:
            raise RuntimeError("tracker returned a detection without a track_id")
        paired.append(TrackedDetection(track_id=int(track_id), det=det))
    return paired


def compute_object_anchors(
    tracked: list[TrackedDetection],
    depth_m: np.ndarray,
    K: Intrinsics,
    mount: MountConfig,
    plane: object | None = None,
    q: float = DEFAULT_ANCHOR_QUANTILE,
    min_pixels: int = DEFAULT_MIN_ANCHOR_PIXELS,
) -> list[ObjectAnchor]:
    """Compute one ground-frame anchor per tracked detection."""
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got {depth.shape}")
    observations: list[ObjectAnchor] = []
    for item in tracked:
        det = item.det
        mask = box_mask(depth.shape, det.box_xyxy)
        anchor = object_anchor(
            depth,
            mask,
            K,
            mount,
            plane=plane,
            q=q,
            min_pixels=min_pixels,
        )
        status = anchor.status.value if isinstance(anchor.status, AnchorStatus) else str(anchor.status)
        observations.append(
            ObjectAnchor(
                track_id=int(item.track_id),
                class_name=str(det.class_name),
                conf=float(det.conf),
                box_xyxy=tuple(float(v) for v in np.asarray(det.box_xyxy).tolist()),
                distance_m=float(anchor.distance_m),
                position_g=tuple(float(v) for v in np.asarray(anchor.position_g).tolist()),
                status=status,
            )
        )
    return observations


def nearest_object(
    observations: list[ObjectAnchor],
    class_name: str | None = None,
    require_ok: bool = True,
) -> ObjectAnchor | None:
    """Return the nearest anchor, optionally restricted to one class."""
    wanted = None if class_name is None else str(class_name).strip().lower()
    candidates = [
        obs
        for obs in observations
        if (not require_ok or obs.status == AnchorStatus.OK.value)
        and (wanted is None or obs.class_name.strip().lower() == wanted)
        and math.isfinite(obs.distance_m)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda obs: obs.distance_m)


__all__ = [
    "DEFAULT_ANCHOR_QUANTILE",
    "DEFAULT_MIN_ANCHOR_PIXELS",
    "ObjectAnchor",
    "TrackedDetection",
    "assign_track_ids",
    "box_mask",
    "compute_object_anchors",
    "nearest_object",
]
