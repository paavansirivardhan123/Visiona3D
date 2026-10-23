"""Resolution and intrinsic handling for processing live RGB frames.

The camera calibration describes the capture resolution. Live mode may
process a downscaled copy for speed, so the intrinsics must be scaled by the
same factors. Aspect ratio changes are rejected: a distorted frame would make
every metric depth value and every 3D anchor wrong.

No OpenCV import here; ``live.runner`` performs the actual resize.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import Intrinsics
from geometry.camera import scale_intrinsics

DEFAULT_PROC_WIDTH = 640  # TO TUNE ON REAL DATA: processing width cap
MIN_PROC_WIDTH = 160
MAX_ASPECT_DEVIATION = 0.02


@dataclass(frozen=True)
class ProcessingSize:
    """Processing resolution derived from a capture resolution."""

    width: int
    height: int
    source_width: int
    source_height: int

    @property
    def resized(self) -> bool:
        return (self.width, self.height) != (self.source_width, self.source_height)

    @property
    def scale_x(self) -> float:
        return self.width / float(self.source_width)

    @property
    def scale_y(self) -> float:
        return self.height / float(self.source_height)


def processing_size(
    width: int,
    height: int,
    max_width: int = DEFAULT_PROC_WIDTH,
) -> ProcessingSize:
    """Return an even, aspect-preserving processing size capped at ``max_width``."""
    src_w, src_h = int(width), int(height)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"capture size must be positive, got {width}x{height}")
    cap = int(max_width)
    if cap < MIN_PROC_WIDTH:
        raise ValueError(
            f"max_width must be at least {MIN_PROC_WIDTH}, got {max_width!r}"
        )
    if src_w <= cap:
        return ProcessingSize(src_w, src_h, src_w, src_h)

    scale = cap / float(src_w)
    out_w = cap - (cap % 2)
    out_h = int(round(src_h * scale))
    out_h -= out_h % 2
    if out_h <= 0:
        raise ValueError(f"capture size {width}x{height} cannot be downscaled")
    return ProcessingSize(out_w, out_h, src_w, src_h)


def scale_calibration(
    K: Intrinsics,
    processing: ProcessingSize,
) -> Intrinsics:
    """Scale calibrated intrinsics to the processing resolution."""
    if (K.width, K.height) == (processing.width, processing.height):
        return K
    sx = processing.width / float(K.width)
    sy = processing.height / float(K.height)
    deviation = abs(sx - sy) / max(sx, sy)
    if deviation > MAX_ASPECT_DEVIATION:
        raise ValueError(
            "processing resize would change the aspect ratio "
            f"({deviation:.1%}); calibrate at the capture resolution and keep "
            "the same aspect ratio"
        )
    return scale_intrinsics(K, sx, sy)


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """Build demo intrinsics from an assumed horizontal field of view.

    This is an explicit, labelled fallback for a camera that has not been
    calibrated. It is never presented as a measurement.
    """
    fov = float(hfov_deg)
    if not 30.0 <= fov <= 150.0:
        raise ValueError(f"hfov_deg must be in [30, 150], got {hfov_deg!r}")
    return Intrinsics.from_fov(width=int(width), height=int(height), hfov_deg=fov)


def provenance_is_measured(provenance: str | None) -> bool:
    """Return True only for intrinsics measured from a physical camera."""
    value = (provenance or "").strip().lower()
    return value in ("calibrated", "measured", "physical", "real")


__all__ = [
    "DEFAULT_PROC_WIDTH",
    "ProcessingSize",
    "intrinsics_from_hfov",
    "processing_size",
    "provenance_is_measured",
    "scale_calibration",
]
