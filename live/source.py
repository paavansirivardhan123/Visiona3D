"""Camera and video-file sources for the live RGB runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

DEFAULT_CAMERA_FPS = 30.0


@dataclass(frozen=True)
class SourceInfo:
    """Native properties of an opened source."""

    label: str
    is_camera: bool
    width: int
    height: int
    fps: float
    frame_count: int


def parse_source(value: str) -> int | str:
    """Turn ``"0"`` into camera index 0 and everything else into a path."""
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


class VideoSource:
    """Thin OpenCV wrapper with explicit open/read/rewind semantics."""

    def __init__(
        self,
        source: int | str,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.source = source
        self.requested_width = None if width is None else int(width)
        self.requested_height = None if height is None else int(height)
        self.cap: cv2.VideoCapture | None = None
        self.info: SourceInfo | None = None

    def open(self) -> SourceInfo:
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source)
            label = f"camera:{self.source}"
            is_camera = True
        else:
            path = Path(str(self.source))
            if not path.is_file():
                raise FileNotFoundError(f"video source not found: {path}")
            cap = cv2.VideoCapture(str(path))
            label = str(path)
            is_camera = False
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"could not open video source {self.source!r}. For a camera, "
                "check that it is connected and not used by another program."
            )
        if is_camera:
            if self.requested_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_width))
            if self.requested_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_height))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0.0:
            fps = DEFAULT_CAMERA_FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(
                f"source {self.source!r} reported an invalid size {width}x{height}"
            )
        self.cap = cap
        self.info = SourceInfo(
            label=label,
            is_camera=is_camera,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
        return self.info

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            raise RuntimeError("video source is not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def rewind(self) -> None:
        if self.cap is None or self.info is None or self.info.is_camera:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def cv2_backend_note() -> str:
    """Short platform note printed once at startup."""
    if sys.platform.startswith("win"):
        return "OpenCV VideoCapture (Windows backend auto-selected)"
    return "OpenCV VideoCapture"


__all__ = [
    "DEFAULT_CAMERA_FPS",
    "SourceInfo",
    "VideoSource",
    "cv2_backend_note",
    "parse_source",
]
