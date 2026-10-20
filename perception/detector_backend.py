"""Real Ultralytics YOLO detector wrapper (lazy import).

Uses the repository's existing ``yolo11n.pt``. Missing weights raise a clear
error naming the file instead of triggering a download. No test imports
ultralytics.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from perception.interfaces import Det2D

REPO_ROOT = Path(__file__).resolve().parent.parent


class YoloDetectorBackend:
    """Ultralytics YOLO backend returning Det2D objects in pixel coordinates."""

    def __init__(
        self,
        weights: str | Path = "yolo11n.pt",
        conf: float = 0.35,
        device: str | None = None,
    ) -> None:
        path = Path(weights)
        if not path.is_absolute() and not path.exists():
            path = REPO_ROOT / path
        self.weights = path
        self.conf = float(conf)
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.weights.exists():
            raise RuntimeError(
                f"YOLO weights not found: {self.weights}. Fetch with: "
                "python scripts/fetch_models.py --dry-run "
                "(yolo11n.pt is downloaded separately or placed in the repository root)."
            )
        from ultralytics import YOLO  # lazy

        self._model = YOLO(str(self.weights))

    def predict(self, image_rgb: np.ndarray) -> list[Det2D]:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"YOLO backend expects RGB (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"YOLO backend expects uint8 RGB, got {image.dtype}")
        self._load()

        predict_kwargs = {"conf": self.conf, "verbose": False}
        if self.device is not None:
            predict_kwargs["device"] = self.device
        results = self._model.predict(image, **predict_kwargs)

        detections: list[Det2D] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, confs, classes):
                detections.append(
                    Det2D(
                        class_name=str(names.get(int(class_id), class_id)),
                        box_xyxy=np.asarray(box, dtype=np.float64),
                        conf=float(score),
                    )
                )
        return detections


__all__ = ["YoloDetectorBackend"]
