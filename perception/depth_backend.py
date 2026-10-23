"""Real Depth Anything V2 metric-depth wrapper (lazy heavy imports).

Indoor and outdoor metric Small checkpoints are selected by the ``variant``
argument, defaulting to the ``VISIONA_DEPTH_VARIANT`` environment flag. Weights
are loaded from a local directory or the Hugging Face cache only; missing
weights raise a clear error naming what to fetch.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

from perception.interfaces import DepthOutput

INDOOR_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
OUTDOOR_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
VARIANT_ENV = "VISIONA_DEPTH_VARIANT"


def depth_processor_size(
    frame_height: int,
    frame_width: int,
    input_size: int,
) -> dict[str, int]:
    """Return a height/width processor size preserving aspect at 14-px steps."""
    height = int(frame_height)
    width = int(frame_width)
    size = int(input_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"frame size must be positive, got {frame_width}x{frame_height}")
    out_w = int(round(size * (width / float(height)) / 14.0)) * 14
    out_w = max(14, out_w)
    return {"height": size, "width": out_w}


class DepthAnythingBackend:
    """Depth Anything V2 metric depth backend (indoor or outdoor Small)."""

    INDOOR_MODEL_ID = INDOOR_MODEL_ID
    OUTDOOR_MODEL_ID = OUTDOOR_MODEL_ID

    def __init__(
        self,
        variant: str | None = None,
        weights_dir: str | Path | None = None,
        device: str = "cpu",
        max_range_m: float = 10.0,
        input_size: int | None = None,
    ) -> None:
        selected = (variant or os.environ.get(VARIANT_ENV, "indoor")).strip().lower()
        if selected not in ("indoor", "outdoor"):
            raise ValueError(
                f"depth variant must be 'indoor' or 'outdoor', got {selected!r} "
                f"(set {VARIANT_ENV} or pass variant=)"
            )
        if input_size is not None:
            size = int(input_size)
            if size % 14 != 0 or not 140 <= size <= 1036:
                raise ValueError(
                    "input_size must be a multiple of 14 in [140, 1036] because "
                    f"Depth Anything V2 uses 14-pixel patches; got {input_size!r}"
                )
        self.variant = selected
        self.model_id = INDOOR_MODEL_ID if selected == "indoor" else OUTDOOR_MODEL_ID
        self.weights_dir = Path(weights_dir) if weights_dir is not None else None
        self.device = str(device)
        self.max_range_m = float(max_range_m)
        self.input_size = None if input_size is None else int(input_size)
        self._model = None
        self._processor = None

    def _source(self) -> str:
        if self.weights_dir is not None:
            if not (self.weights_dir / "config.json").exists():
                raise RuntimeError(
                    f"Depth weights not found in {self.weights_dir}. Expected config.json. "
                    "Fetch with: python scripts/fetch_models.py --dry-run"
                )
            return str(self.weights_dir)
        return self.model_id

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # lazy

        source = self._source()
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                source, local_files_only=True
            )
            self._model = AutoModelForDepthEstimation.from_pretrained(
                source, local_files_only=True
            ).to(self.device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 - re-raised with fetch guidance
            raise RuntimeError(
                f"Depth Anything weights for {self.model_id} are not available locally or "
                "in the Hugging Face cache. Fetch with: python scripts/fetch_models.py "
                f"--dry-run then python scripts/fetch_models.py --yes. Original error: {exc}"
            ) from exc
        self._torch = torch

    def predict(self, image_rgb: np.ndarray) -> DepthOutput:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Depth backend expects RGB (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Depth backend expects uint8 RGB, got {image.dtype}")
        self._load()

        torch = self._torch
        h, w = image.shape[:2]
        processor_kwargs: dict[str, object] = {}
        if self.input_size is not None:
            processor_kwargs["size"] = depth_processor_size(h, w, self.input_size)
        inputs = self._processor(
            images=image, return_tensors="pt", **processor_kwargs
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            predicted = self._model(**inputs).predicted_depth
            predicted = torch.nn.functional.interpolate(
                predicted.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
        depth = predicted[0].detach().cpu().numpy().astype(np.float32)
        return DepthOutput(
            depth_m=depth,
            max_range_m=self.max_range_m,
            model_id=self.model_id,
        )


__all__ = [
    "DepthAnythingBackend",
    "INDOOR_MODEL_ID",
    "OUTDOOR_MODEL_ID",
    "VARIANT_ENV",
    "depth_processor_size",
]
