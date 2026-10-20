"""Real SegFormer semantic-segmentation wrapper (lazy heavy imports).

Weights are loaded from a local directory or the Hugging Face cache only.
``from_pretrained(..., local_files_only=True)`` is deliberate: a missing model
raises a clear error naming the weights instead of silently downloading.
No test imports torch or transformers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import numpy as np

from core.config import MASK_OTHER
from perception.class_map import build_class_map, normalize_class_name
from perception.interfaces import SegOutput
from perception.postprocess import labels_from_probs, merge_class_probs

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerBackend:
    """SegFormer-B0 ADE20K backend returning merged 4-class probabilities."""

    MODEL_ID = MODEL_ID

    def __init__(
        self,
        model_id: str = MODEL_ID,
        weights_dir: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = str(model_id)
        self.weights_dir = Path(weights_dir) if weights_dir is not None else None
        self.device = str(device)
        self._model = None
        self._processor = None
        self._class_map: dict[int, int] | None = None
        self._sky_ids: list[int] = []

    def _source(self) -> str:
        if self.weights_dir is not None:
            if not (self.weights_dir / "config.json").exists():
                raise RuntimeError(
                    f"SegFormer weights not found in {self.weights_dir}. Expected config.json. "
                    f"Fetch with: python scripts/fetch_models.py --dry-run"
                )
            return str(self.weights_dir)
        return self.model_id

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # lazy

        source = self._source()
        try:
            self._processor = SegformerImageProcessor.from_pretrained(
                source, local_files_only=True
            )
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                source, local_files_only=True
            ).to(self.device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 - re-raised with fetch guidance
            raise RuntimeError(
                f"SegFormer weights for {self.model_id} are not available locally or in the "
                "Hugging Face cache. Fetch with: python scripts/fetch_models.py --dry-run "
                f"then python scripts/fetch_models.py --yes. Original error: {exc}"
            ) from exc

        id2label: Mapping[int, str] = self._model.config.id2label
        self._class_map = build_class_map(id2label)
        self._sky_ids = [
            int(idx)
            for idx, name in id2label.items()
            if normalize_class_name(name) == "sky"
        ]
        self._torch = torch

    def predict(self, image_rgb: np.ndarray) -> SegOutput:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"SegFormer expects RGB (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"SegFormer expects uint8 RGB, got {image.dtype}")
        self._load()

        torch = self._torch
        h, w = image.shape[:2]
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False
            )
        logits_np = logits[0].permute(1, 2, 0).detach().cpu().numpy()

        probs = merge_class_probs(logits_np, self._class_map or {})
        labels = labels_from_probs(probs)
        if self._sky_ids:
            sky_mask = np.isin(np.argmax(logits_np, axis=-1), np.asarray(self._sky_ids))
        else:
            sky_mask = np.zeros((h, w), dtype=bool)
        # Sky is excluded from label metrics and is never allowed to become floor.
        labels[sky_mask] = MASK_OTHER
        return SegOutput(labels=labels, probs=probs, sky_mask=sky_mask)


__all__ = ["SegFormerBackend", "MODEL_ID"]
