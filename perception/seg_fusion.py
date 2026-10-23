"""SegFormer semantics plus promptable-mask floor refinement.

The fusion is deliberately asymmetric about safety:

- SegFormer supplies the semantics and the probabilities. The refined output
  changes ``labels`` only; ``probs``, ``sky_mask`` and every probability-based
  decision (including conformal floor certification) stay SegFormer-only. A
  promptable mask can therefore never certify free space.
- A mask is accepted only when it overlaps the SegFormer floor mask by at
  least ``min_accept_iou``. Otherwise the SegFormer output is returned.
- Even an accepted mask cannot claim pixels that SegFormer confidently labels
  wall/obstacle/other (``veto_prob``) or any sky pixel.
- Promptable-mask failure (missing weights, runtime error, empty or malformed
  mask) is a fallback, not an exception: the base segmentation is returned
  with a reason code.

``MobileSAMBackend`` loads only a local checkpoint and imports the third-party
``mobile_sam`` package lazily. No test imports it or touches the network.

Every numeric default is a starting point, not a validated real-navigation
setting. TO TUNE ON REAL DATA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import time
from typing import Any, Protocol, runtime_checkable
import numpy as np

from core.config import MASK_FLOOR, MASK_OTHER
from perception.interfaces import SegOutput

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "weights" / "mobile-sam" / "mobile_sam.pt"

# TO TUNE ON REAL DATA.
DEFAULT_MIN_FLOOR_PROB = 0.60
DEFAULT_EROSION_PX = 2
DEFAULT_MAX_PROMPTS = 8
DEFAULT_MIN_PROMPT_SPACING_PX = 32
DEFAULT_MIN_ACCEPT_IOU = 0.50
DEFAULT_VETO_PROB = 0.60


@dataclass(frozen=True)
class PromptMaskOutput:
    """One promptable-mask result at the input-image resolution."""

    mask: np.ndarray
    score: float = float("nan")

    def validate(self, image_hw: tuple[int, int]) -> list[str]:
        """Return contract violations for the boolean mask."""
        errors: list[str] = []
        h, w = int(image_hw[0]), int(image_hw[1])
        mask = np.asarray(self.mask)
        if mask.shape != (h, w):
            errors.append(
                f"PromptMaskOutput.mask: expected {(h, w)}, got {mask.shape}"
            )
        if mask.dtype != np.bool_:
            errors.append(
                f"PromptMaskOutput.mask: expected bool, got {mask.dtype}"
            )
        return errors


@dataclass
class FusionResult:
    """Fused segmentation plus the decision record for one frame."""

    seg: SegOutput
    accepted: bool
    reason: str
    prompt_count: int
    iou: float = float("nan")
    sam_score: float = float("nan")
    added_floor_pixels: int = 0
    removed_floor_pixels: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FloorMaskDecision:
    """One promptable-mask application decision, independent of prompting.

    ``seg`` is None when the mask is rejected by the IoU gate; the caller then
    owns the fallback to the base segmentation.
    """

    seg: SegOutput | None
    accepted: bool
    reason: str
    iou: float
    fused_floor: np.ndarray | None


def apply_floor_mask(
    base: SegOutput,
    sam_mask: np.ndarray,
    *,
    min_accept_iou: float = DEFAULT_MIN_ACCEPT_IOU,
    veto_prob: float = DEFAULT_VETO_PROB,
) -> FloorMaskDecision:
    """Apply a validated, non-empty boolean mask to a base segmentation.

    This is the pure decision core shared by :meth:`SegSAMFusion.fuse` and the
    threshold tuning tool. It never mutates ``base`` and never raises on a
    rejected mask: the IoU gate returns ``accepted=False`` with ``seg=None``.

    The mask must have the base label shape and boolean dtype. Sky, and pixels
    whose base non-floor probability is at least ``veto_prob``, are never
    claimed as floor.
    """
    if not (0.0 <= float(min_accept_iou) <= 1.0):
        raise ValueError(
            f"min_accept_iou must be in [0, 1], got {min_accept_iou!r}"
        )
    if not (0.0 <= float(veto_prob) <= 1.0):
        raise ValueError(f"veto_prob must be in [0, 1], got {veto_prob!r}")
    mask = np.asarray(sam_mask)
    base_labels = np.asarray(base.labels)
    if mask.dtype != np.bool_:
        raise ValueError(f"sam_mask must be bool, got {mask.dtype}")
    if mask.shape != base_labels.shape:
        raise ValueError(
            f"sam_mask shape {mask.shape} != base labels shape {base_labels.shape}"
        )

    base_floor = base_labels == MASK_FLOOR
    union = int(np.count_nonzero(base_floor | mask))
    intersection = int(np.count_nonzero(base_floor & mask))
    iou = float(intersection / union) if union else float("nan")
    if not np.isfinite(iou) or iou < float(min_accept_iou):
        return FloorMaskDecision(
            seg=None,
            accepted=False,
            reason=f"low_iou:{iou:.4f}",
            iou=iou,
            fused_floor=None,
        )

    nonfloor_prob = np.max(np.asarray(base.probs)[..., 1:], axis=-1)
    veto = np.isfinite(nonfloor_prob) & (nonfloor_prob >= float(veto_prob))
    sky_mask = np.asarray(base.sky_mask, dtype=bool)
    fused_floor = mask & ~veto & ~sky_mask

    labels = base_labels.copy()
    labels[base_floor] = MASK_OTHER
    labels[fused_floor] = MASK_FLOOR
    labels[sky_mask] = MASK_OTHER
    fused = SegOutput(
        labels=labels,
        probs=np.asarray(base.probs),
        sky_mask=sky_mask,
    )
    return FloorMaskDecision(
        seg=fused,
        accepted=True,
        reason="accepted",
        iou=iou,
        fused_floor=fused_floor,
    )


def _erode_bool(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Binary erosion with an 8-neighbour structuring element, numpy only."""
    if pixels <= 0:
        return mask.copy()
    result = mask.copy()
    h, w = result.shape
    for _ in range(int(pixels)):
        eroded = result.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.zeros_like(result)
                src_y = slice(max(0, -dy), h + min(0, -dy))
                src_x = slice(max(0, -dx), w + min(0, -dx))
                dst_y = slice(max(0, dy), h + min(0, dy))
                dst_x = slice(max(0, dx), w + min(0, dx))
                shifted[dst_y, dst_x] = result[src_y, src_x]
                eroded &= shifted
        result = eroded
        if not result.any():
            break
    return result


def _even_picks(count: int, limit: int) -> np.ndarray:
    """Choose at most ``limit`` evenly spread indices from ``0..count-1``."""
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    picks = np.round(np.linspace(0, count - 1, num=limit)).astype(np.int64)
    return np.unique(picks)


def select_floor_prompts(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    min_floor_prob: float = DEFAULT_MIN_FLOOR_PROB,
    erosion_px: int = DEFAULT_EROSION_PX,
    max_points: int = DEFAULT_MAX_PROMPTS,
    min_spacing_px: int = DEFAULT_MIN_PROMPT_SPACING_PX,
) -> np.ndarray:
    """Select deterministic, spread-out positive prompts on trusted floor.

    Candidates are floor-labelled pixels whose floor probability is at least
    ``min_floor_prob``, eroded by ``erosion_px`` so prompts avoid object
    boundaries. If erosion would remove all candidates the un-eroded set is
    used (a prompt is a proposal, never a certification).

    Returns:
        ``(N, 2)`` int64 array of ``(x, y)`` pixel coordinates, N >= 0.
    """
    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)
    if labels_arr.ndim != 2:
        raise ValueError(f"labels must be 2D (H, W), got {labels_arr.shape}")
    if probs_arr.shape != labels_arr.shape + (4,):
        raise ValueError(
            f"probs must be {labels_arr.shape + (4,)}, got {probs_arr.shape}"
        )
    if not (0.0 <= float(min_floor_prob) <= 1.0):
        raise ValueError(f"min_floor_prob must be in [0, 1], got {min_floor_prob!r}")
    if int(erosion_px) < 0:
        raise ValueError(f"erosion_px must be non-negative, got {erosion_px!r}")
    if int(max_points) < 1:
        raise ValueError(f"max_points must be >= 1, got {max_points!r}")
    if int(min_spacing_px) < 0:
        raise ValueError(f"min_spacing_px must be non-negative, got {min_spacing_px!r}")

    candidate = (
        (labels_arr == MASK_FLOOR)
        & np.isfinite(probs_arr[..., 0])
        & (probs_arr[..., 0] >= float(min_floor_prob))
    )
    eroded = _erode_bool(candidate, int(erosion_px))
    if eroded.any():
        candidate = eroded
    elif not candidate.any():
        return np.zeros((0, 2), dtype=np.int64)

    rows, cols = np.where(candidate)
    spacing = int(min_spacing_px)
    if spacing <= 0:
        order = np.lexsort((cols, rows))
        chosen = order[_even_picks(order.size, int(max_points))]
    else:
        cell_rows = rows // spacing
        cell_cols = cols // spacing
        cells_wide = int(cols.max() // spacing) + 1
        cell_key = cell_rows * cells_wide + cell_cols
        centre_r = (cell_rows + 0.5) * spacing
        centre_c = (cell_cols + 0.5) * spacing
        distance_sq = (rows - centre_r) ** 2 + (cols - centre_c) ** 2
        order = np.lexsort((distance_sq, cell_key))
        sorted_key = cell_key[order]
        first_in_cell = np.ones(order.size, dtype=bool)
        first_in_cell[1:] = sorted_key[1:] != sorted_key[:-1]
        representatives = order[first_in_cell]
        chosen = representatives[_even_picks(representatives.size, int(max_points))]

    return np.stack([cols[chosen], rows[chosen]], axis=1).astype(np.int64)


@runtime_checkable
class PromptMaskBackend(Protocol):
    """Promptable-mask backend contract (positive point prompts only)."""

    def predict(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> PromptMaskOutput:
        """Return a boolean object mask at the input-image resolution."""


class MobileSAMBackend:
    """Lazy MobileSAM ViT-T wrapper using a local checkpoint only."""

    MODEL_ID = "mobile_sam.pt (MobileSAM ViT-T)"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = (
            Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        )
        self.device = str(device)
        self._predictor = None

    def preflight(self) -> list[str]:
        """Return actionable problems without importing the heavy package."""
        errors: list[str] = []
        if not self.checkpoint.exists():
            errors.append(
                f"MobileSAM checkpoint not found at {self.checkpoint}; fetch it with "
                "python scripts/fetch_models.py --dry-run (the checkpoint is a manual "
                "download, see docs/perception_backends.md)"
            )
        if importlib.util.find_spec("mobile_sam") is None:
            errors.append(
                "The third-party MobileSAM package is not installed; install it from "
                "https://github.com/ChaoningZhang/MobileSAM"
            )
        return errors

    def _load(self) -> None:
        if self._predictor is not None:
            return
        if not self.checkpoint.exists():
            raise RuntimeError(
                f"MobileSAM checkpoint not found at {self.checkpoint}. Fetch it with "
                "python scripts/fetch_models.py --dry-run; the checkpoint is a manual "
                "download listed in docs/perception_backends.md."
            )
        try:
            from mobile_sam import SamPredictor, sam_model_registry  # lazy
        except ImportError as exc:
            raise RuntimeError(
                "The third-party MobileSAM package is not installed. Install it from "
                "https://github.com/ChaoningZhang/MobileSAM; this tool does not "
                "download packages or use the network."
            ) from exc

        model = sam_model_registry["vit_t"](checkpoint=str(self.checkpoint))
        model.to(self.device)
        model.eval()
        self._predictor = SamPredictor(model)

    def predict(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> PromptMaskOutput:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"MobileSAM expects RGB uint8 (H, W, 3), got {image.shape} {image.dtype}"
            )
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
            raise ValueError(
                f"points_xy must be (N, 2) with N >= 1, got {points.shape}"
            )
        if not np.isfinite(points).all():
            raise ValueError("points_xy must be finite")
        h, w = image.shape[:2]
        if (
            (points[:, 0] < 0.0).any()
            or (points[:, 0] > w - 1).any()
            or (points[:, 1] < 0.0).any()
            or (points[:, 1] > h - 1).any()
        ):
            raise ValueError(f"points_xy outside image bounds {w}x{h}")

        self._load()
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            point_coords=points.astype(np.float32),
            point_labels=np.ones(points.shape[0], dtype=np.int32),
            multimask_output=False,
        )
        masks_arr = np.asarray(masks)
        if masks_arr.ndim == 2:
            mask = masks_arr
        elif masks_arr.ndim == 3:
            mask = masks_arr[0]
        elif masks_arr.ndim == 4:
            mask = masks_arr[0, 0]
        else:
            raise RuntimeError(f"unexpected MobileSAM mask shape {masks_arr.shape}")
        scores_arr = np.asarray(scores).reshape(-1)
        score = float(scores_arr[0]) if scores_arr.size else float("nan")
        return PromptMaskOutput(mask=mask.astype(bool), score=score)


class SegSAMFusion:
    """Refine a segmentation backend's floor labels with a promptable mask."""

    def __init__(
        self,
        seg: Any,
        sam: Any,
        *,
        min_floor_prob: float = DEFAULT_MIN_FLOOR_PROB,
        erosion_px: int = DEFAULT_EROSION_PX,
        max_prompts: int = DEFAULT_MAX_PROMPTS,
        min_prompt_spacing_px: int = DEFAULT_MIN_PROMPT_SPACING_PX,
        min_accept_iou: float = DEFAULT_MIN_ACCEPT_IOU,
        veto_prob: float = DEFAULT_VETO_PROB,
    ) -> None:
        for name, value in (
            ("min_floor_prob", min_floor_prob),
            ("min_accept_iou", min_accept_iou),
            ("veto_prob", veto_prob),
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if int(max_prompts) < 1:
            raise ValueError(f"max_prompts must be >= 1, got {max_prompts!r}")
        if int(erosion_px) < 0:
            raise ValueError(f"erosion_px must be non-negative, got {erosion_px!r}")
        if int(min_prompt_spacing_px) < 0:
            raise ValueError(
                f"min_prompt_spacing_px must be non-negative, got {min_prompt_spacing_px!r}"
            )

        self.seg = seg
        self.sam = sam
        self.min_floor_prob = float(min_floor_prob)
        self.erosion_px = int(erosion_px)
        self.max_prompts = int(max_prompts)
        self.min_prompt_spacing_px = int(min_prompt_spacing_px)
        self.min_accept_iou = float(min_accept_iou)
        self.veto_prob = float(veto_prob)
        self.last_result: FusionResult | None = None

    def predict(self, image_rgb: np.ndarray) -> SegOutput:
        """Run the base segmenter and refine its floor labels."""
        base = self.seg.predict(image_rgb)
        return self.fuse(image_rgb, base).seg

    def fuse(self, image_rgb: np.ndarray, base: SegOutput) -> FusionResult:
        """Refine an existing base segmentation. Never raises on SAM failure."""
        start = time.perf_counter()
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"SegSAMFusion expects RGB uint8 (H, W, 3), got {image.shape} {image.dtype}"
            )
        base_errors = base.validate()
        if base_errors:
            raise ValueError("base segmentation is invalid: " + "; ".join(base_errors))
        h, w = image.shape[:2]
        if base.labels.shape != (h, w):
            raise ValueError(
                f"base labels shape {base.labels.shape} != image {(h, w)}"
            )

        prompt_start = time.perf_counter()
        prompts = select_floor_prompts(
            base.labels,
            base.probs,
            min_floor_prob=self.min_floor_prob,
            erosion_px=self.erosion_px,
            max_points=self.max_prompts,
            min_spacing_px=self.min_prompt_spacing_px,
        )
        prompt_ms = (time.perf_counter() - prompt_start) * 1000.0
        if prompts.shape[0] == 0:
            return self._fallback(
                base, "fallback:no_prompts", 0, prompt_ms, start
            )

        sam_start = time.perf_counter()
        try:
            candidate = self.sam.predict(image, prompts)
        except Exception as exc:  # noqa: BLE001 - SAM is an optional refiner
            sam_ms = (time.perf_counter() - sam_start) * 1000.0
            reason = f"fallback:sam_error:{type(exc).__name__}:{exc}"
            return self._fallback(
                base, reason, int(prompts.shape[0]), prompt_ms, start, sam_ms=sam_ms
            )
        sam_ms = (time.perf_counter() - sam_start) * 1000.0

        mask_errors = candidate.validate((h, w))
        if mask_errors:
            reason = "fallback:invalid_sam:" + "; ".join(mask_errors)
            return self._fallback(
                base, reason, int(prompts.shape[0]), prompt_ms, start, sam_ms=sam_ms
            )
        sam_mask = np.asarray(candidate.mask, dtype=bool)
        if not sam_mask.any():
            return self._fallback(
                base,
                "fallback:empty_sam",
                int(prompts.shape[0]),
                prompt_ms,
                start,
                sam_ms=sam_ms,
                sam_score=candidate.score,
            )

        decision = apply_floor_mask(
            base,
            sam_mask,
            min_accept_iou=self.min_accept_iou,
            veto_prob=self.veto_prob,
        )
        if not decision.accepted:
            return self._fallback(
                base,
                f"fallback:{decision.reason}",
                int(prompts.shape[0]),
                prompt_ms,
                start,
                sam_ms=sam_ms,
                iou=decision.iou,
                sam_score=candidate.score,
            )

        fused = decision.seg
        if fused is None:  # defensive: accepted decisions always carry a seg
            raise RuntimeError("accepted floor-mask decision without segmentation")
        fused_errors = fused.validate()
        if fused_errors:
            return self._fallback(
                base,
                "fallback:invalid_fused:" + "; ".join(fused_errors),
                int(prompts.shape[0]),
                prompt_ms,
                start,
                sam_ms=sam_ms,
                iou=decision.iou,
                sam_score=candidate.score,
            )

        base_floor = np.asarray(base.labels) == MASK_FLOOR
        fused_floor = decision.fused_floor
        if fused_floor is None:  # defensive: accepted decisions always carry a mask
            raise RuntimeError("accepted floor-mask decision without fused floor")
        result = FusionResult(
            seg=fused,
            accepted=True,
            reason="accepted",
            prompt_count=int(prompts.shape[0]),
            iou=decision.iou,
            sam_score=float(candidate.score),
            added_floor_pixels=int(np.count_nonzero(fused_floor & ~base_floor)),
            removed_floor_pixels=int(np.count_nonzero(base_floor & ~fused_floor)),
            timings_ms={
                "prompt": prompt_ms,
                "sam": sam_ms,
                "total": (time.perf_counter() - start) * 1000.0,
            },
        )
        self.last_result = result
        return result

    def _fallback(
        self,
        base: SegOutput,
        reason: str,
        prompt_count: int,
        prompt_ms: float,
        start: float,
        *,
        sam_ms: float = 0.0,
        iou: float = float("nan"),
        sam_score: float = float("nan"),
    ) -> FusionResult:
        result = FusionResult(
            seg=base,
            accepted=False,
            reason=reason,
            prompt_count=int(prompt_count),
            iou=float(iou),
            sam_score=float(sam_score),
            timings_ms={
                "prompt": prompt_ms,
                "sam": float(sam_ms),
                "total": (time.perf_counter() - start) * 1000.0,
            },
        )
        self.last_result = result
        return result


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_EROSION_PX",
    "DEFAULT_MAX_PROMPTS",
    "DEFAULT_MIN_ACCEPT_IOU",
    "DEFAULT_MIN_FLOOR_PROB",
    "DEFAULT_MIN_PROMPT_SPACING_PX",
    "DEFAULT_VETO_PROB",
    "FloorMaskDecision",
    "FusionResult",
    "MobileSAMBackend",
    "PromptMaskBackend",
    "PromptMaskOutput",
    "SegSAMFusion",
    "apply_floor_mask",
    "select_floor_prompts",
]
