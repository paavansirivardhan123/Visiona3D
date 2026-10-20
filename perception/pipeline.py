"""Synchronous, non-raising perception pipeline.

``PerceptionPipeline.run`` calls segmentation, depth and detection backends,
sanitises depth, then lifts floor/wall/obstacle pixels to metric 3D using the
existing geometry package. Backend failures never propagate: they are reported
as status codes in :class:`FrameOutputs`.
"""

from __future__ import annotations

import time
from typing import Any, Optional
import numpy as np

from core.config import Intrinsics, MountConfig, RansacConfig
from geometry.ground_plane import FitStatus
from geometry.lift import LiftResult, lift_frame
from geometry.scale import ScaleEstimator
from perception.interfaces import DepthOutput, Det2D, FrameOutputs, SegOutput
from perception.postprocess import map_back, plain_resize_info, sanitize_depth


def align_depth_to_image(depth_m: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Defensively map a model-resolution depth map back to the original image.

    Backends normally return original-resolution depth; this fallback keeps the
    pipeline correct when a backend returns a model-resolution map.
    """
    arr = np.asarray(depth_m, dtype=np.float32)
    h, w = int(image_hw[0]), int(image_hw[1])
    if arr.shape == (h, w):
        return arr
    if arr.ndim != 2:
        raise ValueError(f"depth must be 2D, got {arr.shape}")
    info = plain_resize_info(h, w, arr.shape[0], arr.shape[1])
    return map_back(arr, info, kind="depth")


def _error_code(stage: str, exc: BaseException) -> str:
    return f"error:{stage}:{type(exc).__name__}:{exc}"


class PerceptionPipeline:
    """Synchronous perception pipeline with per-stage timings and status codes."""

    def __init__(
        self,
        seg: Any,
        depth: Any,
        detector: Any,
        K: Intrinsics,
        mount: MountConfig,
        estimator: ScaleEstimator,
        rng: Optional[np.random.Generator] = None,
        ransac_cfg: Optional[RansacConfig] = None,
    ) -> None:
        self.seg = seg
        self.depth = depth
        self.detector = detector
        self.K = K
        self.mount = mount
        self.estimator = estimator
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.ransac_cfg = ransac_cfg

    def run(self, image_rgb: np.ndarray) -> FrameOutputs:
        """Run all stages. Never raises; failures become status codes."""
        timings: dict[str, float] = {}
        statuses: dict[str, str] = {}
        overall_start = time.perf_counter()

        seg_output: SegOutput | None = None
        depth_output: DepthOutput | None = None
        detections: list[Det2D] = []
        lift_result: LiftResult | None = None

        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            image_hw = image.shape[:2] if image.ndim >= 2 else (0, 0)
            statuses["input"] = (
                "error:input:expected RGB uint8 (H, W, 3), got "
                f"{image.shape} {image.dtype}"
            )
            for stage in ("seg", "depth", "detector", "lift"):
                statuses[stage] = "skipped:invalid input"
            return FrameOutputs(
                image_size=(int(image_hw[0]), int(image_hw[1])),
                seg=None,
                depth=None,
                detections=[],
                lift=None,
                timings={"total": (time.perf_counter() - overall_start) * 1000.0},
                statuses=statuses,
            )

        h, w = int(image.shape[0]), int(image.shape[1])

        # --- Segmentation ---
        start = time.perf_counter()
        try:
            if self.seg is None:
                statuses["seg"] = "skipped:backend is None"
            else:
                candidate = self.seg.predict(image)
                errors = candidate.validate()
                if errors:
                    statuses["seg"] = "invalid:" + "; ".join(errors)
                elif candidate.labels.shape != (h, w):
                    statuses["seg"] = (
                        f"invalid:label shape {candidate.labels.shape} != image {(h, w)}"
                    )
                else:
                    seg_output = candidate
                    statuses["seg"] = "OK"
        except Exception as exc:  # noqa: BLE001 - pipeline contract is non-raising
            statuses["seg"] = _error_code("seg", exc)
        timings["seg"] = (time.perf_counter() - start) * 1000.0

        # --- Depth ---
        start = time.perf_counter()
        try:
            if self.depth is None:
                statuses["depth"] = "skipped:backend is None"
            else:
                candidate = self.depth.predict(image)
                errors = candidate.validate()
                if errors:
                    statuses["depth"] = "invalid:" + "; ".join(errors)
                else:
                    aligned = align_depth_to_image(candidate.depth_m, (h, w))
                    if aligned.shape != (h, w):
                        statuses["depth"] = (
                            f"invalid:depth shape {aligned.shape} != image {(h, w)}"
                        )
                    else:
                        sanitized = sanitize_depth(
                            aligned,
                            candidate.max_range_m,
                            seg_output.sky_mask if seg_output is not None else None,
                        )
                        depth_output = DepthOutput(
                            depth_m=sanitized,
                            max_range_m=candidate.max_range_m,
                            model_id=candidate.model_id,
                        )
                        statuses["depth"] = "OK"
        except Exception as exc:  # noqa: BLE001
            statuses["depth"] = _error_code("depth", exc)
        timings["depth"] = (time.perf_counter() - start) * 1000.0

        # --- Detection ---
        start = time.perf_counter()
        try:
            if self.detector is None:
                statuses["detector"] = "skipped:backend is None"
            else:
                candidate_dets = list(self.detector.predict(image))
                det_errors: list[str] = []
                valid_dets: list[Det2D] = []
                for det in candidate_dets:
                    errs = det.validate((h, w))
                    if errs:
                        det_errors.extend(errs)
                    else:
                        valid_dets.append(det)
                detections = valid_dets
                statuses["detector"] = "OK" if not det_errors else "invalid:" + "; ".join(det_errors)
        except Exception as exc:  # noqa: BLE001
            statuses["detector"] = _error_code("detector", exc)
        timings["detector"] = (time.perf_counter() - start) * 1000.0

        # --- Lift ---
        start = time.perf_counter()
        try:
            if seg_output is None or depth_output is None:
                statuses["lift"] = "skipped:seg or depth unavailable"
            else:
                labels = seg_output.labels
                floor_mask = labels == 1
                other_masks = {
                    "wall": labels == 2,
                    "obstacle": labels == 3,
                }
                lift_result = lift_frame(
                    depth_output.depth_m,
                    floor_mask,
                    other_masks,
                    self.K,
                    self.mount,
                    self.estimator,
                    rng=self.rng,
                    ransac_cfg=self.ransac_cfg,
                )
                statuses["lift"] = lift_result.plane_fit.status.value
                statuses["scale"] = lift_result.scale_result.status
        except Exception as exc:  # noqa: BLE001
            statuses["lift"] = _error_code("lift", exc)
        timings["lift"] = (time.perf_counter() - start) * 1000.0

        timings["total"] = (time.perf_counter() - overall_start) * 1000.0
        return FrameOutputs(
            image_size=(h, w),
            seg=seg_output,
            depth=depth_output,
            detections=detections,
            lift=lift_result,
            timings=timings,
            statuses=statuses,
        )

    def reset(self) -> None:
        """Reset the scale estimator state (keeps configuration)."""
        self.estimator = ScaleEstimator(self.mount, self.estimator.cfg)


__all__ = ["PerceptionPipeline", "align_depth_to_image", "FitStatus"]
