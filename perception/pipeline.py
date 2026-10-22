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
        conformal_depth: Any | None = None,
        floor_threshold: float | None = None,
        tau0_m: float = 0.10,
        tau1: float = 0.02,
    ) -> None:
        self.seg = seg
        self.depth = depth
        self.detector = detector
        self.K = K
        self.mount = mount
        self.estimator = estimator
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.ransac_cfg = ransac_cfg
        self.conformal_depth = conformal_depth
        self.floor_threshold = floor_threshold
        self.tau0_m = float(tau0_m)
        self.tau1 = float(tau1)

    def run(
        self,
        image_rgb: np.ndarray,
        *,
        conformal_depth: Any | None = None,
        floor_threshold: float | None = None,
        tau0_m: float | None = None,
        tau1: float | None = None,
    ) -> FrameOutputs:
        """Run all stages. Never raises; failures become status codes."""
        timings: dict[str, float] = {}
        statuses: dict[str, str] = {}
        overall_start = time.perf_counter()

        cdepth = conformal_depth if conformal_depth is not None else self.conformal_depth
        fthresh = floor_threshold if floor_threshold is not None else self.floor_threshold
        t0 = tau0_m if tau0_m is not None else self.tau0_m
        t1 = tau1 if tau1 is not None else self.tau1

        seg_output: SegOutput | None = None
        depth_output: DepthOutput | None = None
        detections: list[Det2D] = []
        lift_result: LiftResult | None = None
        certified_final: np.ndarray | None = None
        vetoed: np.ndarray | None = None
        pessimistic_pointmap_g: np.ndarray | None = None

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

                # Optional conformal floor certification, geometric veto, and pessimistic point map
                if fthresh is not None:
                    try:
                        import math
                        from calibration.conformal_floor import certified_floor
                        from calibration.geometric_veto import veto_floor_mask
                        from geometry.camera import backproject
                        from geometry.frames import camera_to_ground

                        if hasattr(seg_output, "probs") and seg_output.probs is not None:
                            c_floor = certified_floor(seg_output.probs, fthresh)
                        else:
                            c_floor = labels == 1

                        if lift_result.point_map_g is not None:
                            certified_final, vetoed = veto_floor_mask(
                                c_floor, lift_result.point_map_g, tau0_m=t0, tau1=t1
                            )
                        else:
                            certified_final = c_floor
                            vetoed = np.zeros_like(c_floor, dtype=bool)

                        if cdepth is not None and getattr(cdepth, "_fitted", False):
                            c_pred = cdepth.predict(depth_output.depth_m.reshape(-1))
                            lower_depth = c_pred.lower.reshape(depth_output.depth_m.shape)
                            s_scale = lift_result.scale_result.s
                            scaled_lower_depth = lower_depth * s_scale
                            lower_pts_c = backproject(scaled_lower_depth, self.K)
                            if lift_result.plane_fit.status == FitStatus.OK:
                                n_norm = lift_result.plane_fit.n
                            else:
                                p_rad = math.radians(self.mount.pitch_deg)
                                n_norm = np.array([0.0, math.cos(p_rad), math.sin(p_rad)], dtype=np.float64)
                            lower_pts_g = camera_to_ground(lower_pts_c, n_norm, self.mount.h_known)
                            inv_lower = ~np.isfinite(scaled_lower_depth) | (scaled_lower_depth <= 0.0)
                            pessimistic_pointmap_g = lower_pts_g.astype(np.float32)
                            pessimistic_pointmap_g[inv_lower] = np.nan
                    except Exception as c_exc:  # noqa: BLE001
                        statuses["conformal"] = _error_code("conformal", c_exc)
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
            certified_final=certified_final,
            vetoed=vetoed,
            pessimistic_pointmap_g=pessimistic_pointmap_g,
        )

    def reset(self) -> None:
        """Reset the scale estimator state (keeps configuration)."""
        self.estimator = ScaleEstimator(self.mount, self.estimator.cfg)


__all__ = ["PerceptionPipeline", "align_depth_to_image", "FitStatus"]
