"""Fake perception backends driven by the analytic synthetic renderer.

Every test and the ``--fake`` smoke mode use these backends: no model, no
network, no camera and no video files. Corruptions are seeded and exactly
counted so tests can assert on individual pixels.
"""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np

from core.config import Intrinsics, MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER
from perception.interfaces import Det2D, DepthOutput, SegOutput
from perception.postprocess import resize_bilinear_nanaware
from perception.seg_fusion import PromptMaskOutput


class _FakeSceneBase:
    """Renders one analytic scene once and shares its ground truth."""

    def __init__(
        self,
        K: Union[Intrinsics, np.ndarray],
        H: int,
        W: int,
        h: float = 1.4,
        pitch_deg: float = 15.0,
        wall_distance: float | None = 4.0,
        boxes: Sequence[Sequence[float]] = (),
        seed: int = 0,
    ) -> None:
        from tests.synthetic import render_scene

        self.K = K
        self.H = int(H)
        self.W = int(W)
        self.seed = int(seed)
        self.depth_true, self.floor_mask, self.wall_mask, self.box_masks, self.gt = render_scene(
            K,
            self.H,
            self.W,
            h,
            pitch_deg,
            wall_distance,
            boxes,
            seed=seed,
        )

    def _check_image(self, image_rgb: np.ndarray) -> None:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape != (self.H, self.W, 3):
            raise ValueError(
                f"fake backend expected RGB image {(self.H, self.W, 3)}, got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"fake backend expected uint8 image, got {image.dtype}")

    def _corrupt_count(self, fraction: float, population: int) -> int:
        fraction = float(fraction)
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"corruption fraction must be in [0, 1], got {fraction}")
        return int(round(fraction * population))


class FakeDepthBackend(_FakeSceneBase):
    """Depth backend with a scale factor, relative noise and outliers."""

    def __init__(
        self,
        K: Union[Intrinsics, np.ndarray],
        H: int,
        W: int,
        h: float = 1.4,
        pitch_deg: float = 15.0,
        wall_distance: float | None = 4.0,
        boxes: Sequence[Sequence[float]] = (),
        seed: int = 0,
        k: float = 1.0,
        rel_noise: float = 0.0,
        outlier_fraction: float = 0.0,
        max_range_m: float = 10.0,
        output_hw: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(K, H, W, h, pitch_deg, wall_distance, boxes, seed)
        self.k = float(k)
        self.rel_noise = float(rel_noise)
        self.outlier_fraction = float(outlier_fraction)
        self.max_range_m = float(max_range_m)
        self.output_hw = output_hw

    def predict(self, image_rgb: np.ndarray) -> DepthOutput:
        self._check_image(image_rgb)
        rng = np.random.default_rng(self.seed)

        depth = self.depth_true * self.k
        valid = depth > 0.0
        if self.rel_noise > 0.0 and valid.any():
            noise = rng.normal(0.0, self.rel_noise, size=depth.shape)
            depth[valid] = depth[valid] * (1.0 + noise[valid])

        n_outliers = self._corrupt_count(self.outlier_fraction, int(np.sum(valid)))
        if n_outliers > 0:
            rows, cols = np.where(valid)
            picks = rng.choice(rows.size, size=n_outliers, replace=False)
            factors = rng.uniform(0.3, 3.0, size=n_outliers)
            depth[rows[picks], cols[picks]] *= factors

        depth = depth.astype(np.float32)
        if self.output_hw is not None and tuple(self.output_hw) != depth.shape:
            out_h, out_w = int(self.output_hw[0]), int(self.output_hw[1])
            depth = resize_bilinear_nanaware(
                np.where(depth > 0.0, depth, np.nan), out_h, out_w
            )
            depth = np.nan_to_num(depth, nan=0.0)

        return DepthOutput(
            depth_m=depth.astype(np.float32),
            max_range_m=self.max_range_m,
            model_id="fake-depth",
        )


class FakeSegBackend(_FakeSceneBase):
    """Segmentation backend with exact-pixel label corruptions.

    ``floor_as_wall_fraction`` relabels floor pixels as wall (safe direction:
    the floor plane fit is not polluted). ``wall_as_floor_fraction`` relabels
    wall pixels as floor (unsafe direction: vertical wall points enter the
    ground-plane fit).
    """

    def __init__(
        self,
        K: Union[Intrinsics, np.ndarray],
        H: int,
        W: int,
        h: float = 1.4,
        pitch_deg: float = 15.0,
        wall_distance: float | None = 4.0,
        boxes: Sequence[Sequence[float]] = (),
        seed: int = 0,
        wall_as_floor_fraction: float = 0.0,
        floor_as_wall_fraction: float = 0.0,
        label_noise_rate: float = 0.0,
        sky_threshold: float = 0.0,
    ) -> None:
        super().__init__(K, H, W, h, pitch_deg, wall_distance, boxes, seed)
        self.wall_as_floor_fraction = float(wall_as_floor_fraction)
        self.floor_as_wall_fraction = float(floor_as_wall_fraction)
        self.label_noise_rate = float(label_noise_rate)
        self.sky_threshold = float(sky_threshold)
        self.sky_mask = self.depth_true <= self.sky_threshold

    def predict(self, image_rgb: np.ndarray) -> SegOutput:
        self._check_image(image_rgb)
        rng = np.random.default_rng(self.seed)

        labels = np.full((self.H, self.W), MASK_OTHER, dtype=np.uint8)
        labels[self.floor_mask] = MASK_FLOOR
        labels[self.wall_mask] = MASK_WALL
        for box_mask in self.box_masks:
            labels[box_mask] = MASK_OBSTACLE
        labels[self.sky_mask] = MASK_OTHER

        wall_pixels = np.where(self.wall_mask)
        n_flip = self._corrupt_count(self.wall_as_floor_fraction, wall_pixels[0].size)
        if n_flip:
            picks = rng.choice(wall_pixels[0].size, size=n_flip, replace=False)
            labels[wall_pixels[0][picks], wall_pixels[1][picks]] = MASK_FLOOR

        floor_pixels = np.where(labels == MASK_FLOOR)
        n_floor_flip = self._corrupt_count(self.floor_as_wall_fraction, floor_pixels[0].size)
        if n_floor_flip:
            picks = rng.choice(floor_pixels[0].size, size=n_floor_flip, replace=False)
            labels[floor_pixels[0][picks], floor_pixels[1][picks]] = MASK_WALL

        valid_pixels = np.where(~self.sky_mask)
        n_noise = self._corrupt_count(self.label_noise_rate, valid_pixels[0].size)
        if n_noise:
            picks = rng.choice(valid_pixels[0].size, size=n_noise, replace=False)
            noisy = rng.integers(1, 5, size=n_noise, dtype=np.uint8)
            labels[valid_pixels[0][picks], valid_pixels[1][picks]] = noisy

        probs = np.zeros((self.H, self.W, 4), dtype=np.float32)
        probs[
            np.arange(self.H)[:, None],
            np.arange(self.W)[None, :],
            labels.astype(np.int64) - 1,
        ] = 1.0
        return SegOutput(labels=labels, probs=probs, sky_mask=self.sky_mask.copy())


class FakeDetector(_FakeSceneBase):
    """Detector backend emitting the scene's rendered boxes, optionally dropped."""

    def __init__(
        self,
        K: Union[Intrinsics, np.ndarray],
        H: int,
        W: int,
        h: float = 1.4,
        pitch_deg: float = 15.0,
        wall_distance: float | None = 4.0,
        boxes: Sequence[Sequence[float]] = (),
        seed: int = 0,
        drop_fraction: float = 0.0,
        class_name: str = "obstacle",
        conf: float = 0.9,
    ) -> None:
        super().__init__(K, H, W, h, pitch_deg, wall_distance, boxes, seed)
        self.drop_fraction = float(drop_fraction)
        self.class_name = str(class_name)
        self.conf = float(conf)

    def predict(self, image_rgb: np.ndarray) -> list[Det2D]:
        self._check_image(image_rgb)
        rng = np.random.default_rng(self.seed)

        detections: list[Det2D] = []
        for mask in self.box_masks:
            rows, cols = np.where(mask)
            if rows.size == 0:
                continue
            detections.append(
                Det2D(
                    class_name=self.class_name,
                    box_xyxy=np.array(
                        [cols.min(), rows.min(), cols.max() + 1, rows.max() + 1],
                        dtype=np.float64,
                    ),
                    conf=self.conf,
                )
            )

        n_drop = self._corrupt_count(self.drop_fraction, len(detections))
        if n_drop:
            drop_idx = set(rng.choice(len(detections), size=n_drop, replace=False).tolist())
            detections = [det for idx, det in enumerate(detections) if idx not in drop_idx]
        return detections


class FakePromptMaskBackend(_FakeSceneBase):
    """Analytic promptable-mask backend for fusion tests.

    Modes:
        ``floor``: exact analytic floor (agrees with a correct segmentation).
        ``floor_plus_wall``: floor and wall (unsafe boundary spill).
        ``wall``: wall only (disagrees completely).
        ``empty``: no mask pixels.
        ``raise``: raises ``RuntimeError`` to exercise the error fallback.

    Prompt coordinates are validated and recorded in ``last_points``; the mask
    itself is analytic so tests never depend on a model or the prompt content.
    """

    MODES = ("floor", "floor_plus_wall", "wall", "empty", "raise")

    def __init__(
        self,
        K: Union[Intrinsics, np.ndarray],
        H: int,
        W: int,
        h: float = 1.4,
        pitch_deg: float = 15.0,
        wall_distance: float | None = 4.0,
        boxes: Sequence[Sequence[float]] = (),
        seed: int = 0,
        mode: str = "floor",
        score: float = 0.9,
    ) -> None:
        super().__init__(K, H, W, h, pitch_deg, wall_distance, boxes, seed)
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = str(mode)
        self.score = float(score)
        self.calls = 0
        self.last_points: np.ndarray | None = None

    def predict(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> PromptMaskOutput:
        self._check_image(image_rgb)
        points = np.asarray(points_xy)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
            raise ValueError(f"points_xy must be (N, 2) with N >= 1, got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("points_xy must be finite")
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] > self.W - 1).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] > self.H - 1).any()
        ):
            raise ValueError(f"points_xy outside image bounds {self.W}x{self.H}")

        self.calls += 1
        self.last_points = points.copy()
        if self.mode == "raise":
            raise RuntimeError("deliberate fake promptable-mask failure")
        if self.mode == "empty":
            mask = np.zeros((self.H, self.W), dtype=bool)
        elif self.mode == "floor":
            mask = self.floor_mask.copy()
        elif self.mode == "floor_plus_wall":
            mask = self.floor_mask | self.wall_mask
        else:
            mask = self.wall_mask.copy()
        return PromptMaskOutput(mask=mask, score=self.score)


__all__ = [
    "FakeSegBackend",
    "FakeDepthBackend",
    "FakeDetector",
    "FakePromptMaskBackend",
]
