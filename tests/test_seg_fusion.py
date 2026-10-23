"""F1-F7: promptable floor refinement on analytic fakes.

No test loads MobileSAM, torch, transformers, a video, a camera or the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Intrinsics, MASK_FLOOR, MASK_OTHER, MountConfig
from geometry.scale import ScaleEstimator
from perception.fakes import (
    FakeDepthBackend,
    FakeDetector,
    FakePromptMaskBackend,
    FakeSegBackend,
)
from perception.interfaces import SegOutput
from perception.pipeline import PerceptionPipeline
from perception.seg_fusion import (
    MobileSAMBackend,
    SegSAMFusion,
    apply_floor_mask,
    select_floor_prompts,
)

H, W = 120, 160
K = Intrinsics.from_fov(width=W, height=H, hfov_deg=75.0)
MOUNT = MountConfig(h_known=1.4, pitch_deg=15.0)
SCENE = dict(
    K=K,
    H=H,
    W=W,
    h=1.4,
    pitch_deg=15.0,
    wall_distance=4.0,
    boxes=[(-0.2, 0.2, 2.4, 2.7, 0.4, 0.8)],
    seed=5,
)
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


def _base(**kwargs) -> FakeSegBackend:
    return FakeSegBackend(**SCENE, **kwargs)


class _OtherSegBackend:
    def predict(self, image_rgb: np.ndarray) -> SegOutput:
        h, w = image_rgb.shape[:2]
        labels = np.full((h, w), MASK_OTHER, dtype=np.uint8)
        probs = np.zeros((h, w, 4), dtype=np.float32)
        probs[..., 3] = 1.0
        return SegOutput(labels=labels, probs=probs, sky_mask=np.zeros((h, w), dtype=bool))


def test_f1_accepted_mask_refines_floor_and_preserves_probabilities():
    base_backend = _base(wall_as_floor_fraction=0.20)
    base = base_backend.predict(IMAGE)
    assert np.any((base.labels == MASK_FLOOR) & base_backend.wall_mask)

    sam = FakePromptMaskBackend(**SCENE, mode="floor")
    result = SegSAMFusion(_base(), sam).fuse(IMAGE, base)

    assert result.accepted is True, result.reason
    assert result.reason == "accepted"
    assert result.iou >= 0.5
    assert result.prompt_count >= 1
    assert sam.calls == 1
    assert result.seg.validate() == []
    assert np.array_equal(result.seg.labels == MASK_FLOOR, base_backend.floor_mask)
    assert result.removed_floor_pixels > 0
    assert result.added_floor_pixels == 0
    # Probabilities and sky stay SegFormer-only by design.
    assert np.array_equal(result.seg.probs, base.probs)
    assert np.array_equal(result.seg.sky_mask, base.sky_mask)
    for key in ("prompt", "sam", "total"):
        assert result.timings_ms[key] >= 0.0


def test_f2_disagreeing_mask_is_rejected_and_base_is_returned():
    base = _base().predict(IMAGE)
    sam = FakePromptMaskBackend(**SCENE, mode="wall")
    result = SegSAMFusion(_base(), sam).fuse(IMAGE, base)

    assert result.accepted is False
    assert result.reason.startswith("fallback:low_iou")
    assert result.iou == pytest.approx(0.0)
    assert np.array_equal(result.seg.labels, base.labels)
    assert np.array_equal(result.seg.probs, base.probs)
    assert sam.calls == 1


@pytest.mark.parametrize(
    ("mode", "reason_prefix"),
    [
        ("empty", "fallback:empty_sam"),
        ("raise", "fallback:sam_error:RuntimeError"),
    ],
)
def test_f3_empty_or_failing_sam_falls_back_without_raising(mode: str, reason_prefix: str):
    base = _base().predict(IMAGE)
    sam = FakePromptMaskBackend(**SCENE, mode=mode)
    result = SegSAMFusion(_base(), sam).fuse(IMAGE, base)

    assert result.accepted is False
    assert result.reason.startswith(reason_prefix)
    assert np.array_equal(result.seg.labels, base.labels)
    assert sam.calls == 1


def test_f4_no_trusted_floor_skips_sam_entirely():
    sam = FakePromptMaskBackend(**SCENE)
    fusion = SegSAMFusion(_OtherSegBackend(), sam)
    output = fusion.predict(IMAGE)

    assert output.validate() == []
    assert fusion.last_result is not None
    assert fusion.last_result.accepted is False
    assert fusion.last_result.reason == "fallback:no_prompts"
    assert fusion.last_result.prompt_count == 0
    assert sam.calls == 0


def test_f5_confident_nonfloor_pixels_veto_sam_spill():
    base = _base().predict(IMAGE)
    sam = FakePromptMaskBackend(**SCENE, mode="floor_plus_wall")
    fusion = SegSAMFusion(_base(), sam, min_accept_iou=0.0)
    result = fusion.fuse(IMAGE, base)

    assert result.accepted is True, result.reason
    assert np.array_equal(result.seg.labels == MASK_FLOOR, base.labels == MASK_FLOOR)
    assert result.added_floor_pixels == 0


def test_f6_prompt_selection_is_deterministic_spread_and_inside_eroded_floor():
    base = _base().predict(IMAGE)
    first = select_floor_prompts(
        base.labels, base.probs, erosion_px=2, max_points=8, min_spacing_px=16
    )
    second = select_floor_prompts(
        base.labels, base.probs, erosion_px=2, max_points=8, min_spacing_px=16
    )
    assert np.array_equal(first, second)
    assert 1 <= first.shape[0] <= 8
    assert np.all(base.labels[first[:, 1], first[:, 0]] == MASK_FLOOR)
    cells = {(int(y) // 16, int(x) // 16) for x, y in first}
    assert len(cells) == first.shape[0]

    labels = np.full((20, 20), MASK_OTHER, dtype=np.uint8)
    probs = np.zeros((20, 20, 4), dtype=np.float32)
    probs[..., 3] = 1.0
    labels[5:15, 5:15] = MASK_FLOOR
    probs[5:15, 5:15] = 0.0
    probs[5:15, 5:15, 0] = 1.0
    eroded = select_floor_prompts(
        labels, probs, erosion_px=2, max_points=16, min_spacing_px=0
    )
    assert eroded.shape[0] > 0
    assert np.all((eroded[:, 0] >= 7) & (eroded[:, 0] <= 12))
    assert np.all((eroded[:, 1] >= 7) & (eroded[:, 1] <= 12))

    low_confidence = probs.copy()
    low_confidence[..., 0] *= 0.4
    low_confidence[..., 3] += 0.6
    assert select_floor_prompts(labels, low_confidence).shape == (0, 2)


def test_f7_fusion_is_a_valid_pipeline_segmentation_backend():
    fusion = SegSAMFusion(_base(), FakePromptMaskBackend(**SCENE))
    pipeline = PerceptionPipeline(
        seg=fusion,
        depth=FakeDepthBackend(**SCENE),
        detector=FakeDetector(**SCENE),
        K=K,
        mount=MOUNT,
        estimator=ScaleEstimator(MOUNT),
        rng=np.random.default_rng(0),
    )
    output = pipeline.run(IMAGE)

    assert output.statuses["seg"] == "OK", output.statuses
    assert output.statuses["lift"] == "OK", output.statuses
    assert output.seg is not None
    assert output.seg.validate() == []
    assert fusion.last_result is not None
    assert fusion.last_result.accepted is True


def test_f8_mobile_sam_missing_checkpoint_is_reported_before_any_import(tmp_path):
    missing = tmp_path / "mobile_sam.pt"
    backend = MobileSAMBackend(checkpoint=missing)
    errors = backend.preflight()
    assert any("mobile_sam.pt" in error for error in errors)

    with pytest.raises(RuntimeError, match="MobileSAM checkpoint"):
        backend.predict(IMAGE, np.array([[W // 2, H - 2]], dtype=np.int64))


def test_f9_invalid_configuration_is_rejected():
    sam = FakePromptMaskBackend(**SCENE)
    with pytest.raises(ValueError):
        SegSAMFusion(_base(), sam, min_accept_iou=1.5)
    with pytest.raises(ValueError):
        SegSAMFusion(_base(), sam, max_prompts=0)
    base = _base().predict(IMAGE)
    with pytest.raises(ValueError):
        select_floor_prompts(base.labels, base.probs, max_points=0)
    with pytest.raises(ValueError):
        select_floor_prompts(base.labels, base.probs, min_floor_prob=2.0)


def test_f13_apply_floor_mask_is_the_fuse_decision_core():
    base_backend = _base(wall_as_floor_fraction=0.20)
    base = base_backend.predict(IMAGE)
    sam = FakePromptMaskBackend(**SCENE, mode="floor_plus_wall")
    mask = sam.floor_mask | sam.wall_mask

    decision = apply_floor_mask(base, mask, min_accept_iou=0.0, veto_prob=0.6)
    fused = SegSAMFusion(_base(), sam, min_accept_iou=0.0, veto_prob=0.6).fuse(
        IMAGE, base
    )
    assert decision.accepted is True
    assert decision.seg is not None
    assert decision.fused_floor is not None
    assert fused.accepted is True
    assert np.array_equal(fused.seg.labels, decision.seg.labels)
    assert np.array_equal(fused.seg.probs, decision.seg.probs)
    assert np.array_equal(fused.seg.sky_mask, decision.seg.sky_mask)
    assert fused.iou == pytest.approx(decision.iou)
    base_floor = base.labels == MASK_FLOOR
    assert fused.added_floor_pixels == int(
        np.count_nonzero(decision.fused_floor & ~base_floor)
    )
    assert fused.removed_floor_pixels == int(
        np.count_nonzero(base_floor & ~decision.fused_floor)
    )

    rejected = apply_floor_mask(base, mask, min_accept_iou=1.0, veto_prob=0.6)
    assert rejected.accepted is False
    assert rejected.seg is None
    assert rejected.reason.startswith("low_iou")
    rejected_fuse = SegSAMFusion(_base(), sam, min_accept_iou=1.0).fuse(IMAGE, base)
    assert rejected_fuse.accepted is False
    assert rejected_fuse.reason.startswith("fallback:low_iou")
    assert np.array_equal(rejected_fuse.seg.labels, base.labels)

    with pytest.raises(ValueError):
        apply_floor_mask(base, mask.astype(np.uint8))
    with pytest.raises(ValueError):
        apply_floor_mask(base, np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError):
        apply_floor_mask(base, mask, veto_prob=2.0)
