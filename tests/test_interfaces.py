"""P1: interface contracts accept valid outputs and reject six corruptions."""

from __future__ import annotations

import numpy as np

from perception.interfaces import DepthOutput, Det2D, SegOutput


def _valid_seg(h: int = 4, w: int = 5) -> SegOutput:
    labels = np.ones((h, w), dtype=np.uint8)
    labels[:, w // 2 :] = 2
    probs = np.zeros((h, w, 4), dtype=np.float32)
    probs[np.arange(h)[:, None], np.arange(w)[None, :], labels.astype(np.int64) - 1] = 1.0
    return SegOutput(labels=labels, probs=probs, sky_mask=np.zeros((h, w), dtype=bool))


def _valid_depth(h: int = 4, w: int = 5) -> DepthOutput:
    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[0, 0] = np.nan
    return DepthOutput(depth_m=depth, max_range_m=10.0, model_id="fake-depth")


def _valid_det() -> Det2D:
    return Det2D(class_name="person", box_xyxy=np.array([2.0, 3.0, 20.0, 30.0]), conf=0.8)


def test_p1_valid_outputs_pass():
    assert _valid_seg().validate() == []
    assert _valid_depth().validate() == []
    assert _valid_det().validate((48, 64)) == []


def test_p1_wrong_shape_specific_message():
    seg = _valid_seg()
    seg.labels = seg.labels[0]
    errors = seg.validate()
    assert any("expected 2D (H, W)" in error for error in errors), errors


def test_p1_wrong_dtype_specific_message():
    seg = _valid_seg()
    seg.labels = seg.labels.astype(np.float32)
    errors = seg.validate()
    assert any("expected uint8" in error for error in errors), errors


def test_p1_probs_not_summing_specific_message():
    seg = _valid_seg()
    seg.probs = seg.probs * 0.5
    errors = seg.validate()
    assert any("must sum to 1" in error for error in errors), errors


def test_p1_label_out_of_range_specific_message():
    seg = _valid_seg()
    seg.labels[0, 0] = 7
    errors = seg.validate()
    assert any("class id 7 outside valid range 0..4" in error for error in errors), errors


def test_p1_negative_depth_without_nan_specific_message():
    depth = _valid_depth()
    depth.depth_m[1, 1] = -1.5
    errors = depth.validate()
    assert any("negative depth values must be NaN" in error for error in errors), errors


def test_p1_box_outside_image_specific_message():
    det = Det2D(class_name="person", box_xyxy=np.array([5.0, 5.0, 70.0, 50.0]), conf=0.9)
    errors = det.validate((48, 64))
    assert any("outside image bounds 64x48" in error for error in errors), errors
