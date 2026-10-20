"""P3-P5: probabilities, resize/intrinsics mapping and depth sanitising."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Intrinsics
from geometry.camera import scale_intrinsics
from perception.postprocess import (
    labels_from_probs,
    letterbox_info,
    map_back,
    merge_class_probs,
    resize_nearest,
    sanitize_depth,
    stable_softmax,
)


def test_p3_softmax_rows_sum_to_one_and_stable():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(7, 5))
    probs = stable_softmax(logits)
    assert probs.dtype == np.float32
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)

    extreme = np.array([[10000.0, 5000.0, -10000.0], [-10000.0, 10000.0, 10000.0]])
    extreme_probs = stable_softmax(extreme)
    assert np.isfinite(extreme_probs).all()
    assert np.allclose(extreme_probs.sum(axis=-1), 1.0, atol=1e-6)
    # Max-subtraction must preserve the true distribution, not just finiteness:
    # exp(10000) would overflow to inf without it, collapsing the row.
    assert extreme_probs[0, 0] > 0.99
    assert extreme_probs[0, 1] < 0.01
    assert np.allclose(extreme_probs[1, 1:], 0.5, atol=1e-6)


def test_p3_merged_floor_probability_is_sum_of_mapped_channels():
    logits = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    class_map = {0: 1, 1: 1, 2: 2}  # two source classes -> floor, one -> wall
    softmax = stable_softmax(logits)[0]
    merged = merge_class_probs(logits, class_map)[0]
    assert merged[0] == pytest.approx(float(softmax[0] + softmax[1]), abs=1e-7)
    assert merged[1] == pytest.approx(float(softmax[2]), abs=1e-7)
    assert merged[2] == pytest.approx(0.0)
    assert merged[3] == pytest.approx(0.0)
    assert np.isclose(merged.sum(), 1.0)


def test_p3_argmax_labels_agree_with_merged_probabilities():
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(6, 8, 7))
    class_map = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 2, 6: 4}
    merged = merge_class_probs(logits, class_map)
    labels = labels_from_probs(merged)
    assert np.array_equal(labels, (np.argmax(merged, axis=-1) + 1).astype(np.uint8))


def test_p4_scale_intrinsics_and_letterbox_hand_computed():
    K = Intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    half = scale_intrinsics(K, 0.5, 0.5)
    assert (half.fx, half.fy, half.cx, half.cy) == pytest.approx((500.0, 500.0, 320.0, 180.0))

    # 1280x720 letterboxed into 640x640: scale=0.5, top pad=140, cy=180+140=320.
    info = letterbox_info(720, 1280, 640, 640)
    assert info.scale == pytest.approx(0.5)
    assert (info.pad_x, info.pad_y) == (0, 140)
    letterboxed = scale_intrinsics(K, info.scale, info.scale, info.pad_x, info.pad_y)
    assert (letterboxed.fx, letterboxed.fy, letterboxed.cx, letterboxed.cy) == pytest.approx(
        (500.0, 500.0, 320.0, 320.0)
    )

    # Tall 400x800 image into 640x640: scale=0.8, left pad=160, cx=320*0.8+160=416.
    tall = Intrinsics(fx=500.0, fy=500.0, cx=200.0, cy=400.0, width=400, height=800)
    tall_info = letterbox_info(800, 400, 640, 640)
    assert (tall_info.pad_x, tall_info.pad_y) == (160, 0)
    tall_scaled = scale_intrinsics(tall, tall_info.scale, tall_info.scale, tall_info.pad_x, tall_info.pad_y)
    assert tall_scaled.cx == pytest.approx(320.0)
    assert tall_scaled.cy == pytest.approx(320.0)


def test_p4_mask_resize_map_back_within_one_pixel():
    h, w = 240, 320
    original = np.zeros((h, w), dtype=np.uint8)
    original[60:120, 80:200] = 1
    info = letterbox_info(h, w, 256, 256)

    resized = resize_nearest(original, info.new_h, info.new_w)
    canvas = np.zeros((info.dst_h, info.dst_w), dtype=np.uint8)
    canvas[info.pad_y : info.pad_y + info.new_h, info.pad_x : info.pad_x + info.new_w] = resized
    mapped = map_back(canvas, info, kind="label")

    assert mapped.shape == (h, w)
    src_rows, src_cols = np.where(original > 0)
    dst_rows, dst_cols = np.where(mapped > 0)
    assert abs(int(src_rows.min()) - int(dst_rows.min())) <= 1
    assert abs(int(src_rows.max()) - int(dst_rows.max())) <= 1
    assert abs(int(src_cols.min()) - int(dst_cols.min())) <= 1
    assert abs(int(src_cols.max()) - int(dst_cols.max())) <= 1


def test_p4_depth_map_back_constant_plane_and_nan_holes():
    h, w = 60, 80
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[20:30, 30:40] = np.nan
    info = letterbox_info(h, w, 64, 64)

    resized = map_back(
        _letterbox_array(depth, info, kind="depth"), info, kind="depth"
    )
    assert resized.shape == (h, w)
    assert np.allclose(resized[np.isfinite(resized)], 4.0, atol=1e-6)
    assert np.isnan(resized[24, 34])
    assert np.isnan(resized).sum() >= 1


def _letterbox_array(array: np.ndarray, info, kind: str) -> np.ndarray:
    from perception.postprocess import resize_bilinear_nanaware

    if kind == "depth":
        resized = resize_bilinear_nanaware(array, info.new_h, info.new_w)
        fill = np.nan
    else:
        resized = resize_nearest(array, info.new_h, info.new_w)
        fill = 0
    canvas = np.full((info.dst_h, info.dst_w), fill, dtype=resized.dtype)
    canvas[info.pad_y : info.pad_y + info.new_h, info.pad_x : info.pad_x + info.new_w] = resized
    return canvas


def test_p5_sanitize_depth_all_invalid_categories():
    depth = np.array(
        [
            [1.0, np.nan, np.inf],
            [0.0, -1.0, 11.0],
            [5.0, 10.0, 2.0],
        ],
        dtype=np.float32,
    )
    sky = np.zeros((3, 3), dtype=bool)
    sky[2, 2] = True
    cleaned = sanitize_depth(depth, max_range_m=10.0, sky_mask=sky)

    assert np.isnan(cleaned[0, 1])  # NaN
    assert np.isnan(cleaned[0, 2])  # inf
    assert np.isnan(cleaned[1, 0])  # zero
    assert np.isnan(cleaned[1, 1])  # negative
    assert np.isnan(cleaned[1, 2])  # beyond max range
    assert np.isnan(cleaned[2, 2])  # sky

    # Valid pixels are untouched bit-for-bit.
    assert cleaned[0, 0] == np.float32(1.0)
    assert cleaned[2, 0] == np.float32(5.0)
    assert cleaned[2, 1] == np.float32(10.0)
