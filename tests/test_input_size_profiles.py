"""Reduced model input sizes (no models loaded)."""

from __future__ import annotations

import pytest

from perception.depth_backend import DepthAnythingBackend, depth_processor_size
from perception.seg_backend import SegFormerBackend


def test_depth_size_is_patch_aligned_and_keeps_landscape_aspect():
    size = depth_processor_size(360, 640, 266)
    assert size == {"height": 266, "width": 476}
    assert size["height"] % 14 == 0 and size["width"] % 14 == 0
    aspect = size["width"] / size["height"]
    assert aspect == pytest.approx(640 / 360, rel=0.02)


def test_depth_size_works_for_portrait_frames():
    size = depth_processor_size(640, 360, 266)
    assert size["height"] == 266
    assert size["width"] % 14 == 0 and size["width"] > 0
    assert 0.5 < size["width"] / size["height"] < 0.65


def test_depth_size_rejects_empty_frame():
    with pytest.raises(ValueError):
        depth_processor_size(0, 640, 266)


def test_depth_backend_validates_input_size():
    with pytest.raises(ValueError):
        DepthAnythingBackend(input_size=250)  # not a multiple of 14
    with pytest.raises(ValueError):
        DepthAnythingBackend(input_size=100)  # below 140
    assert DepthAnythingBackend(input_size=266).input_size == 266


def test_seg_backend_validates_input_size():
    with pytest.raises(ValueError):
        SegFormerBackend(input_size=250)  # not a multiple of 32
    with pytest.raises(ValueError):
        SegFormerBackend(input_size=64)  # below 128
    assert SegFormerBackend(input_size=256).input_size == 256
