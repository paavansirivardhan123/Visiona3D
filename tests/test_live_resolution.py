"""Live resolution and intrinsics handling (no camera, no models)."""

from __future__ import annotations

import pytest

from core.config import Intrinsics
from live.imaging import (
    intrinsics_from_hfov,
    processing_size,
    provenance_is_measured,
    scale_calibration,
)


def test_processing_size_caps_width_and_preserves_aspect():
    spec = processing_size(3840, 2160, max_width=640)
    assert (spec.width, spec.height) == (640, 360)
    assert spec.resized
    assert abs(spec.width / spec.height - 16 / 9) < 0.01


def test_processing_size_keeps_small_frames_native():
    spec = processing_size(640, 360, max_width=640)
    assert (spec.width, spec.height) == (640, 360)
    assert not spec.resized


def test_processing_size_rounds_to_even_pixels():
    spec = processing_size(1042, 500, max_width=640)
    assert spec.width % 2 == 0
    assert spec.height % 2 == 0
    assert abs(spec.scale_x - spec.scale_y) < 0.01


def test_scale_calibration_scales_focal_and_principal_point():
    K = Intrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, width=1920, height=1080)
    spec = processing_size(1920, 1080, max_width=640)
    scaled = scale_calibration(K, spec)
    assert scaled.fx == pytest.approx(1000.0 * 640 / 1920)
    assert scaled.cx == pytest.approx(960.0 * 640 / 1920)
    assert (scaled.width, scaled.height) == (640, 360)


def test_scale_calibration_returns_input_when_size_matches():
    K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=180.0, width=640, height=360)
    spec = processing_size(640, 360, max_width=640)
    assert scale_calibration(K, spec) is K


def test_scale_calibration_rejects_aspect_change():
    K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    spec = processing_size(1280, 720, max_width=640)  # 16:9 processing of 4:3 K
    with pytest.raises(ValueError, match="aspect ratio"):
        scale_calibration(K, spec)


def test_hfov_validation_and_provenance():
    K = intrinsics_from_hfov(640, 360, 70.0)
    assert K.width == 640 and K.fx > 0
    with pytest.raises(ValueError):
        intrinsics_from_hfov(640, 360, 10.0)
    assert provenance_is_measured("calibrated")
    assert not provenance_is_measured("synthetic")
    assert not provenance_is_measured("assumed_hfov_demo")
    assert not provenance_is_measured(None)
