"""Tests for the synthetic checkerboard calibration generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_synthetic_calibration import (
    SYNTHETIC_WARNING,
    generate_synthetic_calibration,
    main,
)


def test_recovers_generated_camera_and_marks_provenance():
    result, images, reference_k = generate_synthetic_calibration(
        640, 480, hfov_deg=70.0, seed=3
    )

    assert len(images) == 20
    assert images[0].shape == (480, 640)
    assert images[0].min() == 0
    assert images[0].max() == 255
    assert result["image_size"] == [640, 480]
    assert result["provenance"] == "synthetic"
    assert result["synthetic"] is True
    assert result["assumed_hfov_deg"] == pytest.approx(70.0)
    assert result["warning"] == SYNTHETIC_WARNING

    assert abs(result["fx"] - reference_k[0, 0]) / reference_k[0, 0] < 0.005
    assert abs(result["fy"] - reference_k[1, 1]) / reference_k[1, 1] < 0.005
    assert abs(result["cx"] - reference_k[0, 2]) / reference_k[0, 2] < 0.005
    assert abs(result["cy"] - reference_k[1, 2]) / reference_k[1, 2] < 0.005
    assert result["rms_reprojection_error"] >= 0.0


def test_same_seed_is_bit_identical():
    first, first_images, _ = generate_synthetic_calibration(
        320, 240, views=10, seed=7
    )
    second, second_images, _ = generate_synthetic_calibration(
        320, 240, views=10, seed=7
    )
    third, third_images, _ = generate_synthetic_calibration(
        320, 240, views=10, seed=8
    )

    assert first["fx"] == second["fx"]
    assert np.array_equal(first_images[0], second_images[0])
    assert not np.array_equal(first_images[0], third_images[0])


def test_rejects_invalid_synthetic_parameters():
    with pytest.raises(ValueError, match="at least 10"):
        generate_synthetic_calibration(640, 480, views=9)
    with pytest.raises(ValueError, match="hfov_deg"):
        generate_synthetic_calibration(640, 480, hfov_deg=5.0)
    with pytest.raises(ValueError, match="square_size_m"):
        generate_synthetic_calibration(640, 480, square_size_m=0.0)
    with pytest.raises(ValueError, match="noise_px"):
        generate_synthetic_calibration(640, 480, noise_px=-0.1)


def test_cli_writes_grid_images_and_tagged_json(tmp_path: Path):
    out_dir = tmp_path / "synthetic"
    code = main(
        [
            "--width",
            "320",
            "--height",
            "240",
            "--views",
            "10",
            "--out",
            str(out_dir),
        ]
    )

    assert code == 0
    payload = json.loads(
        (out_dir / "camera_intrinsics.json").read_text(encoding="utf-8")
    )
    assert payload["provenance"] == "synthetic"
    assert payload["synthetic"] is True
    assert payload["assumed_hfov_deg"] == pytest.approx(70.0)
    assert len(payload["images_used"]) == 10
    assert len(list((out_dir / "images").glob("checkerboard_*.png"))) == 10
