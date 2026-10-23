"""F10-F11: frame-evaluation harness fake mode and missing-weights contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "eval_frames.py"


def _write_ppm(path: Path, width: int, height: int, offset: int) -> None:
    rows, cols = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            ((cols * 3 + offset) % 256),
            ((rows * 5 + offset) % 256),
            ((rows + cols + offset) % 256),
        ],
        axis=-1,
    ).astype(np.uint8)
    path.write_bytes(
        f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
    )


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def _make_images(folder: Path) -> None:
    folder.mkdir()
    _write_ppm(folder / "frame_0.ppm", 64, 48, 0)
    _write_ppm(folder / "frame_1.ppm", 80, 60, 40)


def test_f10_fake_mode_writes_json_csv_and_all_variants():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        out = root / "out"
        _make_images(images)

        proc = _run("--fake", "--images", str(images), "--out", str(out))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report = json.loads((out / "frame_eval.json").read_text(encoding="utf-8"))
        assert report["mode"] == "fake"
        assert report["ground_truth"] == "synthetic"
        assert report["n_frames"] == 2
        assert set(report["aggregate"]) == {"segformer", "sam", "fusion"}
        for variant in ("segformer", "sam", "fusion"):
            entry = report["aggregate"][variant]
            assert entry["floor_iou"] == pytest.approx(1.0)
            assert entry["unsafe_free_rate"] == pytest.approx(0.0)
            assert entry["wall_as_floor_rate"] == pytest.approx(0.0)
            assert entry["floor_recall"] == pytest.approx(1.0)
            assert entry["depth_absrel"] == pytest.approx(0.0)
            assert entry["latency_p50_ms"] >= 0.0
            assert entry["n_label_metrics"] == 2
        assert report["aggregate"]["sam"]["wall_iou"] == pytest.approx(0.0)
        assert report["aggregate"]["fusion"]["accepted_rate"] == pytest.approx(1.0)
        for frame in report["frames"]:
            assert frame["variants"]["fusion"]["accepted"] is True
            assert frame["variants"]["fusion"]["reason"] == "accepted"

        with (out / "frame_eval.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2 * 3
        assert {row["variant"] for row in rows} == {"segformer", "sam", "fusion"}
        assert all(row["floor_iou"] for row in rows)
        assert {row["accepted"] for row in rows if row["variant"] == "fusion"} == {"true"}


def test_f11_corrupted_fake_floor_is_measured_and_fusion_removes_it():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        out = root / "out"
        _make_images(images)

        proc = _run(
            "--fake",
            "--fake-wall-as-floor",
            "0.2",
            "--images",
            str(images),
            "--out",
            str(out),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report = json.loads((out / "frame_eval.json").read_text(encoding="utf-8"))
        segformer = report["aggregate"]["segformer"]
        fusion = report["aggregate"]["fusion"]

        assert segformer["wall_as_floor_rate"] == pytest.approx(0.2, abs=0.02)
        assert segformer["unsafe_free_rate"] > 0.15
        assert fusion["wall_as_floor_rate"] == pytest.approx(0.0)
        assert fusion["unsafe_free_rate"] == pytest.approx(0.0)
        assert fusion["floor_iou"] > segformer["floor_iou"]
        assert fusion["accepted_rate"] == pytest.approx(1.0)


def test_f12_real_mode_missing_weights_exits_2_with_fetch_instructions():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        out = root / "out"
        empty_weights = root / "weights"
        _make_images(images)
        empty_weights.mkdir()

        proc = _run(
            "--images",
            str(images),
            "--out",
            str(out),
            "--weights-dir",
            str(empty_weights),
        )
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "fetch_models.py" in proc.stdout
        assert "segformer-b0" in proc.stdout
        assert "mobile_sam.pt" in proc.stdout
        assert not (out / "frame_eval.json").exists()
