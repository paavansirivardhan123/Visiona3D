"""P7, P8, P12, P13: pipeline behaviour, logging, smoke script and determinism."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.config import Intrinsics, MountConfig
from evaluation.logging_jsonl import JSONLLogger, read_jsonl, validate_jsonl_schema
from evaluation.metrics_depth import compute_depth_metrics
from geometry.ground_plane import FitStatus
from geometry.scale import ScaleEstimator
from perception.fakes import FakeDepthBackend, FakeDetector, FakeSegBackend
from perception.interfaces import FrameOutputs
from perception.pipeline import PerceptionPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
H, W = 240, 320
K = Intrinsics.from_fov(width=W, height=H, hfov_deg=75.0)
MOUNT = MountConfig(h_known=1.4, pitch_deg=15.0)
WALL_M = 4.0
BOXES = [(-0.2, 0.2, 2.4, 2.7, 0.4, 0.8)]
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)
SCENE = dict(K=K, H=H, W=W, h=1.4, pitch_deg=15.0, wall_distance=WALL_M, boxes=BOXES, seed=3)


def _pipeline(seg=None, depth=None, detector=None, seed: int = 0) -> PerceptionPipeline:
    seg = seg if seg is not None else FakeSegBackend(**SCENE)
    depth = depth if depth is not None else FakeDepthBackend(**SCENE)
    detector = detector if detector is not None else FakeDetector(**SCENE)
    return PerceptionPipeline(
        seg=seg,
        depth=depth,
        detector=detector,
        K=K,
        mount=MOUNT,
        estimator=ScaleEstimator(MOUNT),
        rng=np.random.default_rng(seed),
    )


def test_p7_perfect_fakes_wall_distance_within_two_percent():
    output = _pipeline().run(IMAGE)
    assert output.validate() == []
    assert output.statuses["seg"] == "OK"
    assert output.statuses["depth"] == "OK"
    assert output.statuses["detector"] == "OK"
    assert output.statuses["lift"] == "OK"
    wall_y = float(np.median(output.lift.points_g["wall"][:, 1]))
    error = abs(wall_y - WALL_M) / WALL_M
    assert error < 0.02, f"wall distance error {error:.4%}"


@pytest.mark.parametrize("k", [0.6, 1.7])
def test_p7_metric_scale_corrects_first_frame(k: float):
    depth_backend = FakeDepthBackend(**SCENE, k=k)
    output = _pipeline(depth=depth_backend).run(IMAGE)
    assert output.statuses["lift"] == "OK"

    truth = depth_backend.depth_true
    pred = output.lift.scaled_depth
    valid = depth_backend.wall_mask & np.isfinite(pred)
    metrics = compute_depth_metrics(truth, pred, valid_mask=valid)
    print(
        f"P7 k={k}: wall AbsRel={metrics['AbsRel']:.6f}, "
        f"scale s={output.lift.scale_result.s:.6f} (true {1.0 / k:.6f})"
    )
    assert metrics["AbsRel"] < 0.02, metrics["AbsRel"]


def test_p7_corrupt_floor_labels_do_not_raise_and_report_status():
    seg_backend = FakeSegBackend(**SCENE, floor_as_wall_fraction=0.30)
    output = _pipeline(seg=seg_backend).run(IMAGE)

    assert output.statuses["lift"] in {status.value for status in FitStatus}
    plane = output.lift.plane_fit
    h_error = abs(plane.h_hat - 1.4) / 1.4
    print(
        f"P7 corrupt labels: lift_status={output.statuses['lift']}, "
        f"plane_h_err={h_error:.4%}, inlier_ratio={plane.inlier_ratio:.4f}, "
        f"scale_status={output.statuses.get('scale')}"
    )
    assert h_error < 0.02, f"corrupt-label plane height error {h_error:.4%} exceeds 2%"
    assert plane.inlier_ratio > 0.8, f"inlier ratio {plane.inlier_ratio:.4f} is too low"
    assert output.statuses["seg"] == "OK"


@pytest.mark.parametrize("fraction", [0.0, 0.10, 0.20, 0.30])
def test_p7_wall_as_floor_sweep_never_reports_ok_with_large_height_error(fraction):
    """The unsafe wall->floor direction must be flagged or numerically harmless.

    ``wall_as_floor_fraction`` relabels wall pixels as floor, so the ground-plane
    fit is fed a vertical surface. For each fraction, a non-OK lift status flags
    the problem; an OK lift is allowed only when the plane height error is <5%.
    """
    seg_backend = FakeSegBackend(**SCENE, wall_as_floor_fraction=fraction)
    output = _pipeline(seg=seg_backend).run(IMAGE)
    assert output.lift is not None

    plane = output.lift.plane_fit
    h_error = abs(plane.h_hat - 1.4) / 1.4
    status = output.statuses["lift"]
    print(
        f"P7 wall_as_floor={fraction:.2f}: lift_status={status}, "
        f"plane_h_err={h_error:.4%}, inlier_ratio={plane.inlier_ratio:.4f}, "
        f"h_hat={plane.h_hat:.4f}"
    )
    assert status != "OK" or h_error < 0.05, (
        f"wall_as_floor={fraction:.2f}: lift reported OK with height error "
        f"{h_error:.4%} (status={status})"
    )


def test_p7_backend_exception_becomes_status():
    class RaisingBackend:
        def predict(self, image_rgb):
            raise RuntimeError("deliberate backend failure")

    output = _pipeline(seg=RaisingBackend()).run(IMAGE)
    assert output.statuses["seg"].startswith("error:seg:RuntimeError:deliberate backend failure")
    assert output.statuses["lift"].startswith("skipped")
    assert output.seg is None


def test_p7_model_resolution_depth_is_mapped_back():
    depth_backend = FakeDepthBackend(**SCENE, output_hw=(H // 2, W // 2))
    output = _pipeline(depth=depth_backend).run(IMAGE)
    assert output.depth.depth_m.shape == (H, W)
    assert output.statuses["depth"] == "OK"
    wall_y = float(np.median(output.lift.points_g["wall"][:, 1]))
    assert abs(wall_y - WALL_M) / WALL_M < 0.02


def test_p8_frame_outputs_convert_to_valid_jsonl_record():
    output = _pipeline().run(IMAGE)
    record = output.to_log_record("frame_001", "route_9", timestamp_ns=123456, wall_time_s=10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "frames.jsonl"
        with JSONLLogger(path, mode="w") as logger:
            logger.write_dict(record)
        assert validate_jsonl_schema(path) == []
        records = read_jsonl(path)
    assert len(records) == 1
    assert records[0]["frame_id"] == "frame_001"
    for stage in ("seg", "depth", "detector", "lift", "total"):
        assert stage in records[0]["module_latencies"]
        assert records[0]["module_latencies"][stage] >= 0.0
    assert records[0]["status_codes"]["seg"] == "OK"


def test_p13_repeat_runs_are_identical():
    first = _pipeline(seed=17).run(IMAGE)
    second = _pipeline(seed=17).run(IMAGE)
    assert np.array_equal(first.seg.labels, second.seg.labels)
    assert np.array_equal(first.seg.probs, second.seg.probs)
    assert np.array_equal(first.depth.depth_m, second.depth.depth_m)
    assert np.array_equal(first.lift.scaled_depth, second.lift.scaled_depth)
    assert np.array_equal(first.lift.plane_fit.n, second.lift.plane_fit.n)
    assert first.lift.scale_result.s == second.lift.scale_result.s


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
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes())


def test_p12_smoke_script_fake_mode_and_missing_weights():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        image_dir.mkdir()
        _write_ppm(image_dir / "a.ppm", 64, 48, 0)
        _write_ppm(image_dir / "b.ppm", 64, 48, 40)
        out_dir = root / "out"
        empty_weights = root / "empty_weights"
        empty_weights.mkdir()

        fake_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "real_models_smoke.py"),
                "--fake",
                "--images",
                str(image_dir),
                "--out",
                str(out_dir),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert fake_proc.returncode == 0, fake_proc.stdout + fake_proc.stderr
        report = json.loads((out_dir / "real_models_smoke.json").read_text(encoding="utf-8"))
        assert report["fake"] is True
        assert len(report["images"]) == 2
        for entry in report["images"]:
            assert (out_dir / entry["overlay"]).exists()
            assert entry["seg_shape"] == [48, 64]
            assert entry["depth_shape"] == [48, 64]
            assert sum(entry["class_histogram"].values()) > 0
            assert "seg_ms" in entry["timings_ms"]

        missing_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "real_models_smoke.py"),
                "--images",
                str(image_dir),
                "--out",
                str(root / "out_real"),
                "--weights-dir",
                str(empty_weights),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert missing_proc.returncode == 2, missing_proc.stdout + missing_proc.stderr
        assert "fetch_models.py" in missing_proc.stdout
        assert "segformer-b0" in missing_proc.stdout
