"""Tests M12 and M13: CLI scripts, ablation invariants, speed and determinism."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from calibration.conformal_depth import ConformalDepth
from calibration.depth_interval_config import RuntimeDepthInterval
from mapping.grid import CELL_FREE, CELL_OCCUPIED, MapConfig
from planning.astar import astar_plan
from planning.costmap import build_costmap
from planning.targets import find_explore_target
from scripts.plan_demo import (
    DEFAULT_FLOOR_THRESHOLD,
    _certified_mask_for_free_space,
    load_intrinsics_json,
)
from scripts.run_planner_eval import run_synthetic_ablation

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_fusion_report(path: Path, *, mode: str) -> Path:
    report = {
        "schema_version": 1,
        "mode": mode,
        "grid": {
            "min_accept_iou": [0.3, 0.6, 0.9],
            "veto_prob": [0.4, 0.8],
        },
        "selection": {
            "variant": "fusion",
            "min_accept_iou": 0.6,
            "veto_prob": 0.8,
            "reason": "calibration_safety_first_winner",
            "eligible": True,
            "improved": True,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _write_ppm(path: Path, width: int, height: int) -> Path:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    raster = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
    path.write_bytes(header + raster)
    return path


def _write_intrinsics(path: Path, width: int, height: int) -> Path:
    payload = {
        "fx": 0.9 * width,
        "fy": 0.9 * width,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "image_size": [width, height],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_depth_interval(
    path: Path,
    *,
    mode: str = "fake",
    model: ConformalDepth | None = None,
) -> Path:
    if model is None:
        predictions = np.arange(2.0, 12.0)
        model = ConformalDepth(alpha=0.10).fit(predictions, predictions * 0.95)
    artifact = RuntimeDepthInterval(
        model=model,
        mode=mode,
        source="tests/test_plan_demo.py",
        generated="2026-01-01T00:00:00",
    )
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return path


def _make_weight_placeholders(root: Path) -> Path:
    """Create the files the preflight checks require; contents are never read."""
    for relative in (
        "segformer-b0-ade/config.json",
        "depth-anything-v2-metric-indoor-small/config.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    (root / "yolo11n.pt").write_bytes(b"")
    return root


def test_m12_plan_demo_fake_exits_zero_and_writes_outputs(tmp_path):
    """M12: plan_demo --fake exits 0 and writes outputs."""
    out_dir = tmp_path / "demo_output"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        "--fake",
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"plan_demo --fake failed: {proc.stderr}"

    png_file = out_dir / "bev_path.png"
    json_file = out_dir / "guidance_summary.json"
    assert png_file.exists(), "bev_path.png was not written"
    assert json_file.exists(), "guidance_summary.json was not written"

    # Validate json content
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "decision" in data
    assert data["decision"] in ("GUIDE", "CAUTION", "STOP_UNCLEAR", "STOP_BLOCKED")
    nearest = data["nearest_obstacle"]
    assert nearest["label"] == "obstacle", nearest
    assert np.isfinite(nearest["distance_nominal_m"])
    # Without a fitted interval, no conformal bound may be fabricated.
    assert "distance_lower_m" not in nearest
    assert "distance_upper_m" not in nearest


def test_m12_plan_demo_missing_weights_exits_two(tmp_path):
    """M12: plan_demo without --fake exits with code 2 and a fetch hint when weights are missing."""
    empty_weights = tmp_path / "empty_weights"
    empty_weights.mkdir()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        "nonexistent_image.jpg",
        "--weights-dir",
        str(empty_weights),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 2, f"Expected exit code 2, got {proc.returncode}"
    assert "fetch_models.py" in proc.stderr or "fetch_models.py" in proc.stdout


def test_m12_ablation_fewer_than_twenty_scenes_exits_nonzero():
    """M12: scripts/run_planner_eval.py with fewer than 20 scenes exits nonzero."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_planner_eval.py"),
        "--scenes",
        "10",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode != 0


def test_m12_ablation_safety_invariants(tmp_path):
    """M12: Ablation runs in < 60s; variants D and C each have unsafe rate <= A on corrupted scenes."""
    t0 = time.perf_counter()
    res = run_synthetic_ablation(n_scenes=30, base_seed=42, out_dir=str(tmp_path / "eval"))
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"Ablation took too long: {elapsed:.2f} s"

    v_a = res["variants"]["A_naive"]
    v_c = res["variants"]["C_geometric_veto"]
    v_d = res["variants"]["D_conformal_margin"]

    rate_a = v_a["unsafe_rate_corrupted_scenes"]
    rate_c = v_c["unsafe_rate_corrupted_scenes"]
    rate_d = v_d["unsafe_rate_corrupted_scenes"]

    assert rate_c <= rate_a, f"Variant C unsafe rate {rate_c:.1%} > Variant A {rate_a:.1%}"
    assert rate_d <= rate_a, f"Variant D unsafe rate {rate_d:.1%} > Variant A {rate_a:.1%}"


def test_m13_speed_under_fifty_ms_on_large_grid():
    """M13: Median planner latency under 50 ms on a 70 x 75 grid."""
    cfg = MapConfig(res=0.10, x_min=-3.5, x_max=3.5, y_min=-0.5, y_max=7.0)
    # Shape is 75 rows x 70 cols
    assert cfg.n_cols == 70
    assert cfg.n_rows == 75

    grid = np.full((75, 70), CELL_FREE, dtype=np.uint8)
    # Add sparse random obstacle blocks
    rng = np.random.default_rng(999)
    for _ in range(8):
        ox = rng.integers(10, 60)
        oy = rng.integers(10, 65)
        grid[oy : oy + 4, ox : ox + 4] = CELL_OCCUPIED

    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.35, safety_margin_m=0.10)
    start = (35, 10)
    goal = (35, 65)

    latencies_ms: list[float] = []
    for _ in range(15):
        t0 = time.perf_counter()
        res = astar_plan(costmap, start, goal)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

    median_lat = float(np.median(latencies_ms))
    assert res.status == "OK"
    assert median_lat < 50.0, f"Median latency {median_lat:.2f} ms exceeded 50 ms limit"


def test_m13_repeat_runs_are_identical():
    """M13: Identical results across repeat runs with the same seed."""
    cfg = MapConfig(res=0.10)
    grid = np.full((cfg.n_rows, cfg.n_cols), CELL_FREE, dtype=np.uint8)
    grid[20:25, 25:35] = CELL_OCCUPIED

    costmap = build_costmap(grid, cfg=cfg)
    start = (30, 5)
    goal = (30, 45)

    p1 = astar_plan(costmap, start, goal)
    p2 = astar_plan(costmap, start, goal)

    assert p1.status == p2.status
    assert p1.cost == p2.cost
    assert p1.path == p2.path
    assert p1.length_m == p2.length_m


def test_m12_plan_demo_fake_with_tuned_fusion_reports_certification(tmp_path):
    """M12: --fake honours a validated fake tuning report and writes provenance."""
    out_dir = tmp_path / "fusion_demo"
    report = _write_fusion_report(tmp_path / "fake_tuning.json", mode="fake")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        "--fake",
        "--fusion-config",
        str(report),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"plan_demo fake fusion failed: {proc.stderr}"

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "fake"
    assert summary["fusion_config"]["enabled"] is True
    assert summary["fusion"] is not None
    assert summary["fusion"]["accepted"] is True, summary["fusion"]
    assert summary["certification"]["available"] is True
    assert summary["certification"]["floor_threshold"] == DEFAULT_FLOOR_THRESHOLD
    assert summary["certification"]["certified_pixels"] > 0
    assert summary["decision"] in ("GUIDE", "CAUTION", "STOP_UNCLEAR", "STOP_BLOCKED")


def test_m12_plan_demo_real_mode_refuses_fake_fusion_report(tmp_path):
    """M12: synthetic tuning reports must not configure a real-image run."""
    report = _write_fusion_report(tmp_path / "fake_tuning.json", mode="fake")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        str(tmp_path / "frame.ppm"),
        "--fusion-config",
        str(report),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "synthetic fake scenes" in combined


def test_m12_plan_demo_warns_on_synthetic_intrinsics(tmp_path):
    """Synthetic intrinsics load but must never pass silently as calibrated."""
    intrinsics = _write_intrinsics(tmp_path / "camera.json", 64, 48)
    payload = json.loads(intrinsics.read_text(encoding="utf-8"))
    payload["provenance"] = "synthetic"
    intrinsics.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.warns(UserWarning, match="SYNTHETIC"):
        K = load_intrinsics_json(intrinsics)

    assert (K.width, K.height) == (64, 48)


def test_m12_plan_demo_real_mode_requires_calibrated_intrinsics(tmp_path):
    """M12: real-image mode never falls back to a guessed field of view."""
    weights = _make_weight_placeholders(tmp_path / "weights")
    image = _write_ppm(tmp_path / "frame.ppm", 8, 6)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        str(image),
        "--weights-dir",
        str(weights),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "requires calibrated intrinsics" in combined
    assert "calibrate_camera.py" in combined


def test_m12_plan_demo_real_mode_rejects_intrinsics_resolution_mismatch(tmp_path):
    """M12: intrinsics must describe the image actually being processed."""
    weights = _make_weight_placeholders(tmp_path / "weights")
    image = _write_ppm(tmp_path / "frame.ppm", 8, 6)
    intrinsics = _write_intrinsics(tmp_path / "camera.json", 16, 12)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        str(image),
        "--weights-dir",
        str(weights),
        "--intrinsics",
        str(intrinsics),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "but intrinsics describe 16x12" in combined


def test_m12_plan_demo_real_mode_requires_mobile_sam_for_tuned_fusion(tmp_path):
    """M12: enabled real fusion preflights MobileSAM before any model runs."""
    weights = _make_weight_placeholders(tmp_path / "weights")
    image = _write_ppm(tmp_path / "frame.ppm", 8, 6)
    intrinsics = _write_intrinsics(tmp_path / "camera.json", 8, 6)
    report = _write_fusion_report(tmp_path / "real_tuning.json", mode="real")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        str(image),
        "--weights-dir",
        str(weights),
        "--intrinsics",
        str(intrinsics),
        "--fusion-config",
        str(report),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "MobileSAM is not ready" in combined
    assert "fetch_models.py" in combined


def test_m12_plan_demo_rejects_invalid_floor_threshold(tmp_path):
    """M12: a non-probability certification threshold is rejected up front."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        "--fake",
        "--floor-threshold",
        "1.5",
        "--out-dir",
        str(tmp_path / "out"),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 2
    assert "finite number in [0, 1]" in (proc.stdout + proc.stderr)


def test_m12_plan_demo_fake_with_conformal_depth_activates_pessimistic_layer(tmp_path):
    """M12: a validated depth interval activates the lower-bound obstacle layer."""
    out_dir = tmp_path / "interval_demo"
    artifact = _write_depth_interval(tmp_path / "depth_interval.json", mode="fake")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        "--fake",
        "--conformal-depth",
        str(artifact),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "Depth interval:     mode=fake" in combined
    assert "no fitted conformal depth interval was supplied" not in combined

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["depth_interval"]["mode"] == "fake"
    assert summary["depth_interval"]["source"] == "tests/test_plan_demo.py"
    assert summary["depth_interval"]["pooled_count"] == 10
    assert summary["depth_interval"]["alpha"] == pytest.approx(0.10)
    assert summary["certification"]["pessimistic_layer"] is True
    nearest = summary["nearest_obstacle"]
    assert nearest["label"] == "obstacle", nearest
    assert nearest["distance_lower_m"] < nearest["distance_nominal_m"]
    assert nearest["distance_nominal_m"] < nearest["distance_upper_m"]
    assert summary["decision"] in ("GUIDE", "CAUTION", "STOP_UNCLEAR", "STOP_BLOCKED")


def test_m12_plan_demo_wide_interval_abstains(tmp_path):
    """M12: a calibrated but wide obstacle interval must stop, not guide."""
    out_dir = tmp_path / "wide_interval_demo"
    predictions = np.arange(2.0, 12.0)
    wide = ConformalDepth(alpha=0.10).fit(predictions, predictions * 0.1)
    artifact = _write_depth_interval(
        tmp_path / "depth_interval.json", mode="fake", model=wide
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        "--fake",
        "--conformal-depth",
        str(artifact),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "STOP_UNCLEAR"
    guidance = json.loads((out_dir / "guidance_summary.json").read_text(encoding="utf-8"))
    assert any("relative interval width" in reason for reason in guidance["reasons"])


def test_m12_plan_demo_real_mode_refuses_fake_depth_interval(tmp_path):
    """M12: synthetic depth quantiles must not configure real obstacle bounds."""
    artifact = _write_depth_interval(tmp_path / "depth_interval.json", mode="fake")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        str(tmp_path / "frame.ppm"),
        "--conformal-depth",
        str(artifact),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "synthetic fake data" in combined


def test_m12_plan_demo_certification_fails_closed_without_probabilities():
    """M12: floor labels are never promoted to certified free space."""
    labels = np.ones((4, 5), dtype=np.uint8)
    mask, available = _certified_mask_for_free_space(labels, None)
    assert available is False
    assert mask.shape == labels.shape
    assert not mask.any()

    certified = np.zeros(labels.shape, dtype=bool)
    certified[1, 2] = True
    mask, available = _certified_mask_for_free_space(labels, certified)
    assert available is True
    assert int(mask.sum()) == 1
    assert mask[1, 2]

    mismatched, available = _certified_mask_for_free_space(
        labels, np.ones((2, 2), dtype=bool)
    )
    assert available is False
    assert not mismatched.any()
