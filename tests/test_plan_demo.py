"""Tests M12 and M13: CLI scripts, ablation invariants, speed and determinism."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from mapping.grid import CELL_FREE, CELL_OCCUPIED, MapConfig
from planning.astar import astar_plan
from planning.costmap import build_costmap
from planning.targets import find_explore_target
from scripts.run_planner_eval import run_synthetic_ablation

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_m12_plan_demo_missing_weights_exits_two():
    """M12: plan_demo without --fake exits with code 2 and a fetch hint when weights are missing."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan_demo.py"),
        "nonexistent_image.jpg",
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
