"""Master validation script for Visiona3D Geometry Module.

Executes the full test suite via pytest on geometry/tests, evaluates
quantitative numerical metrics for T1-T10, prints a summary table,
and writes the final report to geometry/reports/geometry_report.json.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from core.config import Intrinsics, MountConfig, RansacConfig, ScaleConfig
from geometry.camera import backproject, project
from geometry.frames import camera_to_ground, ground_to_camera
from geometry.ground_plane import FitStatus, fit_ground_plane
from geometry.lift import AnchorStatus, lift_frame, object_anchor
from geometry.scale import ScaleEstimator
from tests.synthetic import add_noise, render_scene, scale_depth


def measure_t1() -> dict:
    K = Intrinsics.from_fov(width=1280, height=720, hfov_deg=80.0)
    rng = np.random.default_rng(42)
    depth = rng.uniform(0.5, 10.0, size=(720, 1280))
    pts_c = backproject(depth, K)
    pixels = project(pts_c, K)
    v_grid, u_grid = np.indices((720, 1280), dtype=np.float64)
    expected_pixels = np.stack([u_grid, v_grid], axis=-1)
    max_pixel_err = float(np.max(np.abs(pixels - expected_pixels)))
    max_depth_err = float(np.max(np.abs(pts_c[..., 2] - depth)))
    passed = max_pixel_err < 1e-6 and max_depth_err < 1e-6
    return {
        "id": "T1",
        "name": "project/backproject roundtrip",
        "measured": f"pixel_err={max_pixel_err:.2e} px, depth_err={max_depth_err:.2e} m",
        "target": "< 1e-6",
        "passed": passed,
        "details": {"max_pixel_error": max_pixel_err, "max_depth_error": max_depth_err},
    }


def measure_t2() -> dict:
    h = 1.4
    rng = np.random.default_rng(123)
    max_floor_z = 0.0
    max_wall_y = 0.0
    for pitch_deg in [0.0, 10.0, 20.0]:
        theta = math.radians(pitch_deg)
        n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
        # Floor points
        x = rng.uniform(-4.0, 4.0, size=200)
        z = rng.uniform(0.5, 15.0, size=200)
        y = (h - n[2] * z) / n[1]
        floor_c = np.stack([x, y, z], axis=-1)
        floor_g = camera_to_ground(floor_c, n, h)
        max_floor_z = max(max_floor_z, float(np.max(np.abs(floor_g[:, 2]))))
        # Wall at Y = 3
        wall_g = np.stack([rng.uniform(-3.0, 3.0, size=200), np.full(200, 3.0), rng.uniform(0.0, 2.0, size=200)], axis=-1)
        wall_c = ground_to_camera(wall_g, n, h)
        wall_rec_g = camera_to_ground(wall_c, n, h)
        max_wall_y = max(max_wall_y, float(np.max(np.abs(wall_rec_g[:, 1] - 3.0))))

    passed = max_floor_z < 1e-6 and max_wall_y < 1e-6
    return {
        "id": "T2",
        "name": "ground frame recovery",
        "measured": f"floor_z_err={max_floor_z:.2e} m, wall_y_err={max_wall_y:.2e} m",
        "target": "< 1e-6",
        "passed": passed,
        "details": {"max_floor_z_error": max_floor_z, "max_wall_y_error": max_wall_y},
    }


def measure_t3() -> dict:
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    cfg = RansacConfig(max_iterations=500, max_points=5000)
    rng = np.random.default_rng(42)
    max_h_err_pct = 0.0
    max_angle_err_deg = 0.0
    for h in [1.2, 1.6]:
        for pitch_deg in [0.0, 10.0, 20.0]:
            depth, floor_mask, _, _, gt = render_scene(K, 480, 640, h, pitch_deg)
            pts_c = backproject(depth, K)[floor_mask]
            fit = fit_ground_plane(pts_c, cfg, rng)
            h_err = abs(fit.h_hat - h) / h * 100.0
            dot_val = float(np.clip(np.dot(fit.n, gt["n"]), -1.0, 1.0))
            angle_err = math.degrees(math.acos(dot_val))
            max_h_err_pct = max(max_h_err_pct, h_err)
            max_angle_err_deg = max(max_angle_err_deg, angle_err)

    passed = max_h_err_pct < 1.0 and max_angle_err_deg < 0.5
    return {
        "id": "T3",
        "name": "clean plane recovery",
        "measured": f"h_err_max={max_h_err_pct:.3f}%, angle_err_max={max_angle_err_deg:.3f} deg",
        "target": "h < 1%, angle < 0.5 deg",
        "passed": passed,
        "details": {"max_h_error_pct": max_h_err_pct, "max_angle_error_deg": max_angle_err_deg},
    }


def measure_t4() -> dict:
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    clean_depth, floor_mask, _, _, _ = render_scene(K, 480, 640, h_true, pitch_true)
    cfg = RansacConfig(max_iterations=600, max_points=5000)
    errors_pct = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        noisy = add_noise(clean_depth, rel_sigma=0.02, outlier_frac=0.10, rng=rng)
        pts_c = backproject(noisy, K)[floor_mask]
        fit = fit_ground_plane(pts_c, cfg, rng)
        errors_pct.append(abs(fit.h_hat - h_true) / h_true * 100.0)

    mean_err_pct = float(np.mean(errors_pct))
    max_err_pct = float(np.max(errors_pct))
    passed = max_err_pct < 3.0
    return {
        "id": "T4",
        "name": "noisy recovery (20 seeds)",
        "measured": f"mean={mean_err_pct:.3f}%, max={max_err_pct:.3f}%",
        "target": "h < 3% over 20 seeds",
        "passed": passed,
        "details": {"mean_h_error_pct": mean_err_pct, "max_h_error_pct": max_err_pct},
    }


def measure_t5() -> dict:
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    wall_dist = 4.0
    depth_true, floor_mask, wall_mask, _, _ = render_scene(K, 480, 640, h_true, pitch_true, wall_dist)
    mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)
    max_wall_err_pct = 0.0
    per_k_errors = {}
    for k in [0.5, 0.8, 1.3, 2.0]:
        estimator = ScaleEstimator(mount)
        uncal = scale_depth(depth_true, k)
        for _ in range(25):
            res = lift_frame(uncal, floor_mask, {"wall": wall_mask}, K, mount, estimator)
        meas_y = float(np.median(res.points_g["wall"][:, 1]))
        err_pct = abs(meas_y - wall_dist) / wall_dist * 100.0
        per_k_errors[str(k)] = err_pct
        max_wall_err_pct = max(max_wall_err_pct, err_pct)

    passed = max_wall_err_pct < 2.0
    return {
        "id": "T5",
        "name": "scale recovery across k",
        "measured": f"max_wall_dist_err={max_wall_err_pct:.3f}% (k in [0.5, 0.8, 1.3, 2.0])",
        "target": "wall dist < 2%",
        "passed": passed,
        "details": {"max_wall_error_pct": max_wall_err_pct, "per_k_errors_pct": per_k_errors},
    }


def measure_t6() -> dict:
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    depth, floor_mask, wall_mask, _, _ = render_scene(K, 480, 640, h_true, pitch_true, 3.0)
    mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)
    estimator = ScaleEstimator(mount)
    # Tiny mask
    tiny = np.zeros_like(floor_mask)
    tiny[240:250, 320:330] = True
    res_tiny = lift_frame(depth, tiny, {}, K, mount, estimator)
    # Wall mask as floor
    res_wall = lift_frame(depth, wall_mask, {}, K, mount, estimator)
    passed = (
        res_tiny.plane_fit.status == FitStatus.TOO_FEW_POINTS
        and res_wall.plane_fit.status in (FitStatus.LOW_INLIER_RATIO, FitStatus.IMPLAUSIBLE_GEOMETRY)
    )
    return {
        "id": "T6",
        "name": "failure handling (no exceptions)",
        "measured": f"tiny={res_tiny.plane_fit.status.value}, wall={res_wall.plane_fit.status.value}",
        "target": "TOO_FEW_POINTS & LOW_INLIER/IMPLAUSIBLE",
        "passed": passed,
        "details": {"tiny_status": res_tiny.plane_fit.status.value, "wall_status": res_wall.plane_fit.status.value},
    }


def measure_t7() -> dict:
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    est = ScaleEstimator(mount)
    est.update(1.4)  # init at 1.0
    r1 = est.update(1.0)  # 40% outlier frame 1
    ignored_ok = (r1.s == 1.0 and r1.status == "OUTLIER_IGNORED")
    est.update(1.4)  # reset normal
    # 5 frames of persistent outlier
    for _ in range(4):
        est.update(1.0)
    r5 = est.update(1.0)
    persistent_ok = (abs(r5.s - 1.4) < 1e-6 and r5.status == "OK")
    passed = ignored_ok and persistent_ok
    return {
        "id": "T7",
        "name": "EMA & outlier temporal logic",
        "measured": f"outlier_ignored={ignored_ok}, frame5_accepted={persistent_ok}",
        "target": "ignore single, accept at frame 5",
        "passed": passed,
        "details": {"single_outlier_ignored": ignored_ok, "frame5_accepted": persistent_ok},
    }


def measure_t8() -> dict:
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    h_true = 1.4
    pitch_true = 15.0
    mount = MountConfig(h_known=h_true, pitch_deg=pitch_true)
    box_bounds = (-0.15, 0.15, 2.95, 3.05, 0.45, 0.55)
    depth, floor_mask, _, box_masks, _ = render_scene(K, 480, 640, h_true, pitch_true, boxes=[box_bounds])
    plane = fit_ground_plane(backproject(depth, K)[floor_mask], rng=np.random.default_rng(42))
    p_front_g = np.array([0.0, 2.95, 0.5])
    theta = math.radians(pitch_true)
    n = np.array([0.0, math.cos(theta), math.sin(theta)])
    true_front_depth = float(ground_to_camera(p_front_g, n, h_true)[2])
    anchor = object_anchor(depth, box_masks[0], K, mount, plane=plane, q=0.10)
    dist_err_pct = abs(anchor.distance_m - true_front_depth) / true_front_depth * 100.0
    pos_err_cm = float(np.linalg.norm(anchor.position_g - p_front_g)) * 100.0
    empty_status = object_anchor(depth, np.zeros_like(floor_mask), K, mount, plane=plane).status
    passed = dist_err_pct < 2.0 and pos_err_cm < 5.0 and empty_status == AnchorStatus.EMPTY_MASK
    return {
        "id": "T8",
        "name": "3D object anchor localization",
        "measured": f"dist_err={dist_err_pct:.3f}%, pos_err={pos_err_cm:.2f} cm",
        "target": "dist < 2%, pos < 5 cm",
        "passed": passed,
        "details": {"distance_error_pct": dist_err_pct, "position_error_cm": pos_err_cm, "empty_status": empty_status.value},
    }


def measure_t9() -> dict:
    import doctest
    docs_file = REPO_ROOT / "geometry" / "notation.md"
    runner = doctest.DocTestRunner(verbose=False)
    parser = doctest.DocTestParser()
    test = parser.get_doctest(docs_file.read_text(encoding="utf-8"), globs={}, name="notation.md", filename=str(docs_file), lineno=0)
    runner.run(test)
    passed = runner.failures == 0 and runner.tries >= 3
    return {
        "id": "T9",
        "name": "notation.md doctest execution",
        "measured": f"attempted={runner.tries}, failures={runner.failures}",
        "target": ">=3 attempted, 0 failures",
        "passed": passed,
        "details": {"attempted": runner.tries, "failures": runner.failures},
    }


def measure_t10() -> dict:
    K = Intrinsics.from_fov(width=640, height=480, hfov_deg=75.0)
    depth, floor_mask, wall_mask, _, _ = render_scene(K, 480, 640, 1.4, 15.0, 4.0)
    mount = MountConfig(1.4, 15.0)
    t0 = time.perf_counter()
    r1 = lift_frame(add_noise(depth, rng=np.random.default_rng(99)), floor_mask, {"w": wall_mask}, K, mount, ScaleEstimator(mount), rng=np.random.default_rng(99))
    r2 = lift_frame(add_noise(depth, rng=np.random.default_rng(99)), floor_mask, {"w": wall_mask}, K, mount, ScaleEstimator(mount), rng=np.random.default_rng(99))
    elapsed = time.perf_counter() - t0
    bit_identical = (
        np.array_equal(r1.plane_fit.n, r2.plane_fit.n)
        and r1.plane_fit.h_hat == r2.plane_fit.h_hat
        and r1.scale_result.s == r2.scale_result.s
        and np.array_equal(r1.scaled_depth, r2.scaled_depth)
        and np.array_equal(r1.points_g["floor"], r2.points_g["floor"])
    )
    passed = bit_identical and elapsed < 60.0
    return {
        "id": "T10",
        "name": "determinism & reproducibility",
        "measured": f"bit_identical={bit_identical}, runtime={elapsed:.2f} s",
        "target": "bit-identical & < 60 s",
        "passed": passed,
        "details": {"bit_identical": bit_identical, "runtime_seconds": elapsed},
    }


def main() -> int:
    print("=" * 85)
    print("VISIONA3D: GEOMETRY MODULE VERIFICATION SUITE")
    print("=" * 85)
    print(f"Timestamp:       {datetime.datetime.now().isoformat()}")
    print(f"Python Version:  {platform.python_version()}")
    print(f"NumPy Version:   {np.__version__}")
    print(f"Pytest Version:  {pytest.__version__}")
    print(f"Platform:        {platform.platform()}")
    print("-" * 85)

    # 1. Run pytest suite on geometry/tests
    print("\n[Step 1/2] Running full pytest test suite on geometry/tests...")
    pytest_cmd = [sys.executable, "-m", "pytest", "tests", "-v"]
    proc = subprocess.run(pytest_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        print("FAIL: pytest suite returned non-zero exit code!")
        return proc.returncode

    # 2. Collect numerical measurements for T1-T10
    print("[Step 2/2] Measuring quantitative requirements T1-T10...")
    testers = [
        measure_t1,
        measure_t2,
        measure_t3,
        measure_t4,
        measure_t5,
        measure_t6,
        measure_t7,
        measure_t8,
        measure_t9,
        measure_t10,
    ]

    results = [fn() for fn in testers]

    print("\n" + "=" * 95)
    print(f"{'Test':<5} | {'Name':<32} | {'Measured Value':<35} | {'Target':<12} | {'Status':<6}")
    print("-" * 95)
    all_passed = True
    for res in results:
        status_str = "PASS" if res["passed"] else "FAIL"
        print(f"{res['id']:<5} | {res['name']:<32} | {res['measured']:<35} | {res['target']:<12} | {status_str:<6}")
        if not res["passed"]:
            all_passed = False
    print("=" * 95)

    # 3. Write report JSON
    reports_dir = REPO_ROOT / "geometry" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "geometry_report.json"

    report_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "PASSED" if all_passed else "FAILED",
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytest": pytest.__version__,
            "platform": platform.platform(),
        },
        "tests": results,
    }

    report_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nWrote geometry verification report to: {report_file}")

    if all_passed:
        print("\nALL GEOMETRY MODULE CHECKS PASSED SUCCESSFULLY.")
        return 0
    else:
        print("\nFAIL: One or more checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
