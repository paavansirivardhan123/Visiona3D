"""F14-F20: safety-first fusion tuning workflow.

No test loads MobileSAM, torch, transformers, a camera, a video or the network.
Fake mode renders analytic scenes; real-mode tests stop at the weight check.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tune_fusion import (  # noqa: E402
    _eligibility,
    select_candidate,
    split_frames,
)

SCRIPT = REPO_ROOT / "scripts" / "tune_fusion.py"


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


def _make_images(folder: Path, count: int = 12) -> list[Path]:
    folder.mkdir()
    paths = []
    for index in range(count):
        path = folder / f"frame_{index:02d}.ppm"
        _write_ppm(path, 64, 48, index)
        paths.append(path)
    return paths


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_labels(folder: Path, names: list[str], height: int = 48, width: int = 64) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        labels = np.zeros((height, width), dtype=np.uint8)
        labels[height // 2 :, :] = 1
        labels[: height // 4, :] = 4
        np.save(folder / f"{name}.npy", labels)


def _row(
    iou: float,
    veto: float,
    *,
    unsafe: float,
    wall: float,
    floor: float,
    recall: float,
    accepted_rate: float,
    eligible: bool = True,
) -> dict:
    return {
        "min_accept_iou": iou,
        "veto_prob": veto,
        "floor_iou": floor,
        "wall_iou": 0.5,
        "mean_iou": 0.5,
        "unsafe_free_rate": unsafe,
        "wall_as_floor_rate": wall,
        "floor_recall": recall,
        "floor_fraction": 0.4,
        "n_frames": 6,
        "n_accepted": int(round(accepted_rate * 6)),
        "accepted_rate": accepted_rate,
        "eligible": eligible,
    }


def test_t1_fake_tuning_writes_report_csv_and_improves_heldout():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        out = root / "out"
        _make_images(images, 12)

        proc = _run(
            "--fake",
            "--images",
            str(images),
            "--out",
            str(out),
            "--seed",
            "3",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report = _read_json(out / "fusion_tuning.json")
        assert report["mode"] == "fake"
        assert report["ground_truth"] == "synthetic"
        assert report["n_frames"] == 12
        assert report["schema_version"] == 1
        assert any("TO TUNE ON REAL DATA" in note for note in report["notes"])
        assert report["policy"]["selection_basis"].startswith("calibration frames only")

        split = report["split"]
        calibration = split["calibration_frames"]
        heldout = split["heldout_frames"]
        assert len(calibration) + len(heldout) == 12
        assert set(calibration).isdisjoint(heldout)
        assert set(calibration) | set(heldout) == {f"frame_{i:02d}" for i in range(12)}
        assert split["strategy"] == "frame"

        candidates = report["calibration"]["candidates"]
        assert len(candidates) == 6 * 7
        assert any(row["is_default"] for row in candidates)
        assert report["calibration"]["n_eligible"] > 0
        baseline = report["calibration"]["segformer"]
        selection = report["selection"]
        assert selection["variant"] == "fusion"
        assert selection["min_accept_iou"] in report["grid"]["min_accept_iou"]
        assert selection["veto_prob"] in report["grid"]["veto_prob"]
        assert selection["improved"] is True
        selected_calibration = selection["calibration_metrics"]
        assert selected_calibration["floor_iou"] >= baseline["floor_iou"] - 1e-9
        assert (
            selected_calibration["unsafe_free_rate"]
            <= baseline["unsafe_free_rate"] + 1e-9
        )
        assert (
            selected_calibration["wall_as_floor_rate"]
            <= baseline["wall_as_floor_rate"] + 1e-9
        )

        held = report["heldout"]
        assert held["n_frames"] == 6
        assert set(held["variants"]) == {
            "segformer",
            "fusion_default",
            "fusion_selected",
        }
        selected_held = held["variants"]["fusion_selected"]
        baseline_held = held["variants"]["segformer"]
        assert selected_held["accepted_rate"] > 0.0
        assert (
            selected_held["unsafe_free_rate"]
            <= baseline_held["unsafe_free_rate"] + 1e-9
        )
        assert selected_held["floor_iou"] >= baseline_held["floor_iou"] - 1e-9
        for variant in held["variants"].values():
            for field in ("unsafe_free_rate", "wall_as_floor_rate", "floor_iou"):
                ci = variant["ci95"][field]
                assert ci["n"] >= 1
                assert ci["ci95"][0] <= ci["ci95"][1]

        assert len(report["frames"]) == 12
        assert {frame["split"] for frame in report["frames"]} == {
            "calibration",
            "heldout",
        }

        with (out / "fusion_tuning.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 12 * 3
        assert {row["variant"] for row in rows} == {
            "segformer",
            "fusion_default",
            "fusion_selected",
        }
        assert {row["split"] for row in rows} == {"calibration", "heldout"}
        selected_rows = [row for row in rows if row["variant"] == "fusion_selected"]
        assert selected_rows
        for row in selected_rows:
            assert float(row["min_accept_iou"]) == pytest.approx(
                selection["min_accept_iou"]
            )
            assert float(row["veto_prob"]) == pytest.approx(selection["veto_prob"])
        segformer_rows = [row for row in rows if row["variant"] == "segformer"]
        assert all(row["min_accept_iou"] == "" for row in segformer_rows)
        assert all(row["accepted"] == "false" for row in segformer_rows)


def test_t2_split_frames_is_deterministic_disjoint_and_route_safe():
    names = [f"f{index:02d}" for index in range(10)]
    first = split_frames(names, calibration_fraction=0.5, seed=7)
    second = split_frames(names, calibration_fraction=0.5, seed=7)
    assert first == second
    assert len(first.calibration) == 5
    assert len(first.heldout) == 5
    assert set(first.calibration).isdisjoint(first.heldout)
    assert set(first.calibration) | set(first.heldout) == set(names)

    route_map = {
        name: f"route_{index // 3}" for index, name in enumerate(names)
    }
    routed = split_frames(
        names, calibration_fraction=0.5, seed=5, route_map=route_map
    )
    assert routed.strategy == "route"
    assert set(routed.calibration_routes).isdisjoint(routed.heldout_routes)
    assert set(routed.calibration_routes) | set(routed.heldout_routes) == set(
        route_map.values()
    )
    for name in routed.calibration:
        assert name in route_map
        assert route_map[name] in routed.calibration_routes
    for name in routed.heldout:
        assert route_map[name] in routed.heldout_routes
    assert routed == split_frames(
        names, calibration_fraction=0.5, seed=5, route_map=route_map
    )

    with pytest.raises(ValueError):
        split_frames(["only_one"], calibration_fraction=0.5, seed=0)
    with pytest.raises(ValueError):
        split_frames(names, calibration_fraction=0.0, seed=0)
    with pytest.raises(ValueError):
        split_frames(names, calibration_fraction=1.0, seed=0)
    with pytest.raises(ValueError):
        split_frames(names, calibration_fraction=0.5, seed=0, route_map={"f00": "r0"})
    with pytest.raises(ValueError):
        split_frames(
            names,
            calibration_fraction=0.5,
            seed=0,
            route_map={name: "single" for name in names},
        )


def test_t3_selection_is_safety_first_and_uses_calibration_rows_only():
    baseline = {
        "floor_iou": 0.80,
        "unsafe_free_rate": 0.10,
        "wall_as_floor_rate": 0.10,
        "floor_recall": 0.95,
        "accepted_rate": 0.0,
        "n_frames": 6,
    }
    safe_but_lower_floor = _row(
        0.7,
        0.8,
        unsafe=0.00,
        wall=0.00,
        floor=0.80,
        recall=0.95,
        accepted_rate=0.5,
    )
    better_floor_but_less_safe = _row(
        0.5,
        0.6,
        unsafe=0.01,
        wall=0.00,
        floor=0.95,
        recall=1.0,
        accepted_rate=1.0,
    )
    selection = select_candidate(
        [better_floor_but_less_safe, safe_but_lower_floor], baseline
    )
    assert selection["variant"] == "fusion"
    assert selection["min_accept_iou"] == pytest.approx(0.7)
    assert selection["veto_prob"] == pytest.approx(0.8)

    conservative = _row(
        0.7,
        0.8,
        unsafe=0.00,
        wall=0.00,
        floor=0.90,
        recall=1.0,
        accepted_rate=0.5,
    )
    permissive = _row(
        0.5,
        0.6,
        unsafe=0.00,
        wall=0.00,
        floor=0.90,
        recall=1.0,
        accepted_rate=1.0,
    )
    tie = select_candidate([permissive, conservative], baseline)
    assert tie["min_accept_iou"] == pytest.approx(0.7)
    assert tie["veto_prob"] == pytest.approx(0.8)

    no_eligible = select_candidate(
        [{**safe_but_lower_floor, "eligible": False}], baseline
    )
    assert no_eligible["variant"] == "segformer_only"
    assert no_eligible["reason"] == (
        "no_grid_config_matched_segformer_floor_iou_and_safety"
    )

    no_improvement = _row(
        0.5,
        0.6,
        unsafe=baseline["unsafe_free_rate"],
        wall=baseline["wall_as_floor_rate"],
        floor=baseline["floor_iou"],
        recall=baseline["floor_recall"],
        accepted_rate=1.0,
    )
    no_gain = select_candidate([no_improvement], baseline)
    assert no_gain["variant"] == "segformer_only"
    assert no_gain["reason"] == (
        "best_eligible_config_did_not_improve_on_calibration"
    )
    assert no_gain["best_candidate"]["min_accept_iou"] == pytest.approx(0.5)

    eligible, reasons = _eligibility(safe_but_lower_floor, baseline)
    assert eligible is True
    assert reasons == []
    below_floor = {**safe_but_lower_floor, "floor_iou": 0.79}
    eligible, reasons = _eligibility(below_floor, baseline)
    assert eligible is False
    assert "floor_iou_below_segformer" in reasons
    unsafe_row = {**safe_but_lower_floor, "unsafe_free_rate": 0.2}
    eligible, reasons = _eligibility(unsafe_row, baseline)
    assert eligible is False
    assert "unsafe_free_rate_above_segformer" in reasons
    wall_row = {**safe_but_lower_floor, "wall_as_floor_rate": 0.2}
    eligible, reasons = _eligibility(wall_row, baseline)
    assert eligible is False
    assert "wall_as_floor_rate_above_segformer" in reasons


def test_t4_heldout_frames_cannot_change_the_selected_configuration():
    """A held-out-only data change must leave the calibration selection fixed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        _make_images(images, 12)
        route_map = {
            f"frame_{index:02d}": f"route_{index // 3}" for index in range(12)
        }
        routes_path = root / "routes.json"
        routes_path.write_text(json.dumps(route_map), encoding="utf-8")

        first_out = root / "first"
        proc = _run(
            "--fake",
            "--images",
            str(images),
            "--route-map",
            str(routes_path),
            "--out",
            str(first_out),
            "--seed",
            "5",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        first = _read_json(first_out / "fusion_tuning.json")
        assert first["split"]["strategy"] == "route"
        calibration = set(first["split"]["calibration_frames"])
        heldout = set(first["split"]["heldout_frames"])
        assert calibration.isdisjoint(heldout)
        assert first["selection"]["variant"] == "fusion"
        assert (
            first["heldout"]["variants"]["fusion_selected"]
            != first["heldout"]["variants"]["fusion_default"]
        ), "test fixture should make the held-out winner differ from the default"

        altered = root / "altered"
        altered.mkdir()
        altered_routes = {}
        for path in images.iterdir():
            if path.stem in calibration:
                new_stem = path.stem
            else:
                new_stem = path.stem + "x"
            shutil.copyfile(path, altered / f"{new_stem}{path.suffix}")
            altered_routes[new_stem] = route_map[path.stem]
        altered_routes_path = root / "altered_routes.json"
        altered_routes_path.write_text(
            json.dumps(altered_routes), encoding="utf-8"
        )

        second_out = root / "second"
        proc = _run(
            "--fake",
            "--images",
            str(altered),
            "--route-map",
            str(altered_routes_path),
            "--out",
            str(second_out),
            "--seed",
            "5",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        second = _read_json(second_out / "fusion_tuning.json")
        assert second["split"]["calibration_frames"] == first["split"][
            "calibration_frames"
        ]
        assert second["split"]["heldout_frames"] != first["split"]["heldout_frames"]
        assert second["selection"] == first["selection"]
        assert (
            second["heldout"]["variants"]["segformer"]
            != first["heldout"]["variants"]["segformer"]
        ), "altered held-out scenes should change the held-out metrics"


def test_t5_real_mode_missing_weights_exits_2_with_fetch_instructions():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        labels = root / "labels"
        out = root / "out"
        empty_weights = root / "weights"
        paths = _make_images(images, 6)
        _write_labels(labels, [path.stem for path in paths])
        empty_weights.mkdir()

        proc = _run(
            "--images",
            str(images),
            "--labels-dir",
            str(labels),
            "--out",
            str(out),
            "--weights-dir",
            str(empty_weights),
        )
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "fetch_models.py" in proc.stdout
        assert "segformer-b0" in proc.stdout
        assert "mobile_sam.pt" in proc.stdout
        assert not (out / "fusion_tuning.json").exists()


def test_t6_real_mode_requires_a_label_for_every_frame():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        out = root / "out"
        paths = _make_images(images, 6)

        proc = _run("--images", str(images), "--out", str(out))
        assert proc.returncode == 2
        assert "requires --labels-dir" in proc.stdout
        assert not (out / "fusion_tuning.json").exists()

        labels = root / "labels"
        _write_labels(labels, [path.stem for path in paths[:-1]])
        proc = _run(
            "--images", str(images), "--labels-dir", str(labels), "--out", str(out)
        )
        assert proc.returncode == 2
        assert "no ground-truth mask" in proc.stdout
        assert paths[-1].stem in proc.stdout
        assert not (out / "fusion_tuning.json").exists()


def test_t8_clean_fake_scenes_select_segformer_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        out = root / "out"
        _make_images(images, 8)

        proc = _run(
            "--fake",
            "--fake-wall-as-floor",
            "0",
            "--images",
            str(images),
            "--out",
            str(out),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report = _read_json(out / "fusion_tuning.json")
        selection = report["selection"]
        assert selection["variant"] == "segformer_only"
        assert selection["reason"] == (
            "best_eligible_config_did_not_improve_on_calibration"
        )
        assert selection["best_candidate"] is not None
        assert selection["best_candidate"]["accepted_rate"] is not None
        assert report["heldout"]["variants"]["fusion_selected"]["accepted_rate"] == 0.0
        assert report["heldout"]["variants"]["fusion_selected"]["floor_iou"] == (
            pytest.approx(report["heldout"]["variants"]["segformer"]["floor_iou"])
        )


def test_t7_cli_rejects_too_few_frames_invalid_grids_and_bad_route_maps():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / "images"
        _make_images(images, 6)
        out = root / "out"

        proc = _run(
            "--fake", "--images", str(images), "--out", str(out), "--min-frames", "8"
        )
        assert proc.returncode == 2
        assert "need at least 8 labelled frames" in proc.stdout

        proc = _run(
            "--fake", "--images", str(images), "--out", str(out), "--iou-grid", "1.2"
        )
        assert proc.returncode == 2
        assert "--iou-grid" in proc.stdout

        proc = _run(
            "--fake",
            "--images",
            str(images),
            "--out",
            str(out),
            "--calibration-fraction",
            "1.0",
        )
        assert proc.returncode == 2
        assert "calibration-fraction" in proc.stdout

        routes_path = root / "routes.json"
        routes_path.write_text(
            json.dumps({path.stem: "one" for path in images.iterdir()}),
            encoding="utf-8",
        )
        proc = _run(
            "--fake",
            "--images",
            str(images),
            "--route-map",
            str(routes_path),
            "--out",
            str(out),
        )
        assert proc.returncode == 2
        assert "at least two distinct routes" in proc.stdout

        partial = {
            path.stem: f"route_{index % 2}"
            for index, path in enumerate(sorted(images.iterdir())[:-1])
        }
        routes_path.write_text(json.dumps(partial), encoding="utf-8")
        proc = _run(
            "--fake",
            "--images",
            str(images),
            "--route-map",
            str(routes_path),
            "--out",
            str(out),
        )
        assert proc.returncode == 2
        assert "route map is missing" in proc.stdout
        assert not (out / "fusion_tuning.json").exists()
