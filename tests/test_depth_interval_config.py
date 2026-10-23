"""D01-D07: validated runtime conformal depth-interval artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from calibration.conformal_depth import ConformalDepth
from calibration.depth_interval_config import (
    DepthIntervalConfigError,
    RuntimeDepthInterval,
    load_depth_interval,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fitted_model() -> ConformalDepth:
    predictions = np.arange(2.0, 12.0)
    ground_truth = predictions * 0.95
    return ConformalDepth(alpha=0.10).fit(predictions, ground_truth)


def _artifact(
    *,
    mode: str = "real",
    model: ConformalDepth | None = None,
    source: str = "unit-test",
    generated: str | None = "2026-01-01T00:00:00",
) -> dict:
    return RuntimeDepthInterval(
        model=_fitted_model() if model is None else model,
        mode=mode,
        source=source,
        generated=generated,
    ).to_dict()


def _write(tmp_path: Path, payload: dict, name: str = "depth_interval.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_d01_round_trip_preserves_model_and_provenance(tmp_path):
    model = _fitted_model()
    path = _write(tmp_path, _artifact(model=model, mode="real"))
    loaded = load_depth_interval(path)

    assert loaded.mode == "real"
    assert loaded.source == "unit-test"
    assert loaded.generated == "2026-01-01T00:00:00"
    assert loaded.model.alpha == pytest.approx(0.10)
    assert loaded.model.score == "two_sided"
    assert loaded.model.pooled_quantile_ == pytest.approx(model.pooled_quantile_)
    assert loaded.model.pooled_count_ == 10
    assert loaded.model.n_valid_ == 10

    probe = np.array([2.5, 4.0, 8.0])
    expected_lower = model.predict(probe).lower
    restored_lower = loaded.model.predict(probe).lower
    np.testing.assert_allclose(restored_lower, expected_lower)

    provenance = loaded.provenance()
    assert provenance["alpha"] == pytest.approx(0.10)
    assert provenance["pooled_quantile"] == pytest.approx(model.pooled_quantile_)
    assert provenance["pooled_count"] == 10


def test_d02_fake_artifact_refused_for_real_runtime(tmp_path):
    path = _write(tmp_path, _artifact(mode="fake"))
    with pytest.raises(DepthIntervalConfigError, match="synthetic fake data"):
        load_depth_interval(path)
    loaded = load_depth_interval(path, allow_fake=True)
    assert loaded.mode == "fake"


def test_d03_rejects_unbounded_interval(tmp_path):
    tiny = ConformalDepth(alpha=0.10).fit([2.0], [1.9])
    assert math.isinf(tiny.pooled_quantile_)
    path = _write(tmp_path, _artifact(model=tiny))
    with pytest.raises(DepthIntervalConfigError, match="unbounded or invalid"):
        load_depth_interval(path)


@pytest.mark.parametrize("bad_value", [-0.01, float("nan")])
def test_d04_rejects_negative_or_nan_quantile(tmp_path, bad_value):
    payload = _artifact()
    payload["model"]["pooled_quantile"] = bad_value
    path = _write(tmp_path, payload)
    with pytest.raises(DepthIntervalConfigError, match="unbounded or invalid"):
        load_depth_interval(path)


def test_d05_rejects_zero_calibration_points(tmp_path):
    payload = _artifact()
    payload["model"]["pooled_count"] = 0
    payload["model"]["n_valid"] = 0
    path = _write(tmp_path, payload)
    with pytest.raises(DepthIntervalConfigError, match="no valid calibration points"):
        load_depth_interval(path)


def test_d06_rejects_inconsistent_calibration_counts(tmp_path):
    payload = _artifact()
    payload["model"]["pooled_count"] = 4
    path = _write(tmp_path, payload)
    with pytest.raises(DepthIntervalConfigError, match="counts are inconsistent"):
        load_depth_interval(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("schema_version", 2), "schema_version"),
        (lambda p: p.__setitem__("mode", "simulation"), "mode must be"),
        (lambda p: p.__setitem__("source", ""), "source"),
        (lambda p: p.__setitem__("generated", 123), "generated"),
        (lambda p: p.pop("model"), "model must be"),
        (lambda p: p.__setitem__("model", []), "model must be"),
        (lambda p: p["model"].__setitem__("schema_version", 2), "schema_version"),
        (lambda p: p["model"].__setitem__("alpha", 1.5), "alpha"),
        (lambda p: p["model"].__setitem__("score", "absolute"), "score"),
    ],
)
def test_d07_rejects_structural_tampering(tmp_path, mutate, message):
    payload = _artifact()
    mutate(payload)
    path = _write(tmp_path, payload)
    with pytest.raises(DepthIntervalConfigError, match=message):
        load_depth_interval(path, allow_fake=True)


def test_d08_rejects_unreadable_malformed_and_non_object_files(tmp_path):
    with pytest.raises(DepthIntervalConfigError, match="cannot read"):
        load_depth_interval(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(DepthIntervalConfigError, match="not valid JSON"):
        load_depth_interval(malformed)

    non_object = tmp_path / "list.json"
    non_object.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(DepthIntervalConfigError, match="JSON object"):
        load_depth_interval(non_object)


def test_d09_fake_producer_writes_a_loadable_artifact(tmp_path):
    out = tmp_path / "calibration"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_calibration_real.py"),
        "--fake",
        "--routes",
        "6",
        "--frames-per-route",
        "4",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    report = json.loads((out / "calibration_report.json").read_text(encoding="utf-8"))
    assert report["files"]["depth_interval_json"] == str(out / "depth_interval.json")
    assert (out / "predictions.jsonl").exists()

    with pytest.raises(DepthIntervalConfigError, match="synthetic fake data"):
        load_depth_interval(out / "depth_interval.json")
    loaded = load_depth_interval(out / "depth_interval.json", allow_fake=True)
    assert loaded.mode == "fake"
    assert loaded.source == "scripts/run_calibration_real.py"
    assert loaded.generated
    assert loaded.model.pooled_count_ > 0
    assert math.isfinite(loaded.model.pooled_quantile_)
