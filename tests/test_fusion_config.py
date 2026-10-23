"""F14-F19: tuned fusion configuration validation and runtime wiring."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from core.config import MASK_FLOOR, MASK_WALL
from perception.fusion_config import (
    FusionConfigError,
    RuntimeFusionConfig,
    load_tuned_fusion_config,
    wrap_seg_with_fusion,
)
from perception.interfaces import SegOutput
from perception.seg_fusion import PromptMaskOutput
from scripts.real_models_smoke import main as smoke_main


def _valid_report(
    *,
    mode: str = "real",
    variant: str = "fusion",
    min_accept_iou: float = 0.6,
    veto_prob: float = 0.8,
) -> dict:
    reason = (
        "calibration_safety_first_winner"
        if variant == "fusion"
        else "best_eligible_config_did_not_improve_on_calibration"
    )
    return {
        "schema_version": 1,
        "mode": mode,
        "grid": {
            "min_accept_iou": [0.3, 0.6, 0.7, 0.9],
            "veto_prob": [0.4, 0.8],
        },
        "selection": {
            "variant": variant,
            "min_accept_iou": min_accept_iou if variant == "fusion" else None,
            "veto_prob": veto_prob if variant == "fusion" else None,
            "reason": reason,
            "eligible": variant == "fusion",
            "improved": variant == "fusion",
        },
    }


def _write_report(tmp_path: Path, report: dict, name: str = "tuning.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _write_ppm(path: Path, width: int = 64, height: int = 48, value: int = 0) -> None:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    raster = np.full((height, width, 3), value, dtype=np.uint8).tobytes()
    path.write_bytes(header + raster)


class _FixedSeg:
    def __init__(self, output: SegOutput) -> None:
        self.output = output

    def predict(self, image_rgb: np.ndarray) -> SegOutput:
        return self.output


class _FixedSAM:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self.calls = 0

    def predict(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> PromptMaskOutput:
        self.calls += 1
        return PromptMaskOutput(mask=self.mask.copy(), score=0.9)


def _test_base_output(shape: tuple[int, int] = (8, 8)) -> SegOutput:
    labels = np.full(shape, MASK_WALL, dtype=np.uint8)
    labels[:, :4] = MASK_FLOOR
    probs = np.zeros(shape + (4,), dtype=np.float32)
    probs[..., 1] = 1.0
    probs[:, :4, 0] = 1.0
    probs[:, :4, 1] = 0.0
    probs[:, 4, :] = (0.3, 0.3, 0.2, 0.2)
    probs[:, 5, :] = (0.1, 0.9, 0.0, 0.0)
    return SegOutput(
        labels=labels,
        probs=probs,
        sky_mask=np.zeros(shape, dtype=bool),
    )


def _load_config(
    tmp_path: Path,
    *,
    min_accept_iou: float = 0.6,
    veto_prob: float = 0.8,
    name: str = "tuning.json",
) -> RuntimeFusionConfig:
    report = _valid_report(
        min_accept_iou=min_accept_iou,
        veto_prob=veto_prob,
    )
    return load_tuned_fusion_config(_write_report(tmp_path, report, name))


def test_f14_selected_thresholds_reach_the_production_decision_core(tmp_path):
    base = _test_base_output()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, :6] = True
    sam = _FixedSAM(mask)

    config = _load_config(tmp_path, min_accept_iou=0.6, veto_prob=0.8)
    assert config.enabled
    assert config.min_accept_iou == 0.6
    assert config.veto_prob == 0.8
    assert config.selection_reason == "calibration_safety_first_winner"
    assert config.tuning_mode == "real"

    fusion = wrap_seg_with_fusion(_FixedSeg(base), sam, config)
    assert fusion.min_accept_iou == 0.6
    assert fusion.veto_prob == 0.8
    result = fusion.fuse(image, base)

    assert result.accepted
    assert result.reason == "accepted"
    assert result.prompt_count > 0
    assert sam.calls == 1
    assert result.iou == pytest.approx(32 / 48)
    assert result.added_floor_pixels == 8
    assert result.removed_floor_pixels == 0
    fused_labels = np.asarray(result.seg.labels)
    assert (fused_labels[:, :5] == MASK_FLOOR).all()
    assert fused_labels[:, 5].tolist() == [MASK_WALL] * 8
    assert np.array_equal(result.seg.probs, base.probs)
    assert np.array_equal(result.seg.sky_mask, base.sky_mask)

    strict = _load_config(
        tmp_path, min_accept_iou=0.9, veto_prob=0.8, name="strict.json"
    )
    rejected = wrap_seg_with_fusion(_FixedSeg(base), sam, strict).fuse(image, base)
    assert not rejected.accepted
    assert rejected.reason.startswith("fallback:low_iou:")
    assert rejected.seg is base
    assert (np.asarray(rejected.seg.labels) == np.asarray(base.labels)).all()


def test_f15_disabled_segformer_only_config_never_uses_sam(tmp_path):
    report = _valid_report(variant="segformer_only")
    config = load_tuned_fusion_config(_write_report(tmp_path, report))
    assert not config.enabled
    assert config.min_accept_iou is None
    assert config.veto_prob is None
    assert config.selection_reason == (
        "best_eligible_config_did_not_improve_on_calibration"
    )

    base = _test_base_output()
    sam = _FixedSAM(np.zeros((8, 8), dtype=bool))
    backend = wrap_seg_with_fusion(_FixedSeg(base), sam, config)
    assert backend is not None
    assert not hasattr(backend, "fuse")
    result = backend.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result is base
    assert sam.calls == 0

    enabled = _load_config(tmp_path, name="enabled.json")
    with pytest.raises(FusionConfigError, match="promptable-mask backend"):
        wrap_seg_with_fusion(_FixedSeg(base), None, enabled)


def test_f16_synthetic_reports_need_explicit_opt_in(tmp_path):
    report = _valid_report(mode="fake")
    path = _write_report(tmp_path, report)
    with pytest.raises(FusionConfigError, match="synthetic fake scenes"):
        load_tuned_fusion_config(path)
    config = load_tuned_fusion_config(path, allow_fake=True)
    assert config.enabled
    assert config.tuning_mode == "fake"


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        ("schema", lambda r: r.update(schema_version=2), "schema_version"),
        ("mode", lambda r: r.update(mode="synthetic"), "mode"),
        ("selection_missing", lambda r: r.pop("selection"), "selection"),
        (
            "variant",
            lambda r: r["selection"].update(variant="sam_only"),
            "variant",
        ),
        (
            "reason",
            lambda r: r["selection"].update(reason="trust_me"),
            "reason",
        ),
        (
            "eligible",
            lambda r: r["selection"].update(eligible=False),
            "eligible",
        ),
        (
            "improved",
            lambda r: r["selection"].update(improved=False),
            "improved",
        ),
        (
            "range",
            lambda r: r["selection"].update(min_accept_iou=1.5),
            "min_accept_iou",
        ),
        (
            "nan",
            lambda r: r["selection"].update(veto_prob=float("nan")),
            "finite",
        ),
        (
            "not_in_grid",
            lambda r: r["selection"].update(min_accept_iou=0.55),
            "swept grid",
        ),
        (
            "grid_missing",
            lambda r: r["grid"].pop("veto_prob"),
            "grid.veto_prob",
        ),
        (
            "grid_empty",
            lambda r: r.update(grid={"min_accept_iou": [], "veto_prob": [0.8]}),
            "grid.min_accept_iou",
        ),
        (
            "segformer_reason",
            lambda r: (
                r["selection"].update(
                    variant="segformer_only",
                    min_accept_iou=None,
                    veto_prob=None,
                    reason="trust_me",
                )
            ),
            "reason",
        ),
    ],
)
def test_f17_rejects_tampered_or_untrusted_selections(
    tmp_path, label, mutate, match
):
    report = copy.deepcopy(_valid_report())
    mutate(report)
    path = _write_report(tmp_path, report, f"{label}.json")
    with pytest.raises(FusionConfigError, match=match):
        load_tuned_fusion_config(path)


def test_f18_rejects_unreadable_and_malformed_files(tmp_path):
    with pytest.raises(FusionConfigError, match="cannot read"):
        load_tuned_fusion_config(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(FusionConfigError, match="not valid JSON"):
        load_tuned_fusion_config(malformed)

    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(FusionConfigError, match="JSON object"):
        load_tuned_fusion_config(not_object)


def test_f19_smoke_cli_uses_fake_tuned_fusion_and_reports_decisions(tmp_path, capsys):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_ppm(image_dir / "a.ppm", value=20)
    _write_ppm(image_dir / "b.ppm", value=60)
    report_path = _write_report(tmp_path, _valid_report(mode="fake"))
    out_dir = tmp_path / "out"

    code = smoke_main(
        [
            "--fake",
            "--images",
            str(image_dir),
            "--out",
            str(out_dir),
            "--fusion-config",
            str(report_path),
        ]
    )
    assert code == 0, capsys.readouterr()
    report = json.loads(
        (out_dir / "real_models_smoke.json").read_text(encoding="utf-8")
    )
    assert report["fusion_config"]["enabled"] is True
    assert report["fusion_config"]["tuning_mode"] == "fake"
    assert report["fusion_config"]["min_accept_iou"] == 0.6
    assert len(report["images"]) == 2
    for entry in report["images"]:
        assert entry["fusion"]["accepted"] is True
        assert entry["fusion"]["reason"] == "accepted"
        assert entry["fusion"]["prompt_count"] > 0
        assert entry["fusion"]["added_floor_pixels"] == 0
        assert entry["fusion"]["removed_floor_pixels"] == 0
        assert entry["class_histogram"]["floor"] > 0
        assert (out_dir / entry["overlay"]).exists()


def test_f19b_smoke_cli_refuses_a_fake_report_in_real_mode(tmp_path, capsys):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_ppm(image_dir / "a.ppm")
    report_path = _write_report(tmp_path, _valid_report(mode="fake"))
    code = smoke_main(
        [
            "--images",
            str(image_dir),
            "--out",
            str(tmp_path / "out"),
            "--weights-dir",
            str(tmp_path / "empty_weights"),
            "--fusion-config",
            str(report_path),
        ]
    )
    assert code == 2
    output = capsys.readouterr().out
    assert "synthetic fake scenes" in output
    assert "labelled real frames" in output


def test_f19c_smoke_cli_preflights_mobilesam_when_fusion_is_enabled(
    tmp_path, capsys
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_ppm(image_dir / "a.ppm")
    weights = tmp_path / "weights"
    (weights / "segformer-b0-ade").mkdir(parents=True)
    (weights / "segformer-b0-ade" / "config.json").write_text("{}", encoding="utf-8")
    (weights / "depth-anything-v2-metric-indoor-small").mkdir(parents=True)
    (weights / "depth-anything-v2-metric-indoor-small" / "config.json").write_text(
        "{}", encoding="utf-8"
    )
    (weights / "yolo11n.pt").write_bytes(b"")
    report_path = _write_report(tmp_path, _valid_report(mode="real"))

    code = smoke_main(
        [
            "--images",
            str(image_dir),
            "--out",
            str(tmp_path / "out"),
            "--weights-dir",
            str(weights),
            "--fusion-config",
            str(report_path),
        ]
    )
    assert code == 2
    output = capsys.readouterr().out
    assert "MobileSAM is not ready" in output
    assert "mobile_sam.pt" in output
    assert "fetch_models.py" in output
