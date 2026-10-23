"""Validated runtime configuration from ``scripts/tune_fusion.py`` reports.

The loader checks provenance and ranges, not accuracy or safety. It accepts
only a schema-v1 tuning report whose fusion selection passed the tool's
eligibility and improvement flags and whose thresholds are values from the
swept grid. Synthetic reports are refused unless the caller explicitly opts in,
so fake thresholds cannot silently configure a real runtime.

A SegFormer-only selection is a valid outcome: the returned config is disabled
and the caller keeps the base segmentation backend. Selected thresholds remain
TO TUNE ON REAL DATA; loading them is not a safety guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from perception.seg_fusion import SegSAMFusion

SCHEMA_VERSION = 1
FUSION_REASONS = frozenset({"calibration_safety_first_winner"})
SEGFORMER_ONLY_REASONS = frozenset(
    {
        "no_grid_config_matched_segformer_floor_iou_and_safety",
        "best_eligible_config_did_not_improve_on_calibration",
    }
)
_MODES = frozenset({"fake", "real"})
_GRID_KEYS = ("min_accept_iou", "veto_prob")
_THRESHOLD_TOL = 1e-9


class FusionConfigError(ValueError):
    """Raised when a tuning report cannot safely configure fusion."""


@dataclass(frozen=True)
class RuntimeFusionConfig:
    """One validated configuration decision for a runtime process."""

    enabled: bool
    min_accept_iou: float | None
    veto_prob: float | None
    selection_reason: str
    tuning_mode: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable provenance record."""
        return {
            "enabled": self.enabled,
            "min_accept_iou": self.min_accept_iou,
            "veto_prob": self.veto_prob,
            "selection_reason": self.selection_reason,
            "tuning_mode": self.tuning_mode,
            "source": self.source,
        }


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FusionConfigError(f"{field} must be a number in [0, 1], got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise FusionConfigError(
            f"{field} must be a finite number in [0, 1], got {value!r}"
        )
    return number


def _grid_values(grid: Any, key: str) -> list[float]:
    if not isinstance(grid, dict):
        raise FusionConfigError("tuning report grid must be a JSON object")
    values = grid.get(key)
    if not isinstance(values, list) or not values:
        raise FusionConfigError(f"tuning report grid.{key} must be a non-empty list")
    return [
        _probability(value, f"grid.{key}[{index}]")
        for index, value in enumerate(values)
    ]


def _in_grid(value: float, grid: list[float]) -> bool:
    return any(
        math.isclose(value, candidate, rel_tol=0.0, abs_tol=_THRESHOLD_TOL)
        for candidate in grid
    )


def load_tuned_fusion_config(
    path: str | Path,
    *,
    allow_fake: bool = False,
) -> RuntimeFusionConfig:
    """Load a validated configuration from a ``fusion_tuning.json`` report.

    Raises:
        FusionConfigError: the file is unreadable or malformed, the schema or
            mode is unsupported, a synthetic report is used without
            ``allow_fake``, the selection is not an official tuning outcome, or
            the thresholds are not valid values from the reported sweep grid.
    """
    report_path = Path(path)
    try:
        raw = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FusionConfigError(
            f"cannot read fusion config {report_path}: {exc}"
        ) from exc
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FusionConfigError(
            f"fusion config {report_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise FusionConfigError("fusion config must contain a JSON object")

    version = report.get("schema_version")
    if version != SCHEMA_VERSION:
        raise FusionConfigError(
            f"unsupported fusion config schema_version {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    mode = report.get("mode")
    if mode not in _MODES:
        raise FusionConfigError(
            f"fusion config mode must be 'real' or 'fake', got {mode!r}"
        )
    if mode == "fake" and not allow_fake:
        raise FusionConfigError(
            "fusion config was tuned on synthetic fake scenes; refusing to "
            "configure real runtime. Rerun scripts/tune_fusion.py on labelled "
            "real frames, or pass allow_fake only for a fake-mode run."
        )

    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise FusionConfigError("fusion config selection must be a JSON object")
    variant = selection.get("variant")
    reason = selection.get("reason")
    source = str(report_path)
    if variant == "segformer_only":
        if reason not in SEGFORMER_ONLY_REASONS:
            raise FusionConfigError(
                f"segformer_only selection has unrecognised reason {reason!r}"
            )
        return RuntimeFusionConfig(
            enabled=False,
            min_accept_iou=None,
            veto_prob=None,
            selection_reason=str(reason),
            tuning_mode=mode,
            source=source,
        )
    if variant != "fusion":
        raise FusionConfigError(
            "fusion config selection.variant must be 'fusion' or "
            f"'segformer_only', got {variant!r}"
        )
    if reason not in FUSION_REASONS:
        raise FusionConfigError(
            f"fusion selection has unrecognised reason {reason!r}"
        )
    if selection.get("eligible") is not True:
        raise FusionConfigError("fusion selection must have eligible=true")
    if selection.get("improved") is not True:
        raise FusionConfigError("fusion selection must have improved=true")

    iou_grid = _grid_values(report.get("grid"), _GRID_KEYS[0])
    veto_grid = _grid_values(report.get("grid"), _GRID_KEYS[1])
    min_accept_iou = _probability(
        selection.get("min_accept_iou"), "selection.min_accept_iou"
    )
    veto_prob = _probability(selection.get("veto_prob"), "selection.veto_prob")
    if not _in_grid(min_accept_iou, iou_grid):
        raise FusionConfigError(
            "selection.min_accept_iou is not one of the swept grid values"
        )
    if not _in_grid(veto_prob, veto_grid):
        raise FusionConfigError(
            "selection.veto_prob is not one of the swept grid values"
        )
    return RuntimeFusionConfig(
        enabled=True,
        min_accept_iou=min_accept_iou,
        veto_prob=veto_prob,
        selection_reason=str(reason),
        tuning_mode=mode,
        source=source,
    )


def wrap_seg_with_fusion(
    seg: Any,
    sam: Any,
    config: RuntimeFusionConfig,
) -> Any:
    """Return the base backend when fusion is disabled, else SegSAMFusion."""
    if not config.enabled:
        return seg
    if sam is None:
        raise FusionConfigError(
            "fusion is enabled but no promptable-mask backend was supplied"
        )
    if config.min_accept_iou is None or config.veto_prob is None:
        raise FusionConfigError(
            "enabled fusion config is missing selected thresholds"
        )
    return SegSAMFusion(
        seg,
        sam,
        min_accept_iou=config.min_accept_iou,
        veto_prob=config.veto_prob,
    )


__all__ = [
    "FusionConfigError",
    "RuntimeFusionConfig",
    "load_tuned_fusion_config",
    "wrap_seg_with_fusion",
]
