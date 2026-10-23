"""Validated runtime artifact for split-conformal depth intervals.

``scripts/run_calibration_real.py`` fits a :class:`ConformalDepth` model and
writes a ``depth_interval.json`` wrapper with provenance. This module loads it
for runtime obstacle lower bounds.

Validation covers provenance and arithmetic, not coverage or accuracy. A model
calibrated on fake scenes is refused for a real run unless the caller opts in,
so synthetic quantiles cannot silently place real obstacles. An unbounded or
degenerate interval is refused because it would collapse every obstacle onto
the camera. Every threshold in the artifact remains TO TUNE ON REAL DATA.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from calibration.conformal_depth import (
    SCORE_OVERESTIMATION,
    SCORE_TWO_SIDED,
    ConformalDepth,
)

SCHEMA_VERSION = 1
_MODES = frozenset({"fake", "real"})
_VALID_SCORES = frozenset({SCORE_TWO_SIDED, SCORE_OVERESTIMATION})


class DepthIntervalConfigError(ValueError):
    """Raised when a depth-interval artifact cannot safely configure runtime."""


@dataclass(frozen=True)
class RuntimeDepthInterval:
    """One fitted depth-interval model plus the provenance to audit it."""

    model: ConformalDepth
    mode: str
    source: str
    generated: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON artifact consumed by :func:`load_depth_interval`."""
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "source": self.source,
            "generated": self.generated,
            "model": self.model.to_dict(),
        }

    def provenance(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for run reports."""
        return {
            "mode": self.mode,
            "source": self.source,
            "generated": self.generated,
            "alpha": float(self.model.alpha),
            "score": str(self.model.score),
            "pooled_quantile": float(self.model.pooled_quantile_),
            "pooled_count": int(self.model.pooled_count_),
            "n_valid": int(self.model.n_valid_),
            "n_invalid": int(self.model.n_invalid_),
        }


def load_depth_interval(
    path: str | Path,
    *,
    allow_fake: bool = False,
) -> RuntimeDepthInterval:
    """Load a validated conformal depth-interval artifact.

    Raises:
        DepthIntervalConfigError: the file is unreadable or malformed, the
            provenance is missing, a synthetic artifact is used in a real run,
            the fitted model is inconsistent, or the pooled quantile is
            unbounded/negative.
    """
    artifact_path = Path(path)
    try:
        raw = artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DepthIntervalConfigError(
            f"cannot read depth interval {artifact_path}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DepthIntervalConfigError(
            f"depth interval {artifact_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DepthIntervalConfigError("depth interval must contain a JSON object")

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DepthIntervalConfigError(
            f"unsupported depth interval schema_version {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    mode = payload.get("mode")
    if mode not in _MODES:
        raise DepthIntervalConfigError(
            f"depth interval mode must be 'real' or 'fake', got {mode!r}"
        )
    if mode == "fake" and not allow_fake:
        raise DepthIntervalConfigError(
            "depth interval was fitted on synthetic fake data; refusing to "
            "configure real obstacle bounds. Rerun scripts/run_calibration_real.py "
            "on real labelled routes, or pass allow_fake only for a fake-mode run."
        )
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise DepthIntervalConfigError(
            "depth interval is missing a non-empty source string"
        )
    generated = payload.get("generated")
    if generated is not None and not isinstance(generated, str):
        raise DepthIntervalConfigError(
            "depth interval generated field must be a string or null"
        )

    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise DepthIntervalConfigError(
            "depth interval model must be a JSON object"
        )
    if model_payload.get("schema_version") != SCHEMA_VERSION:
        raise DepthIntervalConfigError(
            "depth interval model has an unsupported schema_version"
        )
    try:
        model = ConformalDepth.from_dict(model_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise DepthIntervalConfigError(
            f"depth interval model is invalid: {exc}"
        ) from exc

    if model.score not in _VALID_SCORES:
        raise DepthIntervalConfigError(
            f"depth interval score must be one of {sorted(_VALID_SCORES)}, "
            f"got {model.score!r}"
        )
    quantile = float(model.pooled_quantile_)
    if not math.isfinite(quantile) or quantile < 0.0:
        raise DepthIntervalConfigError(
            "depth interval pooled quantile is unbounded or invalid; "
            "calibrate on more labelled points"
        )
    if model.pooled_count_ < 1 or model.n_valid_ < 1:
        raise DepthIntervalConfigError(
            "depth interval contains no valid calibration points"
        )
    if model.pooled_count_ != model.n_valid_:
        raise DepthIntervalConfigError(
            "depth interval calibration counts are inconsistent "
            f"(pooled_count={model.pooled_count_}, n_valid={model.n_valid_})"
        )

    return RuntimeDepthInterval(
        model=model,
        mode=mode,
        source=source,
        generated=generated,
    )


__all__ = [
    "DepthIntervalConfigError",
    "RuntimeDepthInterval",
    "SCHEMA_VERSION",
    "load_depth_interval",
]
