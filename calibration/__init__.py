"""Conformal calibration, certified floor, geometric veto and abstention.

The modules in this package operate on the existing Visiona3D representations:

* depth is metric z-depth in camera frame C (metres);
* masks use the ``core.config`` class indices (0 unlabeled, 1 floor,
  2 wall, 3 obstacle, 4 other, 255 ignore);
* lifted points are ``[X, Y, Z]`` in the gravity-aligned ground frame G,
  where ``Y`` is forward distance and ``Z`` is height above the fitted floor.

The statistical methods are deliberately conservative.  They do not make a
claim of per-frame safety; see ``docs/calibration.md`` for assumptions and
limitations.
"""

from calibration.conformal_depth import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_GROUP_SIZE,
    SCORE_OVERESTIMATION,
    SCORE_TWO_SIDED,
    ConformalDepth,
    ConformalPrediction,
    conformal_quantile,
    depth_bin_groups,
)
from calibration.conformal_floor import (
    DEFAULT_DELTA,
    DEFAULT_LAMBDA_GRID,
    FloorCalibration,
    certified_floor,
    floor_loss,
    select_floor_threshold,
)
from calibration.geometric_veto import (
    DEFAULT_TAU0_M,
    DEFAULT_TAU1,
    combine_certified_floor,
    geometric_veto,
    height_threshold,
    veto_recall_by_band,
)
from calibration.abstain import (
    AbstainConfig,
    AbstainDecision,
    AbstainFeatures,
    AbstainRule,
    frame_is_unsafe,
    sweep_thresholds,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_GROUP_SIZE",
    "SCORE_TWO_SIDED",
    "SCORE_OVERESTIMATION",
    "ConformalDepth",
    "ConformalPrediction",
    "conformal_quantile",
    "depth_bin_groups",
    "DEFAULT_DELTA",
    "DEFAULT_LAMBDA_GRID",
    "FloorCalibration",
    "certified_floor",
    "floor_loss",
    "select_floor_threshold",
    "DEFAULT_TAU0_M",
    "DEFAULT_TAU1",
    "combine_certified_floor",
    "geometric_veto",
    "height_threshold",
    "veto_recall_by_band",
    "AbstainConfig",
    "AbstainDecision",
    "AbstainFeatures",
    "AbstainRule",
    "frame_is_unsafe",
    "sweep_thresholds",
]
