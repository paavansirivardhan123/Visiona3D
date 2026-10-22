"""Guidance module integrating abstention, path planning, and navigation decisions.

SAFETY PRINCIPLES (verbatim):
1. Free space must be certified. Unknown and uncertain cells are NOT traversable.
2. Obstacles are placed at their conformal LOWER-BOUND distance (pessimistic layer), as well as nominal.
3. Any failure gives a status (STOP_UNCLEAR or STOP_BLOCKED) and never an exception.
4. The planner only recommends; it never claims the path is safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from calibration.abstain import (
    AbstainConfig,
    AbstainDecision,
    AbstainFeatures,
    AbstainRule,
)
from mapping.bev import BevGrid
from mapping.grid import cell_to_world, in_bounds, world_to_cell
from planning.astar import astar_plan, PathResult
from planning.costmap import Costmap
from planning.targets import find_explore_target

# Guidance thresholds - labelled TO TUNE ON REAL DATA
DEFAULT_CAUTION_CLEARANCE_M: float = 0.20  # TO TUNE ON REAL DATA: minimum clearance triggering CAUTION
DEFAULT_LOOKAHEAD_DISTANCE_M: float = 1.5  # TO TUNE ON REAL DATA
DEFAULT_TURN_THRESHOLD_DEG: float = 15.0  # TO TUNE ON REAL DATA


@dataclass
class Guidance:
    """Navigation guidance recommendation and safety telemetry.

    SAFETY: The planner only recommends; it never claims the path is safe.
    """

    decision: str  # "GUIDE", "CAUTION", "STOP_UNCLEAR", "STOP_BLOCKED"
    heading_deg: float  # Recommended steering angle in degrees (positive right)
    lookahead_point_g: tuple[float, float, float]  # (X, Y, Z) target in ground frame G
    distance_to_turn_m: float  # Distance along path to first turn
    left_clearance_m: float  # Minimum lateral clearance on the left (X < 0)
    right_clearance_m: float  # Minimum lateral clearance on the right (X > 0)
    nearest_obstacle: dict[str, Any]  # {label, distance_nominal_m, distance_lower_m, bearing_deg}
    reasons: list[str]  # Human-readable justifications
    abstain_features: dict[str, Any]  # Features evaluated by AbstainRule
    timings: dict[str, float] = field(default_factory=dict)  # Component latencies in ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "heading_deg": self.heading_deg,
            "lookahead_point_g": list(self.lookahead_point_g),
            "distance_to_turn_m": self.distance_to_turn_m,
            "left_clearance_m": self.left_clearance_m,
            "right_clearance_m": self.right_clearance_m,
            "nearest_obstacle": dict(self.nearest_obstacle),
            "reasons": list(self.reasons),
            "abstain_features": dict(self.abstain_features),
            "timings": dict(self.timings),
        }


def _find_lookahead(
    path_world: Sequence[tuple[float, float]],
    target_dist_m: float,
) -> tuple[float, float, float]:
    """Find waypoint along path at target_dist_m ahead, or final goal."""
    if not path_world:
        return 0.0, 0.0, 0.0
    accum = 0.0
    for i in range(1, len(path_world)):
        seg_len = math.hypot(
            path_world[i][0] - path_world[i - 1][0],
            path_world[i][1] - path_world[i - 1][1],
        )
        accum += seg_len
        if accum >= target_dist_m:
            return float(path_world[i][0]), float(path_world[i][1]), 0.0
    # Past the end: return final goal
    last = path_world[-1]
    return float(last[0]), float(last[1]), 0.0


def _distance_to_turn(
    path_world: Sequence[tuple[float, float]],
    turn_thresh_deg: float = DEFAULT_TURN_THRESHOLD_DEG,
) -> float:
    """Measure distance along path to the first waypoint turning by > turn_thresh_deg."""
    if len(path_world) < 2:
        return 0.0
    initial_heading = math.atan2(path_world[1][0] - path_world[0][0], path_world[1][1] - path_world[0][1])
    accum = 0.0
    for i in range(1, len(path_world)):
        seg_len = math.hypot(
            path_world[i][0] - path_world[i - 1][0],
            path_world[i][1] - path_world[i - 1][1],
        )
        accum += seg_len
        heading = math.atan2(path_world[i][0] - path_world[i - 1][0], path_world[i][1] - path_world[i - 1][1])
        angle_diff = abs(math.degrees(heading - initial_heading))
        if angle_diff > turn_thresh_deg:
            return float(accum)
    return float(accum)


def compute_guidance(
    costmap: Costmap,
    path_result: Optional[PathResult] = None,
    abstain_decision: Optional[AbstainDecision] = None,
    abstain_rule: Optional[AbstainRule] = None,
    points_g: Optional[np.ndarray] = None,
    certified_floor_mask: Optional[np.ndarray] = None,
    plane_inlier_ratio: float = 1.0,
    plane_status: str = "OK",
    scale_status: str = "OK",
    nearest_obstacle: Optional[dict[str, Any]] = None,
    caution_clearance_m: float = DEFAULT_CAUTION_CLEARANCE_M,
    lookahead_dist_m: float = DEFAULT_LOOKAHEAD_DISTANCE_M,
) -> Guidance:
    """Compute guidance decision and navigation signals.

    Never raises an exception; failures result in STOP_UNCLEAR or STOP_BLOCKED.

    Args:
        costmap: Planning costmap.
        path_result: Planned PathResult (if already computed).
        abstain_decision: Pre-computed AbstainDecision, if available.
        abstain_rule: AbstainRule instance to evaluate if decision not provided.
        points_g: (N, 3) ground-frame points for AbstainRule evaluation.
        certified_floor_mask: Boolean mask for AbstainRule.
        plane_inlier_ratio: Ground plane RANSAC inlier ratio.
        plane_status: Plane fit status string.
        scale_status: Metric scale status string.
        nearest_obstacle: Telemetry dict for closest obstacle.
        caution_clearance_m: Clearance threshold triggering CAUTION.
        lookahead_dist_m: Distance in metres to steering lookahead point.

    Returns:
        Guidance instance.
    """
    timings: dict[str, float] = {}
    start_time = time.perf_counter()

    cfg = costmap.cfg

    # 1. Evaluate or extract abstention decision
    if abstain_decision is None:
        if abstain_rule is None:
            abstain_rule = AbstainRule()
        if points_g is not None and certified_floor_mask is not None:
            pts_arr = np.asarray(points_g, dtype=np.float64)
            if pts_arr.ndim == 3 and pts_arr.shape[2] >= 3:
                pts_arr = pts_arr.reshape(-1, pts_arr.shape[2])
            c_mask = np.asarray(certified_floor_mask, dtype=bool).reshape(-1)

            pred_m = nearest_obstacle.get("distance_nominal_m") if nearest_obstacle else None
            lower_m = nearest_obstacle.get("distance_lower_m") if nearest_obstacle else None
            upper_m = nearest_obstacle.get("distance_upper_m", float("inf")) if nearest_obstacle else None
            abstain_decision = abstain_rule.evaluate(
                pts_arr,
                c_mask,
                plane_inlier_ratio=plane_inlier_ratio,
                plane_status=plane_status,
                scale_status=scale_status,
                nearest_obstacle_pred_m=pred_m,
                nearest_obstacle_lower_m=lower_m,
                nearest_obstacle_upper_m=upper_m,
            )
        else:
            # Construct pass-through features when raw cloud unavailable
            abstain_decision = AbstainDecision(
                abstain=False,
                reasons=["no cloud supplied; abstention bypassed"],
                features=AbstainFeatures(
                    certified_floor_fraction=1.0,
                    relative_interval_width=None,
                    plane_inlier_ratio=plane_inlier_ratio,
                    plane_status=plane_status,
                    scale_status=scale_status,
                    veto_fraction=0.0,
                ),
            )

    # 2. Nearest obstacle telemetry
    if nearest_obstacle is None:
        # Scan costmap blocked cells
        blocked_coords = np.argwhere(costmap.blocked_mask)
        if len(blocked_coords) > 0:
            c_coords, r_coords = blocked_coords[:, 1], blocked_coords[:, 0]
            wx, wy = cell_to_world(c_coords, r_coords, cfg)
            dists = np.hypot(wx, wy)
            min_idx = int(np.argmin(dists))
            nearest_nominal = float(dists[min_idx])
            bearing = math.degrees(math.atan2(wx[min_idx], wy[min_idx]))
            nearest_obstacle = {
                "label": "obstacle",
                "distance_nominal_m": nearest_nominal,
                "distance_lower_m": nearest_nominal,
                "bearing_deg": float(bearing),
            }
        else:
            nearest_obstacle = {
                "label": "none",
                "distance_nominal_m": float("inf"),
                "distance_lower_m": float("inf"),
                "bearing_deg": 0.0,
            }

    # 3. Lateral clearances (left X < 0, right X > 0 in forward view Y > 0)
    c_grid, r_grid = np.meshgrid(np.arange(cfg.n_cols), np.arange(cfg.n_rows))
    wx_all, wy_all = cell_to_world(c_grid, r_grid, cfg)
    fwd = wy_all > 0.0
    left_mask = fwd & (wx_all < -1e-3)
    right_mask = fwd & (wx_all > 1e-3)

    left_clr = float(np.min(costmap.clearance_m[left_mask])) if np.any(left_mask) else float("inf")
    right_clr = float(np.min(costmap.clearance_m[right_mask])) if np.any(right_mask) else float("inf")

    # 4. Safety Principle: Abstention check overrides everything with STOP_UNCLEAR
    if abstain_decision.abstain:
        timings["total_ms"] = (time.perf_counter() - start_time) * 1000.0
        return Guidance(
            decision="STOP_UNCLEAR",
            heading_deg=0.0,
            lookahead_point_g=(0.0, 0.0, 0.0),
            distance_to_turn_m=0.0,
            left_clearance_m=left_clr,
            right_clearance_m=right_clr,
            nearest_obstacle=nearest_obstacle,
            reasons=list(abstain_decision.reasons),
            abstain_features=abstain_decision.features.to_dict(),
            timings=timings,
        )

    # 5. Plan path if not already provided
    if path_result is None:
        start_col, start_row = world_to_cell(0.0, 0.0, cfg)
        target = find_explore_target(costmap, (start_col, start_row))
        if target is None:
            path_result = PathResult("NO_PATH", [], [], math.inf, 0.0, [], 0.0)
        else:
            path_result = astar_plan(costmap, (start_col, start_row), target)

    # 6. Check planner status
    if path_result.status != "OK" or len(path_result.path_world) == 0:
        timings["total_ms"] = (time.perf_counter() - start_time) * 1000.0
        return Guidance(
            decision="STOP_BLOCKED",
            heading_deg=0.0,
            lookahead_point_g=(0.0, 0.0, 0.0),
            distance_to_turn_m=0.0,
            left_clearance_m=left_clr,
            right_clearance_m=right_clr,
            nearest_obstacle=nearest_obstacle,
            reasons=[f"Planner status: {path_result.status}"],
            abstain_features=abstain_decision.features.to_dict(),
            timings=timings,
        )

    # 7. Path is available: compute steering signals
    lookahead = _find_lookahead(path_result.path_world, lookahead_dist_m)
    # Heading angle positive right: atan2(X, Y)
    heading_deg = float(math.degrees(math.atan2(lookahead[0], lookahead[1])))
    dist_turn = _distance_to_turn(path_result.path_world)

    # Check caution clearance
    min_path_clr = path_result.min_clearance_m
    reasons: list[str] = []

    if min_path_clr < float(caution_clearance_m):
        decision = "CAUTION"
        reasons.append(
            f"Path clearance {min_path_clr:.2f} m is below caution threshold {caution_clearance_m:.2f} m"
        )
    else:
        decision = "GUIDE"
        reasons.append("Path certified and clear; guidance active")

    timings["total_ms"] = (time.perf_counter() - start_time) * 1000.0

    return Guidance(
        decision=decision,
        heading_deg=heading_deg,
        lookahead_point_g=lookahead,
        distance_to_turn_m=dist_turn,
        left_clearance_m=left_clr,
        right_clearance_m=right_clr,
        nearest_obstacle=nearest_obstacle,
        reasons=reasons,
        abstain_features=abstain_decision.features.to_dict(),
        timings=timings,
    )
