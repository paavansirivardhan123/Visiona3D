"""Tests M10 and M8: Guidance telemetry, decisions, and conformal pessimistic margin."""

from __future__ import annotations

import math
import numpy as np
import pytest

from calibration.abstain import AbstainDecision, AbstainFeatures
from calibration.conformal_depth import ConformalDepth
from core.config import Intrinsics
from geometry.camera import backproject
from geometry.frames import camera_to_ground
from mapping.bev import build_bev
from mapping.grid import CELL_FREE, CELL_OCCUPIED, MapConfig, world_to_cell
from planning.astar import astar_plan, PathResult
from planning.costmap import build_costmap
from planning.guidance import compute_guidance, Guidance
from planning.targets import find_explore_target
from tests.synthetic import render_scene


def test_m10_guidance_decisions_and_reasons_on_hand_crafted_cases():
    """M10: Hand-made maps and features give expected decision and reasons for GUIDE, CAUTION, STOP_UNCLEAR, STOP_BLOCKED."""
    cfg = MapConfig(res=0.10, x_min=-2.0, x_max=2.0, y_min=0.0, y_max=5.0)

    # 1. STOP_UNCLEAR: Abstention rule triggers
    grid_clear = np.full((cfg.n_rows, cfg.n_cols), CELL_FREE, dtype=np.uint8)
    costmap = build_costmap(grid_clear, cfg=cfg, user_half_width_m=0.10, safety_margin_m=0.0)

    abstain_stop = AbstainDecision(
        abstain=True,
        reasons=["certified floor fraction in corridor 0.150 < 0.300"],
        features=AbstainFeatures(
            certified_floor_fraction=0.15,
            relative_interval_width=0.2,
            plane_inlier_ratio=0.8,
            plane_status="OK",
            scale_status="OK",
            veto_fraction=0.0,
        ),
    )
    g_unclear = compute_guidance(costmap, abstain_decision=abstain_stop)
    assert g_unclear.decision == "STOP_UNCLEAR"
    assert any("certified floor fraction" in r for r in g_unclear.reasons)

    # 2. STOP_BLOCKED: No path possible
    grid_blocked = np.full((cfg.n_rows, cfg.n_cols), CELL_OCCUPIED, dtype=np.uint8)
    cm_blocked = build_costmap(grid_blocked, cfg=cfg)
    path_blocked = PathResult("NO_PATH", [], [], math.inf, 0.0, [], 0.0)
    g_blocked = compute_guidance(cm_blocked, path_result=path_blocked)
    assert g_blocked.decision == "STOP_BLOCKED"
    assert any("NO_PATH" in r for r in g_blocked.reasons)

    # 3. CAUTION: Path clear but minimum clearance < caution threshold (0.20m)
    path_caution = PathResult(
        status="OK",
        path=[(20, 10), (20, 20)],
        path_world=[(0.0, 1.0), (0.0, 2.0)],
        cost=1.0,
        length_m=1.0,
        clearances=[0.12, 0.15],
        min_clearance_m=0.12,
    )
    g_caution = compute_guidance(costmap, path_result=path_caution, caution_clearance_m=0.20)
    assert g_caution.decision == "CAUTION"
    assert any("below caution threshold" in r for r in g_caution.reasons)

    # 4. GUIDE: Path clear and clearance >= caution threshold
    path_guide = PathResult(
        status="OK",
        path=[(20, 10), (20, 20)],
        path_world=[(0.0, 1.0), (0.0, 2.0)],
        cost=1.0,
        length_m=1.0,
        clearances=[0.45, 0.50],
        min_clearance_m=0.45,
    )
    g_guide = compute_guidance(costmap, path_result=path_guide, caution_clearance_m=0.20)
    assert g_guide.decision == "GUIDE"
    assert any("guidance active" in r for r in g_guide.reasons)


def _guidance_with_interval(nearest_obstacle):
    cfg = MapConfig(res=0.10, x_min=-2.0, x_max=2.0, y_min=0.0, y_max=5.0)
    grid = np.full((cfg.n_rows, cfg.n_cols), CELL_FREE, dtype=np.uint8)
    costmap = build_costmap(grid, cfg=cfg, user_half_width_m=0.10, safety_margin_m=0.0)
    path = PathResult(
        status="OK",
        path=[(20, 10), (20, 20)],
        path_world=[(0.0, 1.0), (0.0, 2.0)],
        cost=1.0,
        length_m=1.0,
        clearances=[0.45, 0.50],
        min_clearance_m=0.45,
    )
    points = np.zeros((10, 3), dtype=np.float64)
    points[:, 1] = np.linspace(1.0, 3.0, 10)
    certified = np.ones(10, dtype=bool)
    return compute_guidance(
        costmap,
        path_result=path,
        points_g=points,
        certified_floor_mask=certified,
        plane_inlier_ratio=1.0,
        plane_status="OK",
        scale_status="OK",
        nearest_obstacle=nearest_obstacle,
    )


def test_guidance_abstains_when_interval_is_too_wide():
    """A wide nearest-obstacle interval must abstain, not guide."""
    guidance = _guidance_with_interval(
        {
            "label": "obstacle",
            "distance_nominal_m": 2.0,
            "distance_lower_m": 0.5,
            "distance_upper_m": 4.0,
            "bearing_deg": 0.0,
        }
    )
    assert guidance.decision == "STOP_UNCLEAR"
    assert any("relative interval width" in reason for reason in guidance.reasons)
    assert guidance.abstain_features["relative_interval_width"] == pytest.approx(1.75)


def test_guidance_passes_and_uses_interval_when_narrow():
    guidance = _guidance_with_interval(
        {
            "label": "obstacle",
            "distance_nominal_m": 2.0,
            "distance_lower_m": 1.8,
            "distance_upper_m": 2.2,
            "bearing_deg": 0.0,
        }
    )
    assert guidance.decision == "GUIDE"
    assert guidance.abstain_features["relative_interval_width"] == pytest.approx(0.2)


def test_guidance_skips_interval_check_without_bounds():
    guidance = _guidance_with_interval(
        {"label": "obstacle", "distance_nominal_m": 2.0, "bearing_deg": 0.0}
    )
    assert guidance.decision == "GUIDE"
    assert guidance.abstain_features["relative_interval_width"] is None


def test_guidance_abstains_on_inverted_interval():
    guidance = _guidance_with_interval(
        {
            "label": "obstacle",
            "distance_nominal_m": 3.0,
            "distance_lower_m": 3.5,
            "distance_upper_m": 2.5,
            "bearing_deg": 0.0,
        }
    )
    assert guidance.decision == "STOP_UNCLEAR"
    assert any("not finite" in reason for reason in guidance.reasons)


def test_m8_conformal_margin_hand_computed_distance_and_clearance():
    """M8: Stub q = 0.20 with obstacle at 4.0 m places pessimistic obstacle at 3.2 m."""
    # 1. Stub ConformalDepth with known quantile q = 0.20
    cdepth = ConformalDepth(alpha=0.10, score="overestimation")
    fake_gt = np.linspace(1.0, 6.0, 100)
    fake_pred = fake_gt * 1.25  # (pred - gt)/pred = 0.25/1.25 = 0.20
    cdepth.fit(fake_pred, fake_gt)

    nominal_dist = 4.0
    pred_res = cdepth.predict([nominal_dist])
    pessimistic_dist = float(pred_res.lower[0])

    # Hand-computed: 4.0 * (1 - 0.20) = 3.20 m
    assert math.isclose(pessimistic_dist, 3.20, abs_tol=1e-3)

    # 2. In grid coordinates with resolution 0.10 m:
    cfg = MapConfig(res=0.10, y_min=0.0, y_max=6.0)
    _, r_nom = world_to_cell(0.0, nominal_dist, cfg)
    _, r_pess = world_to_cell(0.0, pessimistic_dist, cfg)

    # Delta rows = floor(4.0/0.1) - floor(3.2/0.1) = 40 - 32 = 8 rows
    assert r_nom - r_pess == 8
    # The clearance to obstacle at (0, 0) is reduced by exactly 0.80 m
    delta_clearance = nominal_dist - pessimistic_dist
    assert math.isclose(delta_clearance, 0.80, abs_tol=1e-3)


def test_m8_pessimistic_layer_in_bev_and_true_clearance_gain():
    """M8: The pessimistic BEV layer places the wall at 3.2 m; the planned path
    keeps 0.80 m more true clearance to the wall at 4.0 m (hand-computed)."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    h, pitch = 1.4, 15.0
    wall_y = 4.0
    depth, floor_mask, wall_mask, _, _ = render_scene(K, 240, 320, h, pitch, wall_distance=wall_y)

    theta = math.radians(pitch)
    n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
    pts_g = camera_to_ground(backproject(depth, K), n, h)

    # Stub ConformalDepth with q = 0.20 (same fit as the hand computation above)
    cdepth = ConformalDepth(alpha=0.10, score="overestimation")
    fake_gt = np.linspace(1.0, 6.0, 100)
    cdepth.fit(fake_gt * 1.25, fake_gt)
    lower_depth = cdepth.predict(depth.reshape(-1)).lower.reshape(depth.shape)
    lower_pts_g = camera_to_ground(backproject(lower_depth, K), n, h)

    labels = np.zeros((240, 320), dtype=np.int64)
    labels[floor_mask] = 1
    labels[wall_mask] = 2

    cfg = MapConfig(res=0.10, y_min=0.0, y_max=6.0)
    bev_nom = build_bev(pts_g, labels, certified_final=floor_mask, cfg=cfg)
    bev_pess = build_bev(
        pts_g, labels, certified_final=floor_mask, pessimistic_pointmap=lower_pts_g, cfg=cfg
    )

    # Hand computation: nominal wall at Y = 4.0 -> row 40;
    # lower bound 4.0 * (1 - 0.2) = 3.2 -> row 32
    c_center, r_nom = world_to_cell(0.0, 4.0, cfg)
    _, r_pess = world_to_cell(0.0, 3.2, cfg)
    assert (r_nom, r_pess) == (40, 32)
    assert bev_nom.grid[r_nom, c_center] == CELL_OCCUPIED
    assert bev_pess.grid[r_pess, c_center] == CELL_OCCUPIED
    # The nominal layer alone does not block at 3.2 m: the pessimistic layer did it
    assert bev_nom.grid[r_pess, c_center] != CELL_OCCUPIED

    # Planner from (0, 1.8) with default inflation (0.35 + 0.10 = 0.45 m).
    # Visible floor starts at Y ~ 1.4 (row 14); rows 0..13 are UNKNOWN and
    # Safety Principle 1 inflation blocks rows <= 17, so the start at row 18
    # (cell-centre Y = 1.85) is the first passable row ahead of the user.
    # Nominal occupied band at row 40 -> last passable row 35 (Y = 3.55).
    # Pessimistic band at row 32 -> last passable row 27 (Y = 2.75).
    # True clearance to the wall: 4.0 - 3.55 = 0.45 m vs 4.0 - 2.75 = 1.25 m
    # -> gain of exactly 0.80 m, the conformal margin.
    start = world_to_cell(0.0, 1.8, cfg)
    max_y = {}
    for name, bev in (("nominal", bev_nom), ("pessimistic", bev_pess)):
        cm = build_costmap(bev.grid, cfg=cfg)
        target = find_explore_target(cm, start)
        assert target is not None, f"{name}: no explore target"
        res = astar_plan(cm, start, target)
        assert res.status == "OK", f"{name}: planner status {res.status}"
        max_y[name] = max(wy for _, wy in res.path_world)

    assert math.isclose(max_y["nominal"], 3.55, abs_tol=0.06)
    assert math.isclose(max_y["pessimistic"], 2.75, abs_tol=0.06)
    true_clearance_gain = (wall_y - max_y["pessimistic"]) - (wall_y - max_y["nominal"])
    assert math.isclose(true_clearance_gain, 0.80, abs_tol=0.06)
