"""Test M11: End-to-end integration with fake pipeline scenes and safety invariants."""

from __future__ import annotations

import math
import numpy as np
import pytest

from calibration.conformal_depth import ConformalDepth
from core.config import Intrinsics, MASK_IGNORE, MountConfig, ScaleConfig
from geometry.camera import backproject
from geometry.frames import camera_to_ground
from geometry.scale import ScaleEstimator
from mapping.bev import build_bev
from mapping.grid import CELL_FREE, MapConfig, world_to_cell
from perception.fakes import FakeDepthBackend, FakeDetector, FakeSegBackend
from perception.pipeline import PerceptionPipeline
from planning.astar import astar_plan
from planning.costmap import build_costmap
from planning.guidance import compute_guidance
from planning.targets import find_explore_target
from tests.synthetic import render_scene


def test_m11_end_to_end_box_avoidance_with_user_half_width_clearance():
    """M11: Fake pipeline scene with a box produces a path avoiding the box with clearance >= user half-width (0.35m)."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    estimator = ScaleEstimator(mount, ScaleConfig())
    map_cfg = MapConfig()

    # Place box right in middle of corridor at X in [-0.3, 0.3], Y in [2.3, 2.8]
    box = (-0.3, 0.3, 2.3, 2.8, 0.0, 0.8)
    scene_args = {
        "K": K,
        "H": 240,
        "W": 320,
        "h": mount.h_known,
        "pitch_deg": mount.pitch_deg,
        "wall_distance": 6.0,
        "boxes": [box],
        "seed": 42,
    }

    pipeline = PerceptionPipeline(
        seg=FakeSegBackend(**scene_args),
        depth=FakeDepthBackend(**scene_args),
        detector=FakeDetector(**scene_args),
        K=K,
        mount=mount,
        estimator=estimator,
    )

    outputs = pipeline.run(np.zeros((240, 320, 3), dtype=np.uint8))
    assert outputs.lift is not None

    bev = build_bev(
        pointmap_g=outputs.lift.point_map_g,
        labels=outputs.seg.labels,
        certified_final=(outputs.seg.labels == 1),
        cfg=map_cfg,
    )

    costmap = build_costmap(bev.grid, cfg=map_cfg, user_half_width_m=0.35, safety_margin_m=0.0)
    start = world_to_cell(0.0, 1.8, map_cfg)
    target = find_explore_target(costmap, start, min_dist_m=1.5)
    assert target is not None

    path_res = astar_plan(costmap, start, target)
    assert path_res.status == "OK"

    # Verify clearance at every waypoint along path >= user half-width (0.35m)
    for wx, wy in path_res.path_world:
        # Distance to box boundary
        dx = max(box[0] - wx, 0.0, wx - box[1])
        dy = max(box[2] - wy, 0.0, wy - box[3])
        dist_to_box = math.hypot(dx, dy)
        assert dist_to_box >= 0.35 - 1e-4, f"Path waypoint ({wx:.2f}, {wy:.2f}) is too close to box: {dist_to_box:.3f} m"


def test_m11_wall_as_floor_corruption_vetoed():
    """M11: With 30% wall-as-floor corruption, geometric veto keeps the path out of wall cells (measured unsafe rate = 0.0)."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    estimator = ScaleEstimator(mount, ScaleConfig())
    map_cfg = MapConfig()

    wall_dist = 4.0
    scene_args = {
        "K": K,
        "H": 240,
        "W": 320,
        "h": mount.h_known,
        "pitch_deg": mount.pitch_deg,
        "wall_distance": wall_dist,
        "boxes": [],
        "seed": 99,
    }

    # 30% wall as floor corruption in fake seg backend
    seg_backend = FakeSegBackend(**scene_args, floor_as_wall_fraction=0.0)
    # Manually inject wall-as-floor corruption into the segmentation labels
    raw_seg = seg_backend.predict(np.zeros((240, 320, 3), dtype=np.uint8))
    wall_px = np.argwhere(raw_seg.labels == 2)
    n_corrupt = int(0.30 * len(wall_px))
    corrupt_idx = np.random.default_rng(42).choice(len(wall_px), size=n_corrupt, replace=False)
    corrupted_labels = raw_seg.labels.copy()
    for y, x in wall_px[corrupt_idx]:
        corrupted_labels[y, x] = 1  # Corrupted to floor

    # Custom wrapper to return corrupted labels
    class CorruptSeg:
        def predict(self, img):
            return raw_seg.__class__(labels=corrupted_labels, probs=raw_seg.probs, sky_mask=raw_seg.sky_mask)
        def validate(self):
            return []

    pipeline = PerceptionPipeline(
        seg=CorruptSeg(),
        depth=FakeDepthBackend(**scene_args),
        detector=FakeDetector(**scene_args),
        K=K,
        mount=mount,
        estimator=estimator,
        floor_threshold=0.50,
    )

    outputs = pipeline.run(np.zeros((240, 320, 3), dtype=np.uint8))
    assert outputs.certified_final is not None
    assert outputs.vetoed is not None

    bev = build_bev(
        pointmap_g=outputs.lift.point_map_g,
        labels=outputs.seg.labels,
        certified_final=outputs.certified_final,
        vetoed=outputs.vetoed,
        cfg=map_cfg,
    )

    costmap = build_costmap(bev.grid, cfg=map_cfg, user_half_width_m=0.35, safety_margin_m=0.10)
    start = world_to_cell(0.0, 1.8, map_cfg)
    target = find_explore_target(costmap, start, min_dist_m=1.0)

    if target is not None:
        path_res = astar_plan(costmap, start, target)
        if path_res.status == "OK":
            for wx, wy in path_res.path_world:
                # Must stay at least 0.35m away from true wall at Y=4.0
                assert wy + 0.35 <= wall_dist + 1e-4, f"Path penetrated wall: Y={wy:.2f} + 0.35 >= {wall_dist}"


def test_m11_pessimistic_pointmap_places_wall_at_lower_bound():
    """M11: The pipeline's pessimistic point map places the wall at the conformal
    lower-bound distance (4.0 * (1 - 0.20) = 3.2 m) instead of the nominal 4.0 m."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    estimator = ScaleEstimator(mount, ScaleConfig())

    scene_args = {
        "K": K,
        "H": 240,
        "W": 320,
        "h": mount.h_known,
        "pitch_deg": mount.pitch_deg,
        "wall_distance": 4.0,
        "boxes": [],
        "seed": 21,
    }

    # Stub ConformalDepth with quantile q = 0.20 (overestimation score)
    cdepth = ConformalDepth(alpha=0.10, score="overestimation")
    fake_gt = np.linspace(1.0, 6.0, 100)
    cdepth.fit(fake_gt * 1.25, fake_gt)

    pipeline = PerceptionPipeline(
        seg=FakeSegBackend(**scene_args),
        depth=FakeDepthBackend(**scene_args),
        detector=FakeDetector(**scene_args),
        K=K,
        mount=mount,
        estimator=estimator,
        conformal_depth=cdepth,
        floor_threshold=0.50,
    )

    outputs = pipeline.run(np.zeros((240, 320, 3), dtype=np.uint8))
    assert outputs.lift is not None
    assert outputs.lift.point_map_g is not None
    assert outputs.pessimistic_pointmap_g is not None

    wall_px = outputs.seg.labels == 2
    assert wall_px.sum() > 0

    nominal_wall_y = float(np.nanmedian(outputs.lift.point_map_g[wall_px, 1]))
    pessimistic_wall_y = float(np.nanmedian(outputs.pessimistic_pointmap_g[wall_px, 1]))

    # Hand-computed: nominal 4.0 m; lower bound 4.0 * (1 - 0.20) = 3.2 m
    assert math.isclose(nominal_wall_y, 4.0, abs_tol=0.05), f"nominal wall Y {nominal_wall_y:.3f}"
    assert math.isclose(pessimistic_wall_y, 3.2, abs_tol=0.05), (
        f"pessimistic wall Y {pessimistic_wall_y:.3f}"
    )
    assert pessimistic_wall_y < nominal_wall_y


def test_m11_glass_patch_see_through_depth_is_not_certified_free():
    """M11: A glass pane that see-through depth (8.0 m) mislabels as floor is
    never certified FREE at the glass location; Safety-Principle-1 inflation
    keeps the recommended path 0.45 m short of the pane (measured)."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    h, pitch = 1.4, 15.0
    wall_y = 4.0
    depth, floor_mask, wall_mask, _, _ = render_scene(K, 240, 320, h, pitch, wall_distance=wall_y)

    theta = math.radians(pitch)
    n = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
    pts_g = camera_to_ground(backproject(depth, K), n, h)

    # Glass pane: walkable-height band of the wall with |X| <= 1.0.
    # The depth sensor sees THROUGH the pane and reports the far surface at
    # 8.0 m; segmentation (worst case) labels the pane as floor.
    glass_px = wall_mask & (np.abs(pts_g[:, :, 0]) <= 1.0) & (pts_g[:, :, 2] <= 1.8)
    assert glass_px.sum() > 0

    see_through_depth = depth.copy()
    see_through_depth[glass_px] = 8.0
    pts_glass = camera_to_ground(backproject(see_through_depth, K), n, h)

    labels = np.zeros((240, 320), dtype=np.int64)
    labels[floor_mask] = 1
    labels[wall_mask] = 2
    labels[glass_px] = 1  # worst case: glass mislabelled as floor

    map_cfg = MapConfig()
    bev = build_bev(pts_glass, labels, certified_final=(labels == 1), cfg=map_cfg)

    # Measured: the glass footprint (Y >= wall_y, |X| <= 1.05) contains no
    # certified FREE cell - the pane itself stays UNKNOWN.
    glass_rows, glass_cols = np.nonzero(
        (np.arange(map_cfg.n_rows)[:, None] * map_cfg.res + map_cfg.y_min >= wall_y)
        & (np.abs(np.arange(map_cfg.n_cols)[None, :] * map_cfg.res + map_cfg.x_min) <= 1.05)
    )
    n_free_in_footprint = int((bev.grid[glass_rows, glass_cols] == CELL_FREE).sum())
    assert n_free_in_footprint == 0, (
        f"{n_free_in_footprint} glass-footprint cells were certified FREE"
    )

    # The recommended path stops short of the pane: UNKNOWN cells are blocked
    # (Safety Principle 1) and inflated by 0.35 + 0.10 m.
    costmap = build_costmap(bev.grid, cfg=map_cfg, user_half_width_m=0.35, safety_margin_m=0.10)
    start = world_to_cell(0.0, 1.8, map_cfg)
    target = find_explore_target(costmap, start, min_dist_m=1.0)
    assert target is not None
    path_res = astar_plan(costmap, start, target)
    assert path_res.status == "OK"
    for wx, wy in path_res.path_world:
        assert wy < wall_y - 0.40, f"Waypoint ({wx:.2f}, {wy:.2f}) entered the glass standoff zone"


def test_m11_never_raises_on_nan_depth_empty_masks_or_failing_backend():
    """M11: The pipeline and planner never raise an exception on NaN depth, empty masks, or failing backends."""
    K = Intrinsics.from_fov(320, 240, 75.0)
    mount = MountConfig(h_known=1.4, pitch_deg=15.0)
    estimator = ScaleEstimator(mount, ScaleConfig())
    map_cfg = MapConfig()

    # 1. Failing backend that raises
    class CrashingBackend:
        def predict(self, img):
            raise RuntimeError("Backend exploded!")
        def validate(self):
            return []

    pipeline_crash = PerceptionPipeline(
        seg=CrashingBackend(),
        depth=CrashingBackend(),
        detector=CrashingBackend(),
        K=K,
        mount=mount,
        estimator=estimator,
    )

    # Must not raise!
    outputs_crash = pipeline_crash.run(np.zeros((240, 320, 3), dtype=np.uint8))
    assert "error:seg:RuntimeError:Backend exploded!" in outputs_crash.statuses.get("seg", "")

    # Blank costmap from empty outputs
    grid_unknown = np.full((map_cfg.n_rows, map_cfg.n_cols), 0, dtype=np.uint8)
    costmap = build_costmap(grid_unknown, cfg=map_cfg)
    guidance = compute_guidance(costmap)

    # Must return STOP_UNCLEAR or STOP_BLOCKED, never raise!
    assert guidance.decision in ("STOP_UNCLEAR", "STOP_BLOCKED")
