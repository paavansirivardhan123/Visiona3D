"""Negative control experiments for Visiona3D geometry module.

Programmatically monkeypatches deliberate bugs into the codebase, runs
the relevant tests, asserts that each bug is caught by test failures,
and prints a verification table.

Deliberate bugs evaluated:
(a) Flip the sign of the normal in fit_ground_plane.
(b) Skip the scale correction in lift_frame.
(c) Use Euclidean range instead of z-depth in backproject.
(d) Swap e_right and e_fwd in camera_to_ground.
(e) Use the mean instead of the 10th percentile in object_anchor.
(f) Disable outlier rejection in ScaleEstimator.
(g) Splits allow the same route in two splits.
(h) AbsRel uses Euclidean range instead of z-depth.
(i) Coverage uses strict instead of inclusive bounds.
(j) Range-bin edges off by one (right-inclusive).
(k) Bootstrap resamples frames instead of routes.
(l) Interval score drops the penalty term.
(m) False-walkable rate counts ignore pixels.
(n) range_to_depth returns range unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from core.config import RANGE_BIN_EDGES

import geometry.camera as camera_mod
import geometry.frames as frames_mod
import geometry.ground_plane as gp_mod
import geometry.lift as lift_mod
import geometry.scale as scale_mod

from tests import (
    test_anchor,
    test_camera,
    test_frames,
    test_ground_plane,
    test_lift,
    test_scale,
    test_splits,
    test_metrics_depth,
    test_metrics_interval,
    test_metrics_seg,
    test_bootstrap,
)


class NoOpMutationError(RuntimeError):
    """Raised when an injected mutation produces output identical to the original."""


def _values_identical(a: Any, b: Any) -> bool:
    """Deep equality where matching NaN positions are treated as identical."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_values_identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_identical(x, y) for x, y in zip(a, b))
    try:
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape != bb.shape:
            return False
        return bool(np.array_equal(aa, bb, equal_nan=True))
    except Exception:
        return a == b


def assert_mutation_effective(name: str, orig_fn, mutated_fn, *args, **kwargs) -> None:
    """Verify on a probe that the mutated function differs from the original.

    Raises NoOpMutationError (distinct message and exit code) when the mutation
    is a no-op, i.e. it cannot be caught by any test because it changes nothing.
    """
    orig_out = orig_fn(*args, **kwargs)
    mutated_out = mutated_fn(*args, **kwargs)
    if _values_identical(orig_out, mutated_out):
        raise NoOpMutationError(f"NO-OP MUTATION: {name}")


def run_test_and_catch(test_fn, *args, **kwargs) -> bool:
    """Run a test function. Return True if it fails/raises (caught bug), False if it passes."""
    try:
        test_fn(*args, **kwargs)
        return False  # Test unexpectedly passed (bug missed)
    except AssertionError:
        return True  # Bug caught by assertion
    except Exception:
        return True  # Bug caught by exception


def test_bug_a() -> list[str]:
    """Bug (a): flip the sign of the normal."""
    orig_fn = gp_mod.fit_ground_plane

    def buggy_fit(*args, **kwargs):
        res = orig_fn(*args, **kwargs)
        return gp_mod.PlaneFit(
            n=-res.n,  # Flipped normal
            h_hat=res.h_hat,
            inlier_ratio=res.inlier_ratio,
            n_inliers=res.n_inliers,
            rms_residual=res.rms_residual,
            status=res.status,
        )

    caught_by = []
    try:
        gp_mod.fit_ground_plane = buggy_fit
        test_ground_plane.fit_ground_plane = buggy_fit
        if run_test_and_catch(test_ground_plane.test_t3_clean_plane_recovery, h=1.4, pitch_deg=10.0):
            caught_by.append("T3 (clean plane recovery)")
    finally:
        gp_mod.fit_ground_plane = orig_fn
        test_ground_plane.fit_ground_plane = orig_fn

    return caught_by


def test_bug_b() -> list[str]:
    """Bug (b): skip the scale correction in lift_frame."""
    orig_fn = lift_mod.lift_frame

    def buggy_lift(depth, floor_mask, other_masks, K, mount, estimator, rng=None, ransac_cfg=None):
        res = orig_fn(depth, floor_mask, other_masks, K, mount, estimator, rng, ransac_cfg)
        # Skip scale correction: points are unscaled!
        pts_c = camera_mod.backproject(depth, K)
        pts_g = frames_mod.camera_to_ground(pts_c, res.plane_fit.n, mount.h_known)
        return lift_mod.LiftResult(
            points_c={"floor": pts_c[floor_mask], **{k: pts_c[m] for k, m in other_masks.items()}},
            points_g={"floor": pts_g[floor_mask], **{k: pts_g[m] for k, m in other_masks.items()}},
            plane_fit=res.plane_fit,
            scale_result=scale_mod.ScaleResult(s=1.0, status=res.scale_result.status),
            scaled_depth=depth,
        )

    caught_by = []
    try:
        lift_mod.lift_frame = buggy_lift
        test_lift.lift_frame = buggy_lift
        if run_test_and_catch(test_lift.test_t5_scale_recovery, k=0.5):
            caught_by.append("T5 (scale recovery k=0.5)")
        if run_test_and_catch(test_lift.test_t5_scale_recovery, k=2.0):
            caught_by.append("T5 (scale recovery k=2.0)")
    finally:
        lift_mod.lift_frame = orig_fn
        test_lift.lift_frame = orig_fn

    return caught_by


def test_bug_c() -> list[str]:
    """Bug (c): use Euclidean range instead of z-depth in backproject."""
    orig_fn = camera_mod.backproject

    def buggy_backproject(depth, K):
        # Treat depth as Euclidean range r: z = r / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)
        depth_arr = np.asarray(depth, dtype=np.float64)
        h, w = depth_arr.shape
        fx, fy, cx, cy = camera_mod._extract_intrinsics(K)
        v_grid, u_grid = np.indices((h, w), dtype=np.float64)
        dx = (u_grid - cx) / fx
        dy = (v_grid - cy) / fy
        r_scale = np.sqrt(1.0 + dx**2 + dy**2)
        z_correct = depth_arr / r_scale  # Buggy: treats input as range
        x = dx * z_correct
        y = dy * z_correct
        return np.stack([x, y, z_correct], axis=-1)

    caught_by = []
    try:
        camera_mod.backproject = buggy_backproject
        test_camera.backproject = buggy_backproject
        if run_test_and_catch(test_camera.test_t1_project_backproject_roundtrip):
            caught_by.append("T1 (project/backproject roundtrip)")
    finally:
        camera_mod.backproject = orig_fn
        test_camera.backproject = orig_fn

    return caught_by


def test_bug_d() -> list[str]:
    """Bug (d): swap e_right and e_fwd in camera_to_ground."""
    orig_fn = frames_mod.camera_to_ground

    def buggy_camera_to_ground(points_c, n, h):
        pts = np.asarray(points_c, dtype=np.float64)
        e_right, e_fwd, e_up = frames_mod.ground_frame_basis(n)
        n_unit = -e_up
        # Swap right and fwd
        x_g = np.sum(pts * e_fwd, axis=-1)
        y_g = np.sum(pts * e_right, axis=-1)
        z_g = h - np.sum(pts * n_unit, axis=-1)
        return np.stack([x_g, y_g, z_g], axis=-1)

    caught_by = []
    try:
        frames_mod.camera_to_ground = buggy_camera_to_ground
        test_frames.camera_to_ground = buggy_camera_to_ground
        if run_test_and_catch(test_frames.test_t2_frames_floor_and_wall, pitch_deg=10.0):
            caught_by.append("T2 (frames wall Y recovery)")
    finally:
        frames_mod.camera_to_ground = orig_fn
        test_frames.camera_to_ground = orig_fn

    return caught_by


def test_bug_e() -> list[str]:
    """Bug (e): use the mean instead of the 10th percentile in object_anchor."""
    orig_fn = lift_mod.object_anchor

    def buggy_anchor(depth, mask, K, mount, plane=None, q=0.10, min_pixels=10):
        valid_mask = mask & np.isfinite(depth) & (depth > 0.0)
        pixel_count = int(np.sum(valid_mask))
        if pixel_count == 0:
            return lift_mod.Anchor(0.0, np.zeros(3), lift_mod.AnchorStatus.EMPTY_MASK)
        if pixel_count < min_pixels:
            return lift_mod.Anchor(0.0, np.zeros(3), lift_mod.AnchorStatus.TINY_MASK)

        # Buggy: mean instead of q-th percentile
        distance_m = float(np.mean(depth[valid_mask]))

        v_coords, u_coords = np.where(valid_mask)
        u_c = float(np.mean(u_coords))
        v_c = float(np.mean(v_coords))
        fx, fy, cx, cy = camera_mod._extract_intrinsics(K)
        x_c = (u_c - cx) * distance_m / fx
        y_c = (v_c - cy) * distance_m / fy
        z_c = distance_m
        p_c = np.array([x_c, y_c, z_c], dtype=np.float64)

        n = plane.n if plane is not None and plane.status == gp_mod.FitStatus.OK else np.array([0.0, 1.0, 0.0])
        position_g = frames_mod.camera_to_ground(p_c, n, mount.h_known)
        return lift_mod.Anchor(distance_m, position_g, lift_mod.AnchorStatus.OK)

    caught_by = []
    try:
        lift_mod.object_anchor = buggy_anchor
        test_anchor.object_anchor = buggy_anchor
        if run_test_and_catch(test_anchor.test_t8_anchor_percentile_vs_mean):
            caught_by.append("T8 (obstacle depth percentile vs mean)")
    finally:
        lift_mod.object_anchor = orig_fn
        test_anchor.object_anchor = orig_fn

    return caught_by


def test_bug_f() -> list[str]:
    """Bug (f): disable outlier rejection in the scale estimator."""
    orig_update = scale_mod.ScaleEstimator.update

    def buggy_update(self, h_hat, status=gp_mod.FitStatus.OK):
        status_str = status.value if isinstance(status, gp_mod.FitStatus) else str(status)
        if status_str != gp_mod.FitStatus.OK.value or h_hat <= 0.0:
            return scale_mod.ScaleResult(self.s, status_str)
        s_raw = self.mount.h_known / h_hat
        # Directly apply EMA without outlier check
        s_ema = (1.0 - self.cfg.alpha) * self.s + self.cfg.alpha * s_raw
        delta_max = self.cfg.max_step_change * self.s
        self.s = float(np.clip(s_ema, self.s - delta_max, self.s + delta_max))
        return scale_mod.ScaleResult(self.s, "OK")

    caught_by = []
    try:
        scale_mod.ScaleEstimator.update = buggy_update
        if run_test_and_catch(test_scale.test_t7_ema_behaviour):
            caught_by.append("T7 (EMA single outlier rejection)")
    finally:
        scale_mod.ScaleEstimator.update = orig_update

    return caught_by


# --- New negative controls for dataset/evaluation ---

def test_bug_g() -> list[str]:
    """Bug (g): splits allow the same route in two splits."""
    import tempfile
    from pathlib import Path
    from dataset.synth_dataset import build_synthetic_dataset
    from dataset.splits import assert_no_leakage

    caught_by = []
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = build_synthetic_dataset(
            out_dir=tmpdir, n_routes=5, frames_per_route=3, seed=42
        )
        routes = summary["routes"]
        # Create splits with duplicate route
        bad_splits = {
            "train": routes[:3],
            "val": [routes[3], routes[3]],  # Duplicate
            "test": [routes[4]],
        }
        try:
            assert_no_leakage(bad_splits, Path(tmpdir))
        except AssertionError:
            caught_by.append("assert_no_leakage (duplicate route)")
    return caught_by


def test_bug_h() -> list[str]:
    """Bug (h): AbsRel compares a Euclidean range (pred) against z-depth (GT).

    A constant conversion factor applied to both sides would cancel out, so the
    injected bug applies the range conversion to the prediction only, simulating
    mismatched units (pred as Euclidean range r = z * sqrt(1 + dx^2 + dy^2) at a
    representative off-axis pixel dx = dy = 0.5, GT as z-depth).
    """
    orig_fn = test_metrics_depth.compute_depth_metrics
    off_axis = float(np.sqrt(1.0 + 0.25 + 0.25))  # ~1.2247

    def buggy_compute_depth_metrics(gt_depth, pred_depth, valid_mask=None, tags=None):
        result = orig_fn(gt_depth, pred_depth, valid_mask, tags)
        gt = np.asarray(gt_depth, dtype=np.float64).ravel()
        pred = np.asarray(pred_depth, dtype=np.float64).ravel()
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool).ravel()
        else:
            valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
        gt, pred = gt[valid], pred[valid]
        if len(gt) == 0:
            return result
        result["AbsRel"] = float(np.mean(np.abs(gt - pred * off_axis) / gt))
        return result

    probe_gt = np.array([1.0, 2.0, 4.0])
    probe_pred = probe_gt * 1.1
    assert_mutation_effective(
        "(h) AbsRel range vs z-depth", orig_fn, buggy_compute_depth_metrics, probe_gt, probe_pred
    )

    caught_by = []
    try:
        test_metrics_depth.compute_depth_metrics = buggy_compute_depth_metrics
        if run_test_and_catch(test_metrics_depth.test_d6_absrel_exact):
            caught_by.append("D6 (AbsRel exact)")
    finally:
        test_metrics_depth.compute_depth_metrics = orig_fn
    return caught_by


def test_bug_i() -> list[str]:
    """Bug (i): coverage uses strict bounds (gt > lo and gt < hi)."""
    orig_fn = test_metrics_interval.compute_interval_metrics

    def buggy_compute_interval_metrics(
        gt, pred_lo, pred_hi, nominal_coverage=0.95, valid_mask=None, depths=None, tags=None
    ):
        result = orig_fn(gt, pred_lo, pred_hi, nominal_coverage, valid_mask, depths, tags)
        gt_arr = np.asarray(gt, dtype=np.float64).ravel()
        lo = np.asarray(pred_lo, dtype=np.float64).ravel()
        hi = np.asarray(pred_hi, dtype=np.float64).ravel()
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool).ravel()
        else:
            valid = (
                np.isfinite(gt_arr) & np.isfinite(lo) & np.isfinite(hi)
                & (gt_arr > 0) & (lo > 0) & (hi > 0)
            )
        gt_arr, lo, hi = gt_arr[valid], lo[valid], hi[valid]
        if len(gt_arr) == 0:
            return result
        coverage = float(np.mean((gt_arr > lo) & (gt_arr < hi)))  # BUG: strict bounds
        result["coverage"] = coverage
        result["coverage_gap"] = coverage - nominal_coverage
        return result

    probe_gt = np.array([1.0, 2.0, 3.0])
    probe_lo = np.array([1.0, 1.5, 4.0])
    probe_hi = np.array([1.5, 2.0, 4.0])
    assert_mutation_effective(
        "(i) coverage strict bounds",
        orig_fn,
        buggy_compute_interval_metrics,
        probe_gt,
        probe_lo,
        probe_hi,
    )

    caught_by = []
    try:
        test_metrics_interval.compute_interval_metrics = buggy_compute_interval_metrics
        if run_test_and_catch(test_metrics_interval.test_d7_coverage_width_score):
            caught_by.append("D7 (coverage width score)")
        if run_test_and_catch(test_metrics_interval.test_d7_inclusive_bounds):
            caught_by.append("D7 (inclusive bounds)")
    finally:
        test_metrics_interval.compute_interval_metrics = orig_fn
    return caught_by


def test_bug_j() -> list[str]:
    """Bug (j): range-bin edges off by one (right-inclusive bins).

    Exact edges (2.0, 4.0, 6.0) are assigned to the LOWER bin: (0,2], (2,4],
    (4,6], (6,inf) instead of the spec's [0,2), [2,4), [4,6), [6,inf).
    """
    orig_fn = test_metrics_depth.compute_depth_metrics

    def _bin_metrics(gt, pred):
        rel = np.abs(gt - pred) / gt
        return {
            "AbsRel": float(np.mean(rel)),
            "RMSE": float(np.sqrt(np.mean((gt - pred) ** 2))),
            "MAE": float(np.mean(np.abs(gt - pred))),
            "Bias": float(np.mean(pred - gt)),
            "Delta125": float(np.mean(np.maximum(gt / pred, pred / gt) < 1.25)),
            "count": int(len(gt)),
        }

    def buggy_compute_depth_metrics(gt_depth, pred_depth, valid_mask=None, tags=None):
        result = orig_fn(gt_depth, pred_depth, valid_mask, tags)
        gt = np.asarray(gt_depth, dtype=np.float64).ravel()
        pred = np.asarray(pred_depth, dtype=np.float64).ravel()
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool).ravel()
        else:
            valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
        gt, pred = gt[valid], pred[valid]
        per_bin = {}
        for i, name in enumerate(["0-2m", "2-4m", "4-6m", "6+m"]):
            if i < 3:
                mask = (gt > RANGE_BIN_EDGES[i]) & (gt <= RANGE_BIN_EDGES[i + 1])  # BUG
            else:
                mask = gt > RANGE_BIN_EDGES[3]  # BUG: drops the exact edge 6.0
            g, p = gt[mask], pred[mask]
            if len(g) > 0:
                per_bin[name] = _bin_metrics(g, p)
            else:
                per_bin[name] = {
                    "AbsRel": float("nan"),
                    "RMSE": float("nan"),
                    "MAE": float("nan"),
                    "Bias": float("nan"),
                    "Delta125": float("nan"),
                    "count": 0,
                }
        result["per_bin"] = per_bin
        return result

    probe = np.array([1.9, 2.0, 3.9, 4.0, 5.9, 6.0, 7.0])
    assert_mutation_effective(
        "(j) range-bin edges right-inclusive", orig_fn, buggy_compute_depth_metrics, probe, probe.copy()
    )

    caught_by = []
    try:
        test_metrics_depth.compute_depth_metrics = buggy_compute_depth_metrics
        if run_test_and_catch(test_metrics_depth.test_d6_per_bin_edges):
            caught_by.append("D6 (per bin edges)")
    finally:
        test_metrics_depth.compute_depth_metrics = orig_fn
    return caught_by


def test_bug_k() -> list[str]:
    """Bug (k): bootstrap resamples frames instead of routes."""
    from evaluation.bootstrap import bootstrap_routes
    import numpy as np

    caught_by = []
    # Create route metrics where route-level mean differs from frame-level mean
    route_metrics = {
        "route_0": np.array([1.0, 1.0, 1.0]),  # mean=1.0
        "route_1": np.array([10.0, 10.0]),     # mean=10.0
    }
    # True route-level mean = (1.0 + 10.0) / 2 = 5.5
    # Frame-level mean = (3*1 + 2*10) / 5 = 23/5 = 4.6
    def mean_fn(values):
        return float(np.mean(values))

    # Monkey-patch to resample frames
    orig_bootstrap = test_bootstrap.bootstrap_routes

    def buggy_bootstrap(route_metrics, metric_fn, n_bootstrap=100, seed=42, confidence=0.95):
        # Resample frames instead of routes
        all_frames = np.concatenate(list(route_metrics.values()))
        rng = np.random.default_rng(seed)
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(all_frames, size=len(all_frames), replace=True)
            bootstrap_samples.append(metric_fn(sampled))
        bootstrap_samples = np.array(bootstrap_samples)
        bootstrap_samples = bootstrap_samples[np.isfinite(bootstrap_samples)]
        alpha = 1.0 - confidence
        lower_pct = 100 * alpha / 2.0
        upper_pct = 100 * (1.0 - alpha / 2.0)
        return {
            "mean": float(np.mean(bootstrap_samples)),
            "ci_lower": float(np.percentile(bootstrap_samples, lower_pct)),
            "ci_upper": float(np.percentile(bootstrap_samples, upper_pct)),
            "samples": bootstrap_samples,
        }

    try:
        test_bootstrap.bootstrap_routes = buggy_bootstrap
        if run_test_and_catch(test_bootstrap.test_d9_ci_contains_true_mean):
            caught_by.append("D9 (CI contains true mean)")
        if run_test_and_catch(test_bootstrap.test_d9_route_vs_frame_resampling_ci_width):
            caught_by.append("D9 (route vs frame CI width)")
    finally:
        test_bootstrap.bootstrap_routes = orig_bootstrap
    return caught_by


def test_bug_l() -> list[str]:
    """Bug (l): interval score drops the penalty term."""
    orig_compute = test_metrics_interval.compute_interval_metrics

    def buggy_compute_interval_metrics(gt, pred_lo, pred_hi, nominal_coverage=0.95, valid_mask=None, depths=None, tags=None):
        # Buggy: interval score = mean width only (no penalty for outliers)
        gt_arr = np.asarray(gt, dtype=np.float64).ravel()
        lo = np.asarray(pred_lo, dtype=np.float64).ravel()
        hi = np.asarray(pred_hi, dtype=np.float64).ravel()
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool).ravel()
            gt_arr = gt_arr[valid]
            lo = lo[valid]
            hi = hi[valid]
        if depths is not None:
            depths = np.asarray(depths, dtype=np.float64).ravel()[valid]
        else:
            valid = np.isfinite(gt_arr) & np.isfinite(lo) & np.isfinite(hi) & (gt_arr > 0) & (lo > 0) & (hi > 0)
            gt_arr = gt_arr[valid]
            lo = lo[valid]
            hi = hi[valid]
            if depths is not None:
                depths = np.asarray(depths, dtype=np.float64).ravel()[valid]

        width = hi - lo
        interval_score = float(np.mean(width))  # No penalty term!

        # Still compute coverage correctly
        inside = (gt_arr >= lo) & (gt_arr <= hi)
        coverage = float(np.mean(inside))

        return {
            "coverage": coverage,
            "mean_width": float(np.mean(width)),
            "mean_rel_width": float(np.mean(width / gt_arr)),
            "coverage_gap": coverage - nominal_coverage,
            "interval_score": interval_score,
            "per_bin": {},
            "per_tag": {},
        }

    caught_by = []
    try:
        test_metrics_interval.compute_interval_metrics = buggy_compute_interval_metrics
        if run_test_and_catch(test_metrics_interval.test_d7_coverage_width_score):
            caught_by.append("D7 (coverage width score)")
    finally:
        test_metrics_interval.compute_interval_metrics = orig_compute
    return caught_by


def test_bug_m() -> list[str]:
    """Bug (m): false-walkable rate counts ignore pixels."""
    orig_compute = test_metrics_seg.compute_segmentation_metrics

    def buggy_compute_segmentation_metrics(pred_mask, gt_mask, depth=None, range_bin=None):
        # Buggy: don't exclude ignore (255) and unlabeled (0) from valid_gt
        # This would make false walkable rate wrong
        H, W = pred_mask.shape
        # Don't filter out ignore and unlabeled
        valid_gt = np.ones((H, W), dtype=bool)  # All pixels valid

        from core.config import MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER, MASK_UNLABELED, MASK_IGNORE

        pred_floor = (pred_mask == MASK_FLOOR) & valid_gt
        if pred_floor.sum() > 0:
            false_walkable = np.logical_and(
                pred_floor,
                (gt_mask == MASK_WALL) | (gt_mask == MASK_OBSTACLE) | (gt_mask == MASK_OTHER)
            ).sum()
            false_walkable_rate = float(false_walkable / pred_floor.sum())
        else:
            false_walkable_rate = float("nan")

        # Compute other metrics correctly for comparison
        result = orig_compute(pred_mask, gt_mask, depth, range_bin)
        result["false_walkable_rate"] = false_walkable_rate
        return result

    caught_by = []
    try:
        test_metrics_seg.compute_segmentation_metrics = buggy_compute_segmentation_metrics
        if run_test_and_catch(test_metrics_seg.test_d5_false_walkable_rate):
            caught_by.append("D5 (false walkable rate)")
        if run_test_and_catch(test_metrics_seg.test_d5_fw_rate_unchanged_by_added_ignore_pixels):
            caught_by.append("D5 (FW unchanged by added 0/255)")
        if run_test_and_catch(test_metrics_seg.test_d5_fw_rate_denominator_excludes_ignore_would_change):
            caught_by.append("D5 (FW denominator excludes 0/255)")
    finally:
        test_metrics_seg.compute_segmentation_metrics = orig_compute
    return caught_by


def test_bug_n() -> list[str]:
    """Bug (n): range_to_depth returns range unchanged."""
    import geometry.camera as camera_mod
    import tests.test_camera as test_camera

    orig_fn = camera_mod.range_to_depth

    def buggy_range_to_depth(range_m, u, v, K):
        # Buggy: return range unchanged (no conversion)
        return np.asarray(range_m, dtype=np.float64)

    caught_by = []
    try:
        camera_mod.range_to_depth = buggy_range_to_depth
        test_camera.range_to_depth = buggy_range_to_depth
        if run_test_and_catch(test_camera.test_range_depth_equal_at_principal_point):
            caught_by.append("D2 (range=depth at principal point)")
        if run_test_and_catch(test_camera.test_range_depth_off_axis_known_value):
            caught_by.append("D2 (off-axis known value)")
    finally:
        camera_mod.range_to_depth = orig_fn
        test_camera.range_to_depth = orig_fn
    return caught_by


SOURCE_CONTROLS: list[dict[str, Any]] = [
    {
        "id": "P(a)",
        "desc": "Class map sends wall to floor",
        "file": "perception/class_map.py",
        "old": "        mapping[class_id] = targets.pop()",
        "new": (
            "        mapping[class_id] = (MASK_FLOOR if targets == {MASK_WALL} "
            "else targets.pop())"
        ),
        "probe": (
            "from perception.class_map import build_class_map; "
            "print(build_class_map({0: 'wall'}))"
        ),
        "test": "tests/test_class_map.py::test_p2_default_rules_map_exact",
    },
    {
        "id": "P(b)",
        "desc": "Unknown names default to floor",
        "file": "perception/class_map.py",
        "old": (
            "            else:\n"
            "                # Unknown names are deliberately other, never floor.\n"
            "                targets.add(MASK_OTHER)"
        ),
        "new": (
            "            else:\n"
            "                # Injected bug: unknown names default to floor.\n"
            "                targets.add(MASK_FLOOR)"
        ),
        "probe": (
            "from perception.class_map import build_class_map; "
            "print(build_class_map({0: 'banana'}))"
        ),
        "test": "tests/test_class_map.py::test_p2_unknown_names_never_map_to_floor",
    },
    {
        "id": "P(c)",
        "desc": "Softmax without max subtraction",
        "file": "perception/postprocess.py",
        "old": "    shifted = values - safe_max",
        "new": "    shifted = values",
        "probe": (
            "import numpy as np; from perception.postprocess import stable_softmax; "
            "p = stable_softmax(np.array([[10000.0, 5000.0, -10000.0]])); "
            "print(np.isfinite(p).all(), p)"
        ),
        "test": "tests/test_postprocess.py::test_p3_softmax_rows_sum_to_one_and_stable",
    },
    {
        "id": "P(d)",
        "desc": "Intrinsics not scaled after resize",
        "file": "geometry/camera.py",
        "old": "    return Intrinsics(\n        fx=fx * sx,\n        fy=fy * sy,",
        "new": "    return Intrinsics(\n        fx=fx,\n        fy=fy,",
        "probe": (
            "from core.config import Intrinsics; from geometry.camera import scale_intrinsics; "
            "k = scale_intrinsics(Intrinsics(1000.0, 900.0, 640.0, 360.0, 1280, 720), 0.5, 0.5); "
            "print(k.fx, k.fy, k.cx, k.cy)"
        ),
        "test": "tests/test_postprocess.py::test_p4_scale_intrinsics_and_letterbox_hand_computed",
    },
    {
        "id": "P(e)",
        "desc": "Letterbox pad offset not applied",
        "file": "geometry/camera.py",
        "old": "        cx=cx * sx + pad_x,",
        "new": "        cx=cx * sx,",
        "probe": (
            "from core.config import Intrinsics; from geometry.camera import scale_intrinsics; "
            "k = scale_intrinsics(Intrinsics(500.0, 500.0, 200.0, 400.0, 400, 800), 0.8, 0.8, 160, 0); "
            "print(k.cx, k.cy)"
        ),
        "test": "tests/test_postprocess.py::test_p4_scale_intrinsics_and_letterbox_hand_computed",
    },
    {
        "id": "P(f)",
        "desc": "sanitize_depth leaves sky pixels valid",
        "file": "perception/postprocess.py",
        "old": "        invalid |= sky",
        "new": "        invalid |= np.zeros_like(invalid, dtype=bool)",
        "probe": (
            "import numpy as np; from perception.postprocess import sanitize_depth; "
            "d = np.ones((2, 2), dtype=np.float32); s = np.zeros((2, 2), dtype=bool); "
            "s[0, 0] = True; print(sanitize_depth(d, 10.0, s))"
        ),
        "test": "tests/test_postprocess.py::test_p5_sanitize_depth_all_invalid_categories",
    },
    {
        "id": "P(g)",
        "desc": "Pipeline maps depth back with wrong axis order",
        "file": "perception/pipeline.py",
        "old": '    return map_back(arr, info, kind="depth")',
        "new": '    return map_back(arr.T, info, kind="depth")',
        "probe": (
            "import numpy as np; from perception.pipeline import align_depth_to_image; "
            "print(align_depth_to_image(np.ones((120, 160), dtype=np.float32), (240, 320)).shape)"
        ),
        "test": "tests/test_pipeline.py::test_p7_model_resolution_depth_is_mapped_back",
    },
    {
        "id": "P(h)",
        "desc": "Initial scale fixed at initial_scale (1.0)",
        "file": "geometry/scale.py",
        "old": "        if not self._initialized:\n            self.s = float(s_raw)",
        "new": "        if not self._initialized:\n            self.s = float(self.cfg.initial_scale)",
        "probe": (
            "from core.config import MountConfig; from geometry.scale import ScaleEstimator; "
            "print(ScaleEstimator(MountConfig()).update(0.7).s)"
        ),
        "test": "tests/test_scale.py::test_first_valid_frame_initialises_scale_directly",
    },
]


def _copy_repo(destination: Path) -> None:
    """Copy the working tree without VCS, virtualenv, media or secret files."""
    ignore = shutil.ignore_patterns(
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".early.coverage",
        ".coverage*",
        ".env",
        "sample-vid",
        "logs",
        "database",
        "*.pt",
        "*.pth",
        "*.mp4",
    )
    shutil.copytree(REPO_ROOT, destination, ignore=ignore)


def _probe(workdir: Path, code: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workdir)
    env.pop("PYTHONHOME", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout.strip()


def _pytest(workdir: Path, nodes: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workdir)
    env.pop("PYTHONHOME", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", *nodes, "-q", "--tb=no", "-p", "no:cacheprovider"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        env=env,
    )


def run_source_level_controls() -> tuple[bool, list[tuple[str, str, str, str]]]:
    """Apply source mutations to isolated copies; assert each is effective and caught."""
    rows: list[tuple[str, str, str, str]] = []
    all_caught = True
    nodes = sorted({control["test"] for control in SOURCE_CONTROLS})

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base = root / "base"
        _copy_repo(base)

        pre = _pytest(base, nodes)
        if pre.returncode != 0:
            print("ERROR: unpatched isolated copy does not pass the selected tests.")
            print(pre.stdout)
            print(pre.stderr)
            return False, [("PRE-RUN", "isolated copy", "n/a", "ERROR")]

        for index, control in enumerate(SOURCE_CONTROLS):
            work = root / f"control_{index}"
            shutil.copytree(
                base,
                work,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
            )
            path = work / control["file"]
            original_text = path.read_text(encoding="utf-8")
            if control["old"] not in original_text:
                raise RuntimeError(
                    f"source control {control['id']}: patch target not found in {control['file']}"
                )
            patched_text = original_text.replace(control["old"], control["new"], 1)
            if patched_text == original_text:
                raise NoOpMutationError(f"NO-OP MUTATION: {control['id']} {control['desc']}")
            path.write_text(patched_text, encoding="utf-8")

            original_probe = _probe(REPO_ROOT, control["probe"])
            mutated_probe = _probe(work, control["probe"])
            if original_probe == mutated_probe:
                raise NoOpMutationError(f"NO-OP MUTATION: {control['id']} {control['desc']}")

            result = _pytest(work, [control["test"]])
            caught = result.returncode != 0
            if not caught:
                all_caught = False
            rows.append(
                (
                    control["id"],
                    control["desc"],
                    "YES" if original_probe != mutated_probe else "NO",
                    "YES" if caught else "NO",
                )
            )
    return all_caught, rows


def main() -> int:
    """Run all negative controls and print verification tables."""
    print("=" * 80)
    print("NEGATIVE CONTROLS VERIFICATION (GEOMETRY MODULE + DATASET/EVAL)")
    print("=" * 80)

    controls = [
        ("(a) Flip normal sign in fit_ground_plane", test_bug_a),
        ("(b) Skip scale correction in lift_frame", test_bug_b),
        ("(c) Euclidean range instead of z-depth", test_bug_c),
        ("(d) Swap e_right and e_fwd in camera_to_ground", test_bug_d),
        ("(e) Mean instead of 10th percentile in anchor", test_bug_e),
        ("(f) Disable outlier rejection in ScaleEstimator", test_bug_f),
        ("(g) Splits allow same route in two splits", test_bug_g),
        ("(h) AbsRel uses Euclidean range instead of z-depth", test_bug_h),
        ("(i) Coverage uses strict instead of inclusive bounds", test_bug_i),
        ("(j) Range-bin edges off by one (right-inclusive)", test_bug_j),
        ("(k) Bootstrap resamples frames instead of routes", test_bug_k),
        ("(l) Interval score drops the penalty term", test_bug_l),
        ("(m) False-walkable rate counts ignore pixels", test_bug_m),
        ("(n) range_to_depth returns range unchanged", test_bug_n),
    ]

    all_passed = True
    print(f"{'Bug Description':<55} | {'Caught By':<25} | {'Status':<6}")
    print("-" * 90)

    for desc, runner in controls:
        try:
            caught_tests = runner()
        except NoOpMutationError as exc:
            print(f"{desc:<55} | {str(exc):<25} | NO-OP ")
            print("=" * 90)
            print(str(exc))
            return 3
        is_caught = len(caught_tests) > 0
        status_str = "YES" if is_caught else "NO"
        tests_str = ", ".join(caught_tests) if is_caught else "NONE"
        print(f"{desc:<55} | {tests_str:<25} | {status_str:<6}")
        if not is_caught:
            all_passed = False

    print("=" * 90)

    print("\nSOURCE-LEVEL PERCEPTION MUTATIONS (isolated temporary package copies)")
    print("=" * 90)
    try:
        source_passed, source_rows = run_source_level_controls()
    except NoOpMutationError as exc:
        print(str(exc))
        return 3
    print(f"{'Control':<7} | {'Mutation':<48} | {'Probe':<5} | {'Caught':<6} | Status")
    print("-" * 90)
    for control_id, desc, probe_changed, caught in source_rows:
        status_str = "YES" if probe_changed == "YES" and caught == "YES" else "NO"
        print(f"{control_id:<7} | {desc:<48} | {probe_changed:<5} | {caught:<6} | {status_str}")
    print("=" * 90)

    total_bugs = len(controls)
    total_source = len(SOURCE_CONTROLS)
    if all_passed and source_passed:
        print(
            f"ALL {total_bugs} ORIGINAL BUGS AND {total_source} SOURCE-LEVEL "
            "PERCEPTION MUTATIONS CAUGHT."
        )
        return 0
    else:
        print("FAIL: One or more bugs were not caught!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
