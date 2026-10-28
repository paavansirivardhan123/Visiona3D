"""Offline tests for the estimated session trail (floor-plane optical flow).

No models, camera, video or network is used. Synthetic floor projections
exercise the motion fit, a warped synthetic texture exercises tracking, and the
trail state machine is driven by stubbed steps.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import live.trace as trace_module
from core.config import Intrinsics
from live.trace import (
    FloorPlane,
    SessionTrace,
    estimate_floor_step,
    fit_floor_motion,
    track_floor_features,
)
from mapping.rolling import Odometry2D

K = Intrinsics(fx=300.0, fy=300.0, cx=320.0, cy=180.0, width=640, height=360)
PLANE = FloorPlane(normal=np.array([0.0, 1.0, 0.0]), height_m=1.4)


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _floor_points(count: int = 400, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(-2.0, 2.0, count),
            np.full(count, PLANE.height_m),
            rng.uniform(1.5, 10.0, count),
        ]
    )


def _project(points: np.ndarray) -> np.ndarray:
    cam = (points @ _camera_matrix().T)
    uv = cam[:, :2] / cam[:, 2:3]
    return uv


def _correspondences(dz: float, yaw_deg: float, seed: int = 7, dx: float = 0.0):
    world = _floor_points(seed=seed)
    src = _project(world)
    moved = (world - np.array([dx, 0.0, dz])) @ _rot_y(yaw_deg)
    dst = _project(moved)
    inside = (
        (src[:, 0] > 0)
        & (src[:, 0] < K.width)
        & (src[:, 1] > 0)
        & (src[:, 1] < K.height)
        & (dst[:, 0] > 0)
        & (dst[:, 0] < K.width)
        & (dst[:, 1] > 0)
        & (dst[:, 1] < K.height)
    )
    rng = np.random.default_rng(seed + 1)
    count = int(inside.sum())
    return (
        src[inside] + rng.normal(0.0, 0.2, (count, 2)),
        dst[inside] + rng.normal(0.0, 0.2, (count, 2)),
        count,
    )


def _texture(height: int = 360, width: int = 640, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 1.2)


def _warp_floor(image: np.ndarray, dz: float, yaw_deg: float, dx: float = 0.0) -> np.ndarray:
    """Warp an image of the floor by the ground-truth step."""
    Km = _camera_matrix()
    n = PLANE.normal.reshape(3, 1)
    C = np.array([[dx], [0.0], [dz]])
    H = _rot_y(-yaw_deg) @ (np.eye(3) - C @ n.T / PLANE.height_m)
    return cv2.warpPerspective(image, Km @ H @ np.linalg.inv(Km), (image.shape[1], image.shape[0]))


def _plane_correspondences(plane, dz: float, yaw_deg: float, dx: float = 0.0, seed: int = 11):
    """Correspondences for a floor plane with an arbitrary (tilted) normal."""
    Km = _camera_matrix()
    rng = np.random.default_rng(seed)
    src = np.column_stack(
        [rng.uniform(0.0, K.width, 600), rng.uniform(0.55 * K.height, K.height, 600)]
    )
    C = np.array([dx, 0.0, dz])
    H = _rot_y(-yaw_deg) @ (np.eye(3) - np.outer(C, plane.normal) / plane.height_m)
    M = Km @ H @ np.linalg.inv(Km)
    q = np.column_stack([src, np.ones(len(src))]) @ M.T
    ok = q[:, 2] > 1e-6
    dst = q[ok, :2] / q[ok, 2:3]
    src = src[ok]
    inside = (
        (dst[:, 0] > 0.0)
        & (dst[:, 0] < K.width)
        & (dst[:, 1] > 0.0)
        & (dst[:, 1] < K.height)
    )
    src, dst = src[inside], dst[inside]
    noise = np.random.default_rng(seed + 1)
    count = len(src)
    return (
        src + noise.normal(0.0, 0.2, (count, 2)),
        dst + noise.normal(0.0, 0.2, (count, 2)),
    )


class TestFitFloorMotion:
    def test_forward_step_is_metric(self):
        src, dst, _ = _correspondences(0.8, 0.0)
        step = fit_floor_motion(src, dst, PLANE, K)
        assert step is not None
        assert abs(float(step.dy) - 0.8) <= 0.06
        assert abs(float(step.dx)) <= 0.02

    def test_clockwise_turn_is_positive_yaw(self):
        src, dst, _ = _correspondences(0.4, 5.0)
        step = fit_floor_motion(src, dst, PLANE, K)
        assert step is not None
        assert abs(math.degrees(step.get_yaw_rad()) - 5.0) <= 0.3
        assert abs(float(step.dy) - 0.4) <= 0.06

    def test_left_turn_is_negative_yaw(self):
        src, dst, _ = _correspondences(0.3, -6.0)
        step = fit_floor_motion(src, dst, PLANE, K)
        assert step is not None
        assert abs(math.degrees(step.get_yaw_rad()) + 6.0) <= 0.3

    def test_right_step_is_metric(self):
        src, dst, _ = _correspondences(0.3, 0.0, dx=0.4)
        step = fit_floor_motion(src, dst, PLANE, K)
        assert step is not None
        assert abs(float(step.dx) - 0.4) <= 0.06
        assert abs(float(step.dy) - 0.3) <= 0.06

    def test_lateral_step_is_not_mistaken_for_a_turn(self):
        src, dst, _ = _correspondences(0.0, 0.0, dx=0.5)
        step = fit_floor_motion(src, dst, PLANE, K)
        assert step is not None
        assert abs(float(step.dx) - 0.5) <= 0.06
        assert abs(float(step.dy)) <= 0.06
        assert abs(math.degrees(step.get_yaw_rad())) <= 0.5

    def test_step_with_lateral_motion_and_turn(self):
        src, dst, _ = _correspondences(0.5, -4.0, dx=-0.25)
        step = fit_floor_motion(src, dst, PLANE, K)
        assert step is not None
        assert abs(float(step.dx) + 0.25) <= 0.06
        assert abs(float(step.dy) - 0.5) <= 0.06
        assert abs(math.degrees(step.get_yaw_rad()) + 4.0) <= 0.4

    def test_tilted_floor_normal_is_handled(self):
        normal = np.array([0.15, 0.97, -0.19])
        plane = FloorPlane(normal / np.linalg.norm(normal), 1.3)
        src, dst = _plane_correspondences(plane, 0.4, 2.0, dx=0.25)
        step = fit_floor_motion(src, dst, plane, K)
        assert step is not None
        assert abs(float(step.dx) - 0.25) <= 0.06
        assert abs(float(step.dy) - 0.4) <= 0.06
        assert abs(math.degrees(step.get_yaw_rad()) - 2.0) <= 0.4

    def test_outliers_are_absorbed_by_the_median(self):
        src, dst, _ = _correspondences(0.6, 2.0)
        rng = np.random.default_rng(3)
        bad = rng.choice(len(src), size=len(src) // 3, replace=False)
        dst = dst.copy()
        dst[bad] += rng.normal(0.0, 40.0, (len(bad), 2))
        step = fit_floor_motion(src, dst, PLANE, K)
        assert step is not None
        assert abs(float(step.dy) - 0.6) <= 0.1
        assert abs(math.degrees(step.get_yaw_rad()) - 2.0) <= 0.6

    def test_too_few_points_are_rejected(self):
        src, dst, _ = _correspondences(0.5, 0.0)
        assert fit_floor_motion(src[:10], dst[:10], PLANE, K) is None

    def test_step_beyond_search_is_rejected(self):
        src, dst, _ = _correspondences(5.0, 0.0)
        assert fit_floor_motion(src, dst, PLANE, K) is None

    def test_implausible_plane_is_rejected(self):
        src, dst, _ = _correspondences(0.5, 0.0)
        assert fit_floor_motion(src, dst, FloorPlane(np.array([0.0, 1.0, 0.0]), 0.0), K) is None
        assert fit_floor_motion(src, dst, FloorPlane(np.array([0.0, 2.0, 0.0]), 1.4), K) is None
        assert (
            fit_floor_motion(src, dst, FloorPlane(np.array([0.0, np.nan, 0.0]), 1.4), K)
            is None
        )
        assert fit_floor_motion(src, dst, FloorPlane(np.array([0.0, 1.0, 0.0]), 9.0), K) is None

    def test_non_finite_correspondence_is_rejected(self):
        src, dst, _ = _correspondences(0.5, 0.0)
        dst = dst.copy()
        dst[0, 0] = np.nan
        assert fit_floor_motion(src, dst, PLANE, K) is None

    def test_shape_mismatch_is_rejected(self):
        src, dst, _ = _correspondences(0.5, 0.0)
        assert fit_floor_motion(src, dst[:-1], PLANE, K) is None


class TestTrackFloorFeatures:
    def test_blank_frames_are_rejected(self):
        gray = np.zeros((180, 320), dtype=np.uint8)
        mask = np.ones_like(gray, dtype=bool)
        assert track_floor_features(gray, gray, mask) is None

    def test_shape_mismatch_is_rejected(self):
        gray = _texture(180, 320)
        assert track_floor_features(gray, gray[:90, :160], np.ones_like(gray)) is None

    def test_empty_floor_mask_is_rejected(self):
        gray = _texture(180, 320)
        assert track_floor_features(gray, gray, np.zeros_like(gray, dtype=bool)) is None

    def test_zero_motion_is_not_a_step(self):
        gray = _texture()
        floor = np.zeros_like(gray, dtype=bool)
        floor[200:, :] = True
        tracked = track_floor_features(gray, gray, floor)
        assert tracked is not None
        src, dst = tracked
        assert np.abs(src - dst).max() <= 0.5
        assert estimate_floor_step(gray, gray, floor, PLANE, K) is None

    def test_clear_motion_beats_the_standing_still_model(self):
        # A larger synthetic step is beyond what LK resolves on this texture,
        # so the acceptance gate is checked on motion the tracker can follow.
        gray = _texture()
        warped = _warp_floor(gray, 0.5, 0.0)
        floor = np.zeros_like(gray, dtype=bool)
        floor[200:, :] = True
        assert estimate_floor_step(gray, warped, floor, PLANE, K) is not None

    def test_warped_floor_recovers_the_step(self):
        gray = _texture()
        warped = _warp_floor(gray, 0.5, 3.0)
        floor = np.zeros_like(gray, dtype=bool)
        floor[200:, :] = True
        step = estimate_floor_step(gray, warped, floor, PLANE, K)
        assert step is not None
        assert abs(float(step.dy) - 0.5) <= 0.08
        assert abs(math.degrees(step.get_yaw_rad()) - 3.0) <= 0.4


class TestSessionTrace:
    def _trace(self) -> SessionTrace:
        return SessionTrace()

    def test_first_frame_does_not_move(self):
        trace = self._trace()
        gray = _texture(180, 320)
        floor = np.ones_like(gray, dtype=bool)
        assert trace.update_frame(gray, PLANE, floor, K) is None
        assert trace.points == [(0.0, 0.0)]
        assert trace.steps == 0

    def test_forward_steps_accumulate(self, monkeypatch):
        steps = iter([Odometry2D(0.0, 1.0), Odometry2D(0.0, 1.0)])
        monkeypatch.setattr(
            trace_module, "estimate_floor_step", lambda *a, **k: next(steps)
        )
        trace = self._trace()
        gray = _texture(180, 320)
        floor = np.ones_like(gray, dtype=bool)
        trace.update_frame(gray, PLANE, floor, K)
        trace.update_frame(gray, PLANE, floor, K)
        trace.update_frame(gray, PLANE, floor, K)
        assert trace.steps == 2
        assert abs(trace.y - 2.0) <= 1e-9
        assert abs(trace.x) <= 1e-9
        assert abs(trace.distance_m - 2.0) <= 1e-9
        assert len(trace.points) == 3

    def test_turn_then_forward_follows_new_heading(self, monkeypatch):
        steps = iter([Odometry2D(0.0, 0.0, dyaw_deg=90.0), Odometry2D(0.0, 1.0)])
        monkeypatch.setattr(
            trace_module, "estimate_floor_step", lambda *a, **k: next(steps)
        )
        trace = self._trace()
        gray = _texture(180, 320)
        floor = np.ones_like(gray, dtype=bool)
        trace.update_frame(gray, PLANE, floor, K)
        trace.update_frame(gray, PLANE, floor, K)
        trace.update_frame(gray, PLANE, floor, K)
        assert abs(trace.heading_deg - 90.0) <= 1e-9
        assert abs(trace.x - 1.0) <= 1e-9
        assert abs(trace.y) <= 1e-9

    def test_lateral_step_moves_along_x(self, monkeypatch):
        steps = iter([Odometry2D(dx=0.5, dy=0.0)])
        monkeypatch.setattr(
            trace_module, "estimate_floor_step", lambda *a, **k: next(steps)
        )
        trace = self._trace()
        gray = _texture(180, 320)
        floor = np.ones_like(gray, dtype=bool)
        trace.update_frame(gray, PLANE, floor, K)
        trace.update_frame(gray, PLANE, floor, K)
        assert abs(trace.x - 0.5) <= 1e-9
        assert abs(trace.y) <= 1e-9
        assert abs(trace.distance_m - 0.5) <= 1e-9
        assert abs(trace.heading_deg) <= 1e-9

    def test_missing_plane_leaves_a_gap(self):
        trace = self._trace()
        gray = _texture(180, 320)
        floor = np.ones_like(gray, dtype=bool)
        trace.update_frame(gray, PLANE, floor, K)
        assert trace.update_frame(gray, None, None, K) is None
        assert trace.misses == 1
        assert trace.steps == 0
        assert len(trace.points) == 1

    def test_three_missing_plane_frames_reset_reference(self):
        trace = self._trace()
        gray = _texture(180, 320)
        floor = np.ones_like(gray, dtype=bool)
        trace.update_frame(gray, PLANE, floor, K)
        for _ in range(2):
            assert trace.update_frame(gray, None, None, K) is None
        assert trace.update_frame(gray, None, None, K) is None
        assert trace._prev_gray is None
        assert trace._prev_plane is None
        assert trace.misses == 0

    def test_long_time_gap_is_not_integrated(self):
        trace = self._trace()
        gray = _texture(180, 320)
        floor = np.ones_like(gray, dtype=bool)
        trace.update_frame(gray, PLANE, floor, K, ts=0.0)
        assert trace.update_frame(gray, PLANE, floor, K, ts=10.0) is None
        assert trace.misses == 1
        assert trace._prev_ts == 10.0

    def test_reset_reference_forgets_previous_frame(self):
        trace = self._trace()
        gray = _texture(180, 320)
        floor = np.ones_like(gray, dtype=bool)
        trace.update_frame(gray, PLANE, floor, K)
        trace.reset_reference()
        assert trace._prev_gray is None
        assert trace.update_frame(gray, PLANE, floor, K) is None
