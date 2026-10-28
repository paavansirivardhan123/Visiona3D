"""Offline tests for the standalone session map image.

No models, camera, video or network is used. Rendering is exercised on a
synthetic grid and the saved PNG is read back.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from live.anchors import ObjectAnchor
from live.map_image import (
    CAMERA_BGR,
    OBJECT_PALETTE_BGR,
    TRAIL_BGR,
    WARN_BGR,
    MapMeta,
    MapObject,
    _draw_objects,
    object_colors,
    render_session_map,
    save_session_map,
)
from live.runner import LiveRunner
from live.trace import SessionTrace
from mapping.grid import CELL_OCCUPIED, CELL_UNKNOWN, MapConfig

CFG = MapConfig()


def _grid() -> np.ndarray:
    grid = np.full(CFG.shape, CELL_UNKNOWN, dtype=np.uint8)
    grid[20:23, 28:32] = CELL_OCCUPIED
    grid[40:43, 18:22] = CELL_OCCUPIED
    return grid


def _trail() -> list[tuple[float, float]]:
    return [(0.0, 0.0), (0.0, 1.0), (0.2, 2.0), (0.6, 3.0)]


def _pixels(image: np.ndarray, colour: tuple[int, int, int]) -> int:
    return int(np.count_nonzero(np.all(image == colour, axis=2)))


def _near_pixels(image: np.ndarray, colour: tuple[int, int, int], tol: int = 60) -> int:
    """Count pixels close to a colour; small text is anti-aliased, not exact."""
    diff = np.abs(image.astype(np.int16) - np.array(colour, dtype=np.int16))
    return int(np.count_nonzero(np.all(diff <= tol, axis=2)))


def _anchor(
    track_id: int,
    class_name: str,
    lateral: float,
    forward: float,
    status: str = "OK",
) -> ObjectAnchor:
    return ObjectAnchor(
        track_id=track_id,
        class_name=class_name,
        conf=0.9,
        box_xyxy=(0.0, 0.0, 10.0, 10.0),
        distance_m=math.hypot(lateral, forward),
        position_g=(lateral, forward, 0.0),
        status=status,
    )


class TestRenderSessionMap:
    def test_shape_and_trail_pixels(self):
        image = render_session_map(
            _trail(),
            _grid(),
            CFG,
            heading_deg=10.0,
            meta=MapMeta(
                session_id="sess",
                frame_index=7,
                decision="GUIDE",
                heading_deg=10.0,
                nearest_m=1.2,
                uncalibrated=True,
                distance_m=3.0,
                steps=3,
                misses=0,
            ),
        )
        assert image.shape == (920, 900, 3)
        trail_pixels = int(np.count_nonzero(np.all(image == TRAIL_BGR, axis=2)))
        assert trail_pixels > 50

    def test_single_point_without_grid(self):
        image = render_session_map([(0.0, 0.0)], None, CFG)
        assert image.shape == (920, 900, 3)
        assert int(np.count_nonzero(np.any(image != (26, 26, 26), axis=2))) > 0

    def test_long_trail_stays_inside_the_image(self):
        trail = [(0.0, float(y)) for y in range(0, 60)]
        image = render_session_map(trail, _grid(), CFG, heading_deg=0.0)
        assert image.shape == (920, 900, 3)

    def test_save_writes_a_readable_png(self, tmp_path):
        out = save_session_map(
            tmp_path / "map.png",
            render_session_map(_trail(), _grid(), CFG, heading_deg=5.0),
        )
        assert out.exists()
        assert out.stat().st_size > 1000
        loaded = cv2.imread(str(out))
        assert loaded is not None
        assert loaded.shape == (920, 900, 3)

    def test_save_adds_png_suffix_when_missing(self, tmp_path):
        out = save_session_map(
            tmp_path / "map", render_session_map(_trail(), _grid(), CFG)
        )
        assert out.name == "map.png"
        assert out.exists()

    def test_map_names_start_and_now_and_flags_a_frozen_trail(self, monkeypatch):
        import live.map_image as map_image_module

        texts: list[str] = []
        original = cv2.putText

        def spy(img, text, *args, **kwargs):
            texts.append(str(text))
            return original(img, text, *args, **kwargs)

        monkeypatch.setattr(map_image_module.cv2, "putText", spy)
        render_session_map(
            [(0.0, 0.0)], None, CFG, meta=MapMeta(frame_index=4, steps=0, gaps=6)
        )
        assert any("START" in text and "NOW" in text for text in texts)
        assert any(text.startswith("NO STEP ACCEPTED YET") for text in texts)

        texts.clear()
        render_session_map(
            _trail(), None, CFG, meta=MapMeta(frame_index=4, steps=3, gaps=6)
        )
        assert "START" in texts
        assert "NOW" in texts
        assert not any(text.startswith("NO STEP ACCEPTED YET") for text in texts)

    def test_frozen_warning_is_visible_in_pixels(self):
        frozen = render_session_map(
            [(0.0, 0.0)], None, CFG, meta=MapMeta(steps=0, gaps=6)
        )
        moving = render_session_map(
            _trail(), None, CFG, meta=MapMeta(steps=3, gaps=6)
        )
        assert _near_pixels(frozen, WARN_BGR) > _near_pixels(moving, WARN_BGR) + 50
        assert _pixels(frozen, CAMERA_BGR) > 20


class TestMapObjects:
    def test_colour_is_fixed_per_class(self):
        first = object_colors(["chair", "person", "chair"])
        assert set(first) == {"chair", "person"}
        assert first["chair"] != first["person"]
        assert object_colors(["person", "chair"]) == first

    def test_classes_get_distinct_colours(self):
        names = [f"class_{i:02d}" for i in range(12)]
        assert len(set(object_colors(names).values())) == 12

    def test_palette_wraps_after_twelve_classes(self):
        names = [f"class_{i:02d}" for i in range(13)]
        colours = object_colors(names)
        assert colours["class_00"] == colours["class_12"]

    def test_colour_index_follows_sorted_class_order(self):
        colours = object_colors(["banana", "apple"])
        assert colours["apple"] == OBJECT_PALETTE_BGR[0]
        assert colours["banana"] == OBJECT_PALETTE_BGR[1]

    def test_colour_assignment_is_stable_across_processes(self):
        # PYTHONHASHSEED changes set iteration order between runs, so the
        # mapping must come from sorting, not from set order.
        code = (
            "import json;"
            "from live.map_image import object_colors;"
            "print(json.dumps(object_colors("
            "['chair', 'person', 'lamp', 'door', 'table', 'sign'])))"
        )
        root = Path(__file__).resolve().parents[1]
        outputs = []
        for seed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(root),
            )
            assert proc.returncode == 0, proc.stderr
            outputs.append(proc.stdout.strip())
        assert outputs[0] == outputs[1]

    def test_same_class_shares_one_colour(self):
        objects = [
            MapObject("chair", 0.0, 2.0, 1),
            MapObject("chair", 2.0, 2.0, 2),
            MapObject("person", -2.0, 3.0, 3),
        ]
        image = render_session_map(_trail(), None, CFG, objects=objects)
        colours = object_colors([obj.class_name for obj in objects])
        chairs = _pixels(image, colours["chair"])
        persons = _pixels(image, colours["person"])
        assert persons > 20
        assert chairs > persons

    def test_marker_follows_the_world_position(self):
        colour = object_colors(["chair"])["chair"]

        def centroid(image):
            ys, xs = np.nonzero(np.all(image == colour, axis=2))
            assert xs.size > 0
            return float(xs.mean()), float(ys.mean())

        # A fixed extent keeps the auto-centred view identical between renders,
        # so pixel shifts measure the world-space placement only.
        frame = [(0.0, 0.0), (0.0, 6.0), (3.0, 0.0)]
        near = render_session_map(
            frame, None, CFG, objects=[MapObject("chair", 0.0, 2.0)], show_legend=False
        )
        far = render_session_map(
            frame, None, CFG, objects=[MapObject("chair", 0.0, 6.0)], show_legend=False
        )
        right = render_session_map(
            frame, None, CFG, objects=[MapObject("chair", 3.0, 2.0)], show_legend=False
        )
        near_x, near_y = centroid(near)
        far_x, far_y = centroid(far)
        right_x, right_y = centroid(right)
        assert far_y < near_y - 100
        assert right_x > near_x + 100

    def test_objects_extend_the_visible_area(self):
        image = render_session_map(
            [(0.0, 0.0)], None, CFG, objects=[MapObject("chair", 0.0, 40.0)]
        )
        assert _pixels(image, object_colors(["chair"])["chair"]) > 20

    def test_overlapping_labels_are_suppressed(self):
        image = np.full((400, 400, 3), 26, dtype=np.uint8)
        objects = [MapObject("person", 0.02 * i, 1.0) for i in range(5)]
        labels = _draw_objects(
            image, objects, (0.0, 1.0), 40.0, (200.0, 200.0), (0, 0, 399, 399)
        )
        assert labels == 1
        assert _pixels(image, object_colors(["person"])["person"]) > 0

    def test_separated_labels_are_all_drawn(self):
        image = np.full((400, 400, 3), 26, dtype=np.uint8)
        objects = [MapObject("person", 0.0, 1.0), MapObject("person", 4.0, 1.0)]
        labels = _draw_objects(
            image, objects, (2.0, 1.0), 40.0, (200.0, 200.0), (0, 0, 399, 399)
        )
        assert labels == 2

    def test_non_finite_positions_are_skipped(self):
        objects = [MapObject("ghost", float("nan"), 1.0)]
        image = render_session_map(
            [(0.0, 0.0)], None, CFG, objects=objects, show_legend=False
        )
        assert _pixels(image, object_colors(["ghost"])["ghost"]) == 0

    def test_legend_shows_a_swatch_per_class(self):
        objects = [MapObject("chair", 0.0, 2.0), MapObject("person", 1.0, 3.0)]
        with_legend = render_session_map([(0.0, 0.0)], None, CFG, objects=objects)
        without = render_session_map(
            [(0.0, 0.0)], None, CFG, objects=objects, show_legend=False
        )
        for colour in object_colors([obj.class_name for obj in objects]).values():
            assert _pixels(with_legend, colour) > _pixels(without, colour)


class TestRunnerMapObjects:
    def test_anchors_are_recorded_in_the_world_frame(self):
        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner._map_objects = {}
        runner._record_map_objects([_anchor(3, "chair", 0.5, 2.0)])
        assert set(runner._map_objects) == {3}
        assert abs(runner._map_objects[3].x - 0.5) <= 1e-9
        assert abs(runner._map_objects[3].y - 2.0) <= 1e-9

        # Facing +X, forward is +X and right is -Y.
        runner.trace.heading_deg = 90.0
        runner._record_map_objects([_anchor(3, "chair", 0.5, 2.0)])
        assert abs(runner._map_objects[3].x - 2.0) <= 1e-9
        assert abs(runner._map_objects[3].y + 0.5) <= 1e-9

    def test_nearby_same_class_is_merged(self):
        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner._map_objects = {}
        runner._record_map_objects([_anchor(1, "chair", 0.5, 2.0)])
        runner._record_map_objects([_anchor(2, "chair", 0.6, 2.0)])
        assert set(runner._map_objects) == {1}
        runner._record_map_objects([_anchor(3, "person", 0.6, 2.0)])
        assert set(runner._map_objects) == {1, 3}
        runner._record_map_objects([_anchor(4, "chair", 3.0, 2.0)])
        assert set(runner._map_objects) == {1, 3, 4}

    def test_existing_track_refreshes_its_marker(self):
        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner._map_objects = {}
        runner._record_map_objects([_anchor(1, "chair", 0.5, 2.0)])
        runner._record_map_objects([_anchor(1, "chair", 3.0, 2.0)])
        assert set(runner._map_objects) == {1}
        assert abs(runner._map_objects[1].x - 3.0) <= 1e-9

    def test_unusable_anchors_are_skipped(self):
        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner._map_objects = {}
        no_position = ObjectAnchor(
            track_id=4,
            class_name="chair",
            conf=0.9,
            box_xyxy=(0.0, 0.0, 10.0, 10.0),
            distance_m=1.0,
            position_g=(float("nan"), 1.0, 0.0),
            status="OK",
        )
        runner._record_map_objects(
            [no_position, _anchor(5, "chair", 0.0, 1.0, status="NO_PIXELS")]
        )
        assert runner._map_objects == {}

    def test_write_map_image_draws_class_coloured_markers(self, tmp_path):
        runner = LiveRunner.__new__(LiveRunner)
        runner.config = SimpleNamespace(map_png=str(tmp_path / "session_map.png"))
        runner.map_cfg = CFG
        runner.trace = SessionTrace()
        runner._map_objects = {}
        runner._map_error_reported = False
        runner.session_id = "sess"
        runner.uncalibrated = True
        runner._write_map_image(
            1, _grid(), "GUIDE", 4.0, {}, [_anchor(9, "chair", 0.5, 2.0)]
        )
        image = cv2.imread(str(tmp_path / "session_map.png"))
        assert image is not None
        assert _pixels(image, object_colors(["chair"])["chair"]) > 20

    def _runner(self) -> LiveRunner:
        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner._map_objects = {}
        return runner

    def test_recreated_track_keeps_one_moving_marker(self):
        runner = self._runner()
        runner._record_map_objects([_anchor(1, "person", 0.5, 2.0)], now=100.0)
        runner._record_map_objects([_anchor(7, "person", 0.5, 3.6)], now=101.5)
        assert set(runner._map_objects) == {1}
        marker = runner._map_objects[1]
        assert marker.track_id == 7
        assert abs(marker.x - 0.5) <= 1e-9
        assert abs(marker.y - 3.6) <= 1e-9
        assert marker.last_seen == 101.5

    def test_walking_person_keeps_one_marker_over_several_frames(self):
        runner = self._runner()
        for step, forward in enumerate((2.0, 3.1, 4.2, 5.3)):
            runner._record_map_objects(
                [_anchor(10 + step, "person", 0.0, forward)],
                now=100.0 + 1.2 * step,
            )
        assert len(runner._map_objects) == 1
        marker = next(iter(runner._map_objects.values()))
        assert abs(marker.y - 5.3) <= 1e-9

    def test_far_same_class_becomes_a_new_marker(self):
        runner = self._runner()
        runner._record_map_objects([_anchor(1, "person", 0.0, 2.0)], now=100.0)
        runner._record_map_objects([_anchor(2, "person", 0.0, 6.0)], now=100.1)
        assert set(runner._map_objects) == {1, 2}

    def test_nearest_detection_wins_a_contested_marker(self):
        runner = self._runner()
        runner._record_map_objects([_anchor(1, "person", 0.0, 2.0)], now=100.0)
        runner._record_map_objects(
            [
                _anchor(3, "person", 0.9, 2.0),
                _anchor(2, "person", 0.2, 2.0),
            ],
            now=101.0,
        )
        assert set(runner._map_objects) == {1, 3}
        assert runner._map_objects[1].track_id == 2
        assert abs(runner._map_objects[1].x - 0.2) <= 1e-9
        assert runner._map_objects[3].track_id == 3
        assert abs(runner._map_objects[3].x - 0.9) <= 1e-9

    def test_person_marker_expires_after_three_seconds(self):
        runner = self._runner()
        runner._record_map_objects([_anchor(1, "person", 0.5, 2.0)], now=100.0)
        runner._record_map_objects(None, now=102.9)
        assert set(runner._map_objects) == {1}
        runner._record_map_objects([], now=103.5)
        assert runner._map_objects == {}

    def test_static_marker_does_not_expire(self):
        runner = self._runner()
        runner._record_map_objects([_anchor(1, "chair", 0.5, 2.0)], now=100.0)
        runner._record_map_objects(None, now=110.0)
        assert set(runner._map_objects) == {1}


class TestRunnerSummaryTrail:
    def _runner(self) -> LiveRunner:
        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner.db = SimpleNamespace(path=None)
        runner.session_id = None
        runner.info = SimpleNamespace(label="synthetic", width=640, height=360)
        runner.proc_w = 320
        runner.proc_h = 180
        runner.profile = "fast"
        runner._goal_lock = threading.Lock()
        runner._goal_tracker = None
        runner._goal_agent = None
        runner.intrinsics_provenance = "default"
        runner.uncalibrated = True
        runner._processed = 4
        runner._started_s = time.monotonic()
        runner._errors = []
        return runner

    def test_summary_reports_the_trail(self):
        runner = self._runner()
        runner.trace.steps = 3
        runner.trace.gaps = 7
        runner.trace.distance_m = 2.5
        runner.trace.x = 0.5
        runner.trace.y = 1.25
        runner.trace.heading_deg = 90.0
        trail = runner.summary()["trail"]
        assert trail == {
            "steps": 3,
            "gaps": 7,
            "route_m": 2.5,
            "x_m": 0.5,
            "y_m": 1.25,
            "heading_deg": 90.0,
        }

    def test_frozen_run_is_visible_in_the_summary(self):
        runner = self._runner()
        runner.trace.gaps = 6
        trail = runner.summary()["trail"]
        assert trail["steps"] == 0
        assert trail["gaps"] == 6
        assert trail["route_m"] == 0.0


class TestRunnerMapWiring:
    def test_write_map_image_creates_the_file(self, tmp_path):
        runner = LiveRunner.__new__(LiveRunner)
        runner.config = SimpleNamespace(map_png=str(tmp_path / "session_map.png"))
        runner.map_cfg = CFG
        runner.trace = SessionTrace()
        runner._map_objects = {}
        runner._map_error_reported = False
        runner.session_id = "sess"
        runner.uncalibrated = True
        runner._write_map_image(
            1, _grid(), "GUIDE", 4.0, {"distance_nominal_m": 1.2}
        )
        assert (tmp_path / "session_map.png").exists()

    def test_no_map_path_writes_nothing(self, tmp_path):
        runner = LiveRunner.__new__(LiveRunner)
        runner.config = SimpleNamespace(map_png=None)
        runner.map_cfg = CFG
        runner.trace = None
        runner._map_objects = {}
        runner._map_error_reported = False
        runner.session_id = "sess"
        runner.uncalibrated = False
        runner._write_map_image(
            1, _grid(), "GUIDE", 4.0, {"distance_nominal_m": 1.2}
        )
        assert list(tmp_path.iterdir()) == []

    def test_map_write_expires_markers_on_the_supplied_clock(self, tmp_path):
        runner = LiveRunner.__new__(LiveRunner)
        runner.config = SimpleNamespace(map_png=str(tmp_path / "session_map.png"))
        runner.map_cfg = CFG
        runner.trace = SessionTrace()
        runner._map_objects = {
            1: MapObject("person", 0.0, 2.0, 1, last_seen=100.0)
        }
        runner._map_error_reported = False
        runner.session_id = "sess"
        runner.uncalibrated = False
        runner._write_map_image(1, None, "GUIDE", 0.0, {}, None, ts=104.0)
        assert runner._map_objects == {}

    def test_trace_time_uses_video_time_for_files(self):
        runner = LiveRunner.__new__(LiveRunner)
        runner.info = SimpleNamespace(is_camera=False, fps=30.0)
        assert abs(runner._trace_time(60, 1.0) - 2.0) <= 1e-9
        runner.info = SimpleNamespace(is_camera=True, fps=30.0)
        assert runner._trace_time(60, 12.5) == 12.5
        runner.info = SimpleNamespace(is_camera=False, fps=0.0)
        assert runner._trace_time(60, 12.5) == 12.5

    def test_update_trace_passes_plane_and_floor_mask(self):
        from core.config import Intrinsics

        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner.K = Intrinsics(
            fx=300.0, fy=300.0, cx=160.0, cy=90.0, width=320, height=180
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        labels = np.zeros((180, 320), dtype=np.uint8)
        labels[120:, :] = 1
        lift = SimpleNamespace(
            plane_fit=SimpleNamespace(
                n=np.array([0.0, 1.0, 0.0]),
                h_hat=2.0,
                status=SimpleNamespace(value="OK"),
            ),
            scale_result=SimpleNamespace(s=0.7),
        )
        outputs = SimpleNamespace(lift=lift, seg=SimpleNamespace(labels=labels))
        runner._update_trace(frame, outputs)
        assert runner.trace._prev_gray is not None
        assert runner.trace._prev_mask is not None
        assert runner.trace._prev_mask.shape == (180, 320)
        assert runner.trace._prev_plane is not None
        assert abs(runner.trace._prev_plane.height_m - 1.4) <= 1e-9

    def test_update_trace_skips_unusable_plane(self):
        from core.config import Intrinsics

        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner.K = Intrinsics(
            fx=300.0, fy=300.0, cx=160.0, cy=90.0, width=320, height=180
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        labels = np.zeros((180, 320), dtype=np.uint8)
        labels[120:, :] = 1
        lift = SimpleNamespace(
            plane_fit=SimpleNamespace(
                n=np.array([0.0, 1.0, 0.0]),
                h_hat=2.0,
                status=SimpleNamespace(value="NO_INLIERS"),
            ),
            scale_result=SimpleNamespace(s=0.7),
        )
        outputs = SimpleNamespace(lift=lift, seg=SimpleNamespace(labels=labels))
        runner._update_trace(frame, outputs)
        assert runner.trace._prev_gray is None
        assert runner.trace.misses == 1
