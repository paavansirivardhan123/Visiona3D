"""Offline tests for the session-map zone layer.

No models, camera, video or network is used. The grid, labels and point map
are synthetic and the rendered PNG is inspected in memory.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from core.config import MASK_FLOOR, MASK_WALL
from live.map_image import (
    BG_BGR,
    TRAIL_BGR,
    MapZone,
    render_session_map,
    trail_length_m,
)
from live.runner import LiveRunner
from live.trace import SessionTrace
from live.zones import (
    frame_zone_votes,
    merge_zone_votes,
    resolve_zones,
    zone_summary,
)
from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    MapConfig,
)

CELL = 0.5
CFG = MapConfig(res=CELL, x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0)
RES = 0.2


def _grid() -> np.ndarray:
    grid = np.full(CFG.shape, CELL_UNKNOWN, dtype=np.uint8)
    grid[3, 0] = CELL_FREE  # world (0.25, 1.75) -> zone cell (1, 8)
    grid[3, 1] = CELL_OCCUPIED  # world (0.75, 1.75) -> zone cell (3, 8)
    grid[2, 2] = CELL_OCCUPIED  # wall-labelled below -> zone cell (6, 6)
    grid[0, 3] = CELL_FREE  # world (1.75, 0.25) -> zone cell (8, 1)
    return grid


def _wall_points() -> tuple[np.ndarray, np.ndarray]:
    points = np.array([[1.25, 1.25, 1.0]], dtype=np.float64)
    labels = np.array([MASK_WALL], dtype=np.int64)
    return points, labels


def _dominant(image: np.ndarray, channel: int, margin: int = 40) -> int:
    data = image[:, :, :].astype(np.int16)
    others = [data[:, :, i] for i in range(3) if i != channel]
    mask = (data[:, :, channel] > others[0] + margin) & (
        data[:, :, channel] > others[1] + margin
    )
    return int(np.count_nonzero(mask))


def _body(image: np.ndarray) -> np.ndarray:
    return image[70:790, 18:882]


class TestFrameZoneVotes:
    def test_free_and_danger_and_wall_cells_are_classified(self):
        points, labels = _wall_points()
        votes = frame_zone_votes(_grid(), labels, points, CFG)

        assert votes[(1, 8)] == (0, 0, 1)  # certified free floor
        assert votes[(3, 8)] == (0, 1, 0)  # occupied, not a wall
        assert votes[(6, 6)] == (1, 0, 0)  # wall label wins over occupied
        assert votes[(8, 1)] == (0, 0, 1)

    def test_pose_rotates_cells_into_the_world(self):
        points, labels = _wall_points()
        votes = frame_zone_votes(
            _grid(), labels, points, CFG, pose=(2.0, 1.0, 90.0)
        )

        # Facing +X: ego (0.25, 1.75) becomes world (3.75, 0.75).
        assert (18, 3) in votes
        assert votes[(18, 3)][2] == 1
        assert (1, 8) not in votes

    def test_a_nearby_object_is_not_voted_as_danger(self):
        points, labels = _wall_points()
        obj = SimpleNamespace(position_g=(0.75, 1.75, 0.0))

        votes = frame_zone_votes(
            _grid(), labels, points, CFG, objects=[obj]
        )

        assert (3, 8) not in votes
        # The wall cell keeps its wall vote; objects never erase structure.
        assert votes[(6, 6)] == (1, 0, 0)

    def test_a_distant_object_leaves_danger_alone(self):
        points, labels = _wall_points()
        obj = SimpleNamespace(position_g=(0.75, 0.25, 0.0))

        votes = frame_zone_votes(
            _grid(), labels, points, CFG, objects=[obj]
        )

        assert votes[(3, 8)] == (0, 1, 0)

    def test_shape_mismatch_returns_no_votes(self):
        grid = np.full((2, 2), CELL_UNKNOWN, dtype=np.uint8)
        points = np.array([[0.1, 0.1, 0.0]])
        labels = np.array([MASK_FLOOR])
        assert frame_zone_votes(grid, labels, points, CFG) == {}

    def test_merge_adds_each_category(self):
        store: dict[tuple[int, int], list[int]] = {}
        merge_zone_votes(store, {(1, 2): (0, 1, 2)})
        merge_zone_votes(store, {(1, 2): (1, 0, 1)})
        assert store[(1, 2)] == [1, 1, 3]


class TestResolveZones:
    def test_wall_wins_then_danger_then_walkable(self):
        store = {
            (0, 0): [1, 9, 9],
            (2, 0): [0, 3, 1],
            (4, 0): [0, 1, 3],
        }
        kinds = {round(z.x, 3): z.kind for z in resolve_zones(
            store, res=RES, unknown_band_cells=0
        )}
        assert kinds[0.1] == "wall"
        assert kinds[0.5] == "danger"
        assert kinds[0.9] == "walkable"

    def test_danger_needs_to_outvote_walkable(self):
        store = {(0, 0): [0, 2, 2]}
        zones = resolve_zones(store, res=RES, unknown_band_cells=0)
        assert [z.kind for z in zones] == ["walkable"]

    def test_unknown_band_surrounds_the_known_area(self):
        store = {(5, 5): [0, 0, 1]}
        banded = resolve_zones(store, res=RES, unknown_band_cells=1)
        unknown = [z for z in banded if z.kind == "unknown"]
        assert len(unknown) == 8
        assert {(round(z.x / RES - 0.5), round(z.y / RES - 0.5)) for z in unknown} == {
            (4, 4), (4, 5), (4, 6),
            (5, 4), (5, 6),
            (6, 4), (6, 5), (6, 6),
        }

    def test_zero_band_has_no_unknown_cells(self):
        zones = resolve_zones({(5, 5): [0, 0, 1]}, res=RES, unknown_band_cells=0)
        assert [z.kind for z in zones] == ["walkable"]

    def test_summary_measures_each_area(self):
        zones = [
            MapZone("walkable", 0.1, 0.1),
            MapZone("walkable", 0.3, 0.1),
            MapZone("wall", 0.5, 0.1),
            MapZone("danger", 0.7, 0.1),
        ]
        summary = zone_summary(zones, res=0.2)
        assert abs(summary["walkable"] - 0.08) <= 1e-9
        assert abs(summary["wall"] - 0.04) <= 1e-9
        assert abs(summary["danger"] - 0.04) <= 1e-9


class TestZoneRendering:
    def test_walkable_zone_paints_blue(self):
        zones = [MapZone("walkable", x, y) for x in (2.0, 2.2) for y in (2.0, 2.2)]
        image = render_session_map(
            [(0.0, 0.0), (0.0, 1.0)], None, MapConfig(), zones=zones, show_legend=False
        )
        assert _dominant(_body(image), 0) > 40

    def test_wall_zone_paints_grey(self):
        zones = [MapZone("wall", x, y) for x in (2.0, 2.2) for y in (2.0, 2.2)]
        image = render_session_map(
            [(0.0, 0.0), (0.0, 1.0)], None, MapConfig(), zones=zones, show_legend=False
        )
        body = _body(image).astype(np.int16)
        grey = (
            (np.abs(body[:, :, 0] - body[:, :, 1]) <= 6)
            & (np.abs(body[:, :, 1] - body[:, :, 2]) <= 6)
            & (body[:, :, 0] > 70)
            & (body[:, :, 0] < 130)
        )
        assert int(np.count_nonzero(grey)) > 20

    def test_danger_zone_paints_bright_red(self):
        zones = [MapZone("danger", x, y) for x in (2.0, 2.2) for y in (2.0, 2.2)]
        image = render_session_map(
            [(0.0, 0.0), (0.0, 1.0)], None, MapConfig(), zones=zones, show_legend=False
        )
        body = _body(image).astype(np.int16)
        red = (body[:, :, 2] > 120) & (body[:, :, 2] > body[:, :, 0] + 60)
        assert int(np.count_nonzero(red)) > 20

    def test_unknown_zone_paints_dark_red(self):
        zones = [MapZone("unknown", x, y) for x in (2.0, 2.2) for y in (2.0, 2.2)]
        image = render_session_map(
            [(0.0, 0.0), (0.0, 1.0)], None, MapConfig(), zones=zones, show_legend=False
        )
        body = _body(image).astype(np.int16)
        dark = (
            (body[:, :, 2] > 55)
            & (body[:, :, 2] < 110)
            & (body[:, :, 2] > body[:, :, 0] + 25)
        )
        assert int(np.count_nonzero(dark)) > 20

    def test_zone_extends_the_visible_area(self):
        image = render_session_map(
            [(0.0, 0.0)],
            None,
            MapConfig(),
            zones=[MapZone("walkable", 0.0, 40.0)],
            show_legend=False,
        )
        assert _dominant(_body(image), 0) > 0

    def test_legend_draws_the_zone_key(self):
        zones = [MapZone("walkable", 2.0, 2.0)]
        with_legend = render_session_map(
            [(0.0, 0.0)], None, MapConfig(), zones=zones
        )
        without = render_session_map(
            [(0.0, 0.0)], None, MapConfig(), zones=zones, show_legend=False
        )
        panel = (slice(100, 260), slice(18, 260))
        changed = int(
            np.count_nonzero(
                np.any(
                    with_legend[panel] != without[panel],
                    axis=2,
                )
            )
        )
        assert changed > 200

    def test_route_is_green(self):
        image = render_session_map([(0.0, 0.0), (0.0, 4.0)], None, MapConfig())
        pixels = int(np.count_nonzero(np.all(image == TRAIL_BGR, axis=2)))
        b, g, r = TRAIL_BGR
        assert pixels > 50
        assert g > r and g > b


class TestTrailLength:
    def test_sum_of_segments(self):
        assert abs(trail_length_m([(0.0, 0.0), (0.0, 3.0), (4.0, 3.0)]) - 7.0) <= 1e-9

    def test_non_finite_points_are_skipped(self):
        points = [(0.0, 0.0), (float("nan"), 1.0), (0.0, 2.0)]
        assert abs(trail_length_m(points) - 2.0) <= 1e-9


class TestRunnerZoneAccumulation:
    def _runner(self) -> LiveRunner:
        runner = LiveRunner.__new__(LiveRunner)
        runner.trace = SessionTrace()
        runner.map_cfg = CFG
        runner._map_zones = {}
        return runner

    def test_frame_votes_accumulate_in_the_world(self):
        points, labels = _wall_points()
        runner = self._runner()
        runner._record_map_zones(_grid(), labels, points, [])
        assert runner._map_zones[(1, 8)][2] == 1
        assert runner._map_zones[(6, 6)][0] == 1

        runner.trace.x = 2.0
        runner.trace.y = 1.0
        runner.trace.heading_deg = 90.0
        runner._record_map_zones(_grid(), labels, points, [])
        assert runner._map_zones[(18, 3)][2] == 1
        assert runner._map_zones[(1, 8)][2] == 1

    def test_write_map_image_paints_accumulated_walkable_cells(self, tmp_path):
        runner = self._runner()
        runner.config = SimpleNamespace(map_png=str(tmp_path / "session_map.png"))
        runner._map_objects = {}
        runner._map_error_reported = False
        runner.session_id = "sess"
        runner.uncalibrated = True
        runner._map_zones = {
            (col, row): [0, 0, 1] for col in range(10, 15) for row in range(10, 15)
        }
        runner._write_map_image(1, None, "GUIDE", 0.0, {})

        image = cv2.imread(str(tmp_path / "session_map.png"))
        assert image is not None
        body = image[70:790, 18:882].astype(np.int16)
        blue = (body[:, :, 0] > body[:, :, 2] + 40) & (body[:, :, 0] > body[:, :, 1] + 40)
        assert int(np.count_nonzero(blue)) > 40
        assert not np.all(image == np.array(BG_BGR, dtype=np.uint8))
