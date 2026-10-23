"""3D object anchors and track IDs (no camera, no models)."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Intrinsics, MountConfig
from live.anchors import (
    assign_track_ids,
    box_mask,
    compute_object_anchors,
    nearest_object,
)
from perception.interfaces import Det2D
from perception.tracker import ObjectTracker

K = Intrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0, width=160, height=120)
MOUNT = MountConfig(h_known=1.4, pitch_deg=0.0)
FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def _det(box, name="person", conf=0.9):
    return Det2D(class_name=name, box_xyxy=np.asarray(box, dtype=np.float64), conf=conf)


def test_box_mask_clips_and_counts_pixels():
    mask = box_mask((120, 160), (-10, -10, 20, 30))
    assert mask.dtype == bool
    assert int(mask.sum()) == 20 * 30
    assert mask[0, 0]
    assert not mask[119, 159]
    with pytest.raises(ValueError):
        box_mask((120, 160), (10, 10, 10, 20))


def test_track_ids_are_stable_for_overlapping_boxes():
    tracker = ObjectTracker()
    first = assign_track_ids([_det((10, 20, 40, 60))], FRAME, tracker)
    second = assign_track_ids([_det((12, 21, 42, 61))], FRAME, tracker)
    assert len(first) == 1 and len(second) == 1
    assert first[0].track_id == second[0].track_id

    other = assign_track_ids([_det((100, 10, 140, 50))], FRAME, tracker)
    assert other[0].track_id != first[0].track_id


def test_anchor_geometry_matches_pinhole_projection():
    det = _det((70, 50, 91, 71))  # exclusive box end -> centroid (80, 60)
    tracked = assign_track_ids([det], FRAME)
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    anchors = compute_object_anchors(tracked, depth, K, MOUNT, plane=None)
    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.status == "OK"
    assert anchor.distance_m == pytest.approx(2.0, abs=1e-3)
    assert anchor.position_g[0] == pytest.approx(0.0, abs=1e-3)
    assert anchor.position_g[1] == pytest.approx(2.0, abs=1e-3)
    assert anchor.position_g[2] == pytest.approx(1.4, abs=1e-3)
    assert anchor.bearing_deg == pytest.approx(0.0, abs=1e-3)


def test_anchor_lateral_sign_is_right_positive():
    det = _det((90, 50, 110, 70))  # centroid right of centre
    tracked = assign_track_ids([det], FRAME)
    depth = np.full((120, 160), 3.0, dtype=np.float32)
    anchor = compute_object_anchors(tracked, depth, K, MOUNT, plane=None)[0]
    assert anchor.position_g[0] > 0.0
    assert anchor.bearing_deg > 0.0


def test_nearest_object_filters_class_and_status():
    det = _det((10, 10, 30, 30), name="chair")
    tracked = assign_track_ids([det], FRAME)
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    anchor = compute_object_anchors(tracked, depth, K, MOUNT, plane=None)[0]
    assert nearest_object([anchor], "chair") is anchor
    assert nearest_object([anchor], "person") is None
    assert nearest_object([anchor], require_ok=False) is anchor
