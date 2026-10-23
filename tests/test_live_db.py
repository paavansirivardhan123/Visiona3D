"""SQLite persistence for the live 3D object grid (no camera, no models)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from live.db import LiveDatabase


@dataclass
class _Obs:
    track_id: int
    class_name: str
    distance_m: float
    x: float
    y: float
    z: float

    def to_row(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "conf": 0.9,
            "distance_m": self.distance_m,
            "anchor_x": self.x,
            "anchor_y": self.y,
            "anchor_z": self.z,
            "anchor_status": "OK",
            "box_x1": 1.0,
            "box_y1": 2.0,
            "box_x2": 3.0,
            "box_y2": 4.0,
        }


class _Cfg:
    res = 0.1
    x_min = -3.0
    y_min = -0.5


def _session(db: LiveDatabase) -> str:
    session = db.start_session(
        source="unit-test",
        width=640,
        height=360,
        proc_width=640,
        proc_height=360,
        intrinsics={"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 180.0, "width": 640, "height": 360},
        intrinsics_provenance="calibrated",
        mount={"height_m": 1.4, "pitch_deg": 15.0},
        depth_variant="indoor",
        profile="fast",
        config={"test": True},
    )
    assert session
    return session


def _frame(db: LiveDatabase, session: str, index: int) -> int:
    frame_id = db.log_frame(
        session,
        frame_index=index,
        capture_ts=100.0 + index,
        processed_ts=100.5 + index,
        process_ms=42.0,
        decision="GUIDE",
        heading_deg=5.0,
        nearest_nominal_m=3.2,
        nearest_lower_m=2.9,
        nearest_bearing_deg=-12.0,
        grid_counts={"free": 10, "occupied": 2, "uncertain": 1, "unknown": 487},
        statuses={"seg": "OK", "depth": "OK"},
        timings={"seg": 100.0, "depth": 200.0},
        reasons=["unit test"],
    )
    assert frame_id is not None
    return frame_id


def test_objects_and_registry_roundtrip(tmp_path):
    db = LiveDatabase(tmp_path / "live.db")
    try:
        session = _session(db)
        frame_id = _frame(db, session, 1)
        db.log_objects(session, frame_id, 1, [_Obs(7, "person", 2.4, 0.3, 2.4, 1.2)])
        db.log_objects(session, frame_id, 2, [_Obs(7, "person", 1.9, 0.2, 1.9, 1.1)])

        rows = db.fetch_objects(session)
        assert len(rows) == 1
        row = rows[0]
        assert row["track_id"] == 7
        assert row["class_name"] == "person"
        assert row["first_frame"] == 1
        assert row["last_frame"] == 2
        assert row["observations"] == 2
        assert row["distance_m"] == 1.9
        assert row["anchor_y"] == 1.9
    finally:
        db.close()


def test_bev_snapshot_roundtrip(tmp_path):
    db = LiveDatabase(tmp_path / "live.db")
    try:
        session = _session(db)
        frame_id = _frame(db, session, 1)
        grid = np.zeros((4, 5), dtype=np.uint8)
        grid[1, 2] = 1
        grid[3, 4] = 2
        db.save_bev(session, frame_id, 1, 101.0, _Cfg(), grid)
        restored = db.fetch_bev(session, 1)
        assert restored is not None
        assert np.array_equal(restored, grid)
        assert db.fetch_bev(session, 999) is None
    finally:
        db.close()


def test_events_and_session_summary(tmp_path):
    db = LiveDatabase(tmp_path / "live.db")
    try:
        session = _session(db)
        _frame(db, session, 1)
        _frame(db, session, 2)
        db.log_event(session, 1, 101.0, "stop", "Stop. The path is unclear.")
        summary = db.session_summary(session)
        assert summary["frames"] == 2
        assert summary["avg_process_ms"] == 42.0
        assert summary["decisions"] == {"GUIDE": 2}
        assert summary["objects"] == 0
    finally:
        db.close()


def test_data_survives_reopen(tmp_path):
    path = tmp_path / "live.db"
    db = LiveDatabase(path)
    session = _session(db)
    frame_id = _frame(db, session, 1)
    db.log_objects(session, frame_id, 1, [_Obs(3, "chair", 4.0, -1.0, 4.0, 0.5)])
    db.close()

    reopened = LiveDatabase(path)
    try:
        rows = reopened.fetch_objects(session)
        assert len(rows) == 1
        assert rows[0]["class_name"] == "chair"
        assert rows[0]["anchor_x"] == -1.0
    finally:
        reopened.close()


def test_disabled_database_writes_nothing(tmp_path):
    path = tmp_path / "disabled.db"
    db = LiveDatabase(path, enabled=False)
    assert db.start_session(
        source="none",
        width=1,
        height=1,
        proc_width=1,
        proc_height=1,
        intrinsics={},
        intrinsics_provenance="none",
        mount={},
        depth_variant="indoor",
        profile="fast",
        config={},
    ) is None
    assert db.log_frame(
        None,
        frame_index=1,
        capture_ts=0.0,
        processed_ts=0.0,
        process_ms=0.0,
        decision="NO_PERCEPTION",
        heading_deg=None,
        nearest_nominal_m=None,
        nearest_lower_m=None,
        nearest_bearing_deg=None,
        grid_counts={},
        statuses={},
        timings={},
        reasons=[],
    ) is None
    assert not path.exists()
