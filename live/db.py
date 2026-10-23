"""SQLite persistence for the live 3D object grid.

Everything the system sees is written to one local database: session
provenance, per-frame decisions, per-object 3D anchors, a queryable object
registry and compressed BEV snapshots. The schema is versioned so a future
migration can be detected instead of silently reading old columns.

Only the standard library (``sqlite3``, ``zlib``) and numpy are used here, so
the storage layer is testable without camera, models or network.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid
import zlib

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_DB_PATH = Path("database") / "visiona_live.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    source TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    proc_width INTEGER NOT NULL,
    proc_height INTEGER NOT NULL,
    intrinsics_json TEXT NOT NULL,
    intrinsics_provenance TEXT NOT NULL,
    mount_json TEXT NOT NULL,
    depth_variant TEXT NOT NULL,
    profile TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    capture_ts REAL NOT NULL,
    processed_ts REAL NOT NULL,
    process_ms REAL NOT NULL,
    decision TEXT NOT NULL,
    heading_deg REAL,
    nearest_nominal_m REAL,
    nearest_lower_m REAL,
    nearest_bearing_deg REAL,
    grid_counts_json TEXT NOT NULL,
    statuses_json TEXT NOT NULL,
    timings_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    conf REAL NOT NULL,
    distance_m REAL NOT NULL,
    anchor_x REAL NOT NULL,
    anchor_y REAL NOT NULL,
    anchor_z REAL NOT NULL,
    anchor_status TEXT NOT NULL,
    box_x1 REAL NOT NULL,
    box_y1 REAL NOT NULL,
    box_x2 REAL NOT NULL,
    box_y2 REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS object_registry (
    session_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    first_frame INTEGER NOT NULL,
    last_frame INTEGER NOT NULL,
    distance_m REAL NOT NULL,
    anchor_x REAL NOT NULL,
    anchor_y REAL NOT NULL,
    anchor_z REAL NOT NULL,
    observations INTEGER NOT NULL,
    PRIMARY KEY (session_id, track_id)
);
CREATE TABLE IF NOT EXISTS bev_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    ts REAL NOT NULL,
    res REAL NOT NULL,
    x_min REAL NOT NULL,
    y_min REAL NOT NULL,
    n_rows INTEGER NOT NULL,
    n_cols INTEGER NOT NULL,
    grid_zlib BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_session_frame
    ON objects(session_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_objects_session_track
    ON objects(session_id, track_id);
CREATE INDEX IF NOT EXISTS idx_bev_session_frame
    ON bev_snapshots(session_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_events_session
    ON events(session_id, frame_index);
"""

_REGISTRY_UPSERT = """
INSERT INTO object_registry (
    session_id, track_id, class_name, first_frame, last_frame,
    distance_m, anchor_x, anchor_y, anchor_z, observations
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
ON CONFLICT(session_id, track_id) DO UPDATE SET
    class_name = excluded.class_name,
    last_frame = excluded.last_frame,
    distance_m = excluded.distance_m,
    anchor_x = excluded.anchor_x,
    anchor_y = excluded.anchor_y,
    anchor_z = excluded.anchor_z,
    observations = object_registry.observations + 1
"""


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class LiveDatabase:
    """Thin, versioned SQLite writer. ``enabled=False`` makes every call a no-op."""

    def __init__(self, path: str | Path | None = DEFAULT_DB_PATH, enabled: bool = True) -> None:
        self.path = None if path is None else Path(path)
        self.enabled = bool(enabled and self.path is not None)
        self._conn: sqlite3.Connection | None = None
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(row["value"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema version {row['value']} != expected "
                f"{SCHEMA_VERSION}; use a new database file or migrate it"
            )
        self._conn.commit()

    def start_session(
        self,
        *,
        source: str,
        width: int,
        height: int,
        proc_width: int,
        proc_height: int,
        intrinsics: dict,
        intrinsics_provenance: str,
        mount: dict,
        depth_variant: str,
        profile: str,
        config: dict,
    ) -> str | None:
        if not self.enabled or self._conn is None:
            return None
        session_id = uuid.uuid4().hex
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT INTO sessions (id, started_at, source, width, height, "
            "proc_width, proc_height, intrinsics_json, intrinsics_provenance, "
            "mount_json, depth_variant, profile, config_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                started,
                str(source),
                int(width),
                int(height),
                int(proc_width),
                int(proc_height),
                _json(intrinsics),
                str(intrinsics_provenance),
                _json(mount),
                str(depth_variant),
                str(profile),
                _json(config),
            ),
        )
        self._conn.commit()
        return session_id

    def log_frame(
        self,
        session_id: str | None,
        *,
        frame_index: int,
        capture_ts: float,
        processed_ts: float,
        process_ms: float,
        decision: str,
        heading_deg: float | None,
        nearest_nominal_m: float | None,
        nearest_lower_m: float | None,
        nearest_bearing_deg: float | None,
        grid_counts: dict,
        statuses: dict,
        timings: dict,
        reasons: list,
    ) -> int | None:
        if not self.enabled or self._conn is None or session_id is None:
            return None
        cursor = self._conn.execute(
            "INSERT INTO frames (session_id, frame_index, capture_ts, "
            "processed_ts, process_ms, decision, heading_deg, "
            "nearest_nominal_m, nearest_lower_m, nearest_bearing_deg, "
            "grid_counts_json, statuses_json, timings_json, reasons_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(session_id),
                int(frame_index),
                float(capture_ts),
                float(processed_ts),
                float(process_ms),
                str(decision),
                None if heading_deg is None else float(heading_deg),
                None if nearest_nominal_m is None else float(nearest_nominal_m),
                None if nearest_lower_m is None else float(nearest_lower_m),
                None if nearest_bearing_deg is None else float(nearest_bearing_deg),
                _json(grid_counts),
                _json(statuses),
                _json(timings),
                _json(reasons),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def log_objects(
        self,
        session_id: str | None,
        frame_id: int | None,
        frame_index: int,
        observations: list,
    ) -> None:
        """Insert per-frame observations and upsert the object registry."""
        if (
            not self.enabled
            or self._conn is None
            or session_id is None
            or frame_id is None
            or not observations
        ):
            return
        rows = [obs.to_row() for obs in observations]
        self._conn.executemany(
            "INSERT INTO objects (session_id, frame_id, frame_index, track_id, "
            "class_name, conf, distance_m, anchor_x, anchor_y, anchor_z, "
            "anchor_status, box_x1, box_y1, box_x2, box_y2) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(session_id),
                    int(frame_id),
                    int(frame_index),
                    row["track_id"],
                    row["class_name"],
                    row["conf"],
                    row["distance_m"],
                    row["anchor_x"],
                    row["anchor_y"],
                    row["anchor_z"],
                    row["anchor_status"],
                    row["box_x1"],
                    row["box_y1"],
                    row["box_x2"],
                    row["box_y2"],
                )
                for row in rows
            ],
        )
        self._conn.executemany(
            _REGISTRY_UPSERT,
            [
                (
                    str(session_id),
                    row["track_id"],
                    row["class_name"],
                    int(frame_index),
                    int(frame_index),
                    row["distance_m"],
                    row["anchor_x"],
                    row["anchor_y"],
                    row["anchor_z"],
                )
                for row in rows
            ],
        )
        self._conn.commit()

    def save_bev(
        self,
        session_id: str | None,
        frame_id: int | None,
        frame_index: int,
        ts: float,
        cfg: object,
        grid: np.ndarray,
    ) -> None:
        if (
            not self.enabled
            or self._conn is None
            or session_id is None
            or frame_id is None
        ):
            return
        arr = np.asarray(grid, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"BEV grid must be 2D, got {arr.shape}")
        payload = zlib.compress(arr.tobytes(), 6)
        self._conn.execute(
            "INSERT INTO bev_snapshots (session_id, frame_id, frame_index, ts, "
            "res, x_min, y_min, n_rows, n_cols, grid_zlib) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(session_id),
                int(frame_id),
                int(frame_index),
                float(ts),
                float(cfg.res),
                float(cfg.x_min),
                float(cfg.y_min),
                int(arr.shape[0]),
                int(arr.shape[1]),
                sqlite3.Binary(payload),
            ),
        )
        self._conn.commit()

    def log_event(
        self,
        session_id: str | None,
        frame_index: int,
        ts: float,
        kind: str,
        text: str,
    ) -> None:
        if not self.enabled or self._conn is None or session_id is None:
            return
        self._conn.execute(
            "INSERT INTO events (session_id, frame_index, ts, kind, text) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(session_id), int(frame_index), float(ts), str(kind), str(text)),
        )
        self._conn.commit()

    def fetch_objects(self, session_id: str) -> list[dict]:
        if not self.enabled or self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT track_id, class_name, first_frame, last_frame, distance_m, "
            "anchor_x, anchor_y, anchor_z, observations "
            "FROM object_registry WHERE session_id = ? "
            "ORDER BY distance_m ASC, track_id ASC",
            (str(session_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def fetch_bev(self, session_id: str, frame_index: int) -> np.ndarray | None:
        if not self.enabled or self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT n_rows, n_cols, grid_zlib FROM bev_snapshots "
            "WHERE session_id = ? AND frame_index = ? "
            "ORDER BY id DESC LIMIT 1",
            (str(session_id), int(frame_index)),
        ).fetchone()
        if row is None:
            return None
        raw = zlib.decompress(bytes(row["grid_zlib"]))
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        return arr.reshape(int(row["n_rows"]), int(row["n_cols"]))

    def session_summary(self, session_id: str) -> dict:
        if not self.enabled or self._conn is None:
            return {}
        frames = self._conn.execute(
            "SELECT COUNT(*) AS n, AVG(process_ms) AS avg_ms FROM frames "
            "WHERE session_id = ?",
            (str(session_id),),
        ).fetchone()
        decisions = self._conn.execute(
            "SELECT decision, COUNT(*) AS n FROM frames WHERE session_id = ? "
            "GROUP BY decision",
            (str(session_id),),
        ).fetchall()
        objects = self._conn.execute(
            "SELECT COUNT(*) AS n FROM object_registry WHERE session_id = ?",
            (str(session_id),),
        ).fetchone()
        return {
            "frames": int(frames["n"] or 0),
            "avg_process_ms": float(frames["avg_ms"] or 0.0),
            "decisions": {str(row["decision"]): int(row["n"]) for row in decisions},
            "objects": int(objects["n"] or 0),
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None
        self.enabled = False


__all__ = ["DEFAULT_DB_PATH", "LiveDatabase", "SCHEMA_VERSION"]
