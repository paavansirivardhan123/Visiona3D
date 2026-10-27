"""Asynchronous live runner connecting capture, perception, mapping and speech.

Synchronisation model
---------------------
* Camera mode: a capture thread keeps only the newest frame. A perception
  thread takes that frame and processes it. If perception is slower than the
  camera, intermediate frames are dropped instead of queued, so the HUD never
  drifts minutes behind the sensor.
* Video-file mode: frames are read sequentially so every processed frame is a
  real video frame, not a repeated one.
* The HUD draws on the exact frame that produced the masks and anchors, and
  shows the frame age. Overlays are never drawn on a newer, different frame.

The metric stack is CPU-bound. The ``fast``/``balanced``/``quality`` profiles
change the model input size; they do not change the safety rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from core.config import Intrinsics, MountConfig, ScaleConfig
from calibration.depth_interval_config import (
    DepthIntervalConfigError,
    load_depth_interval,
)
from geometry.scale import ScaleEstimator
from live import overlay as hud
from live.anchors import (
    ObjectAnchor,
    assign_track_ids,
    compute_object_anchors,
)
from live.db import DEFAULT_DB_PATH, LiveDatabase
from live.goal_agent import GeminiGoalInterpreter
from live.goals import (
    DEFAULT_ARRIVAL_STABLE_FRAMES,
    DEFAULT_GOAL_ARRIVAL_M,
    GoalSpec,
    GoalTracker,
    build_goal_tracker,
    is_cancel_goal,
    parse_goal_text,
)
from live.guidance_bridge import GuidanceAnnouncer
from live.imaging import (
    DEFAULT_PROC_WIDTH,
    intrinsics_from_hfov,
    processing_size,
    provenance_is_measured,
    scale_calibration,
)
from live.source import SourceInfo, VideoSource, cv2_backend_note, parse_source
from mapping.bev import build_bev
from mapping.grid import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNCERTAIN,
    CELL_UNKNOWN,
    MapConfig,
    world_to_cell,
)
from mapping.obstacles import nearest_obstacle_telemetry
from perception.depth_backend import DepthAnythingBackend
from perception.detector_backend import YoloDetectorBackend
from perception.fusion_config import (
    FusionConfigError,
    load_tuned_fusion_config,
    wrap_seg_with_fusion,
)
from perception.pipeline import PerceptionPipeline
from perception.seg_backend import SegFormerBackend
from perception.seg_fusion import MobileSAMBackend
from perception.tracker import ObjectTracker
from planning.astar import astar_plan
from planning.costmap import build_costmap
from planning.guidance import compute_guidance
from planning.targets import find_anchor_target, find_explore_target
from scripts.plan_demo import _certified_mask_for_free_space, load_intrinsics_json
from scripts.real_models_smoke import _yolo_weights, check_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights"
DEFAULT_FLOOR_THRESHOLD = 0.90  # TO TUNE ON REAL DATA

PROFILE_SIZES = {
    # (segmentation input, depth input). Both are reduced model inputs, not
    # frame sizes; the safety pipeline still runs at the processing resolution.
    "fast": (256, 266),
    "balanced": (320, 364),
    "quality": (512, 518),
}
PROFILES = tuple(PROFILE_SIZES)


@dataclass(frozen=True)
class LiveConfig:
    source: Any = 0
    intrinsics_json: str | None = None
    hfov_deg: float | None = None
    mount_height_m: float = 1.4  # TO TUNE ON REAL DATA
    pitch_deg: float = 15.0  # TO TUNE ON REAL DATA
    proc_width: int = DEFAULT_PROC_WIDTH  # TO TUNE ON REAL DATA
    profile: str = "fast"
    weights_dir: str | Path = DEFAULT_WEIGHTS
    depth_variant: str = "indoor"
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD
    fusion_config_json: str | None = None
    conformal_depth_json: str | None = None
    goal_text: str | None = None
    goal_class: str | None = None
    goal_model: str | None = None
    voice_agent: bool = True
    arrival_distance_m: float = DEFAULT_GOAL_ARRIVAL_M  # TO TUNE ON REAL DATA
    arrival_stable_frames: int = DEFAULT_ARRIVAL_STABLE_FRAMES  # TO TUNE ON REAL DATA
    db_path: str | Path | None = DEFAULT_DB_PATH
    enable_db: bool = True
    speech: bool = True
    speak_objects: bool = True
    voice: bool = True
    headless: bool = False
    out_video: str | None = None
    video_fps: float = 4.0
    max_frames: int | None = None
    start_frame: int = 0
    loop: bool = False
    realtime: bool = True
    grid_every: int = 1
    camera_width: int | None = None
    camera_height: int | None = None

    def validate(self) -> None:
        if self.profile not in PROFILE_SIZES:
            raise ValueError(
                f"profile must be one of {PROFILES}, got {self.profile!r}"
            )
        if not 0.0 <= float(self.floor_threshold) <= 1.0:
            raise ValueError("floor_threshold must be in [0, 1]")
        if self.proc_width < 160:
            raise ValueError("proc_width must be at least 160")
        if self.max_frames is not None and int(self.max_frames) <= 0:
            raise ValueError("max_frames must be positive or None")
        if int(self.start_frame) < 0:
            raise ValueError("start_frame must be non-negative")
        if int(self.grid_every) <= 0:
            raise ValueError("grid_every must be positive")
        if self.goal_text and self.goal_class:
            raise ValueError("pass goal_text or goal_class, not both")
        if float(self.arrival_distance_m) <= 0.0:
            raise ValueError("arrival_distance_m must be positive")
        if int(self.arrival_stable_frames) < 1:
            raise ValueError("arrival_stable_frames must be at least 1")
        if not self.intrinsics_json and self.hfov_deg is None:
            raise ValueError(
                "live mode needs calibrated intrinsics (--intrinsics). "
                "For an uncalibrated demo only, pass --hfov 75 explicitly."
            )


@dataclass
class FrameResult:
    seq: int
    frame_index: int
    capture_ts: float
    processed_ts: float
    process_ms: float
    frame_bgr: np.ndarray
    labels: np.ndarray | None
    observations: list[ObjectAnchor] = field(default_factory=list)
    bev_grid: np.ndarray | None = None
    path_cells: list[tuple[int, int]] = field(default_factory=list)
    target_cell: tuple[int, int] | None = None
    guidance: Any = None
    nearest: dict = field(default_factory=dict)
    status_line: str = ""
    db_line: str = ""
    plane_normal: np.ndarray | None = None
    goal_line: str = ""
    goal_candidate: ObjectAnchor | None = None
    goal_complete: bool = False


def _grid_counts(grid: np.ndarray) -> dict[str, int]:
    return {
        "free": int(np.count_nonzero(grid == CELL_FREE)),
        "occupied": int(np.count_nonzero(grid == CELL_OCCUPIED)),
        "uncertain": int(np.count_nonzero(grid == CELL_UNCERTAIN)),
        "unknown": int(np.count_nonzero(grid == CELL_UNKNOWN)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class LiveRunner:
    """Owns sources, workers, DB and HUD for one live session."""

    def __init__(self, config: LiveConfig) -> None:
        config.validate()
        self.config = config
        self.profile = str(config.profile)
        self.seg_input, self.depth_input = PROFILE_SIZES[self.profile]
        self.weights = Path(config.weights_dir)
        self.map_cfg = MapConfig()
        self.mount = MountConfig(
            h_known=float(config.mount_height_m),
            pitch_deg=float(config.pitch_deg),
        )
        self.estimator = ScaleEstimator(self.mount, ScaleConfig())
        self.tracker = ObjectTracker()

        self._goal_lock = threading.Lock()
        self._goal_tracker = build_goal_tracker(
            goal_text=config.goal_text,
            goal_class=config.goal_class,
            arrival_distance_m=config.arrival_distance_m,
            stable_frames=config.arrival_stable_frames,
        )
        self._goal_agent = (
            GeminiGoalInterpreter(model=config.goal_model)
            if config.voice_agent and config.voice
            else None
        )
        self._last_index = int(config.start_frame)
        self._voice = None
        self._voice_ready = False
        self._voice_listening = False
        self._voice_hint_given = False
        self._last_voice_error_s: dict[str, float] = {}
        self._pending_goal_events: list[tuple[float, str, str]] = []

        self.source = VideoSource(
            parse_source(config.source),
            width=config.camera_width,
            height=config.camera_height,
        )
        self.info = self._open_source()
        self.processing = processing_size(
            self.info.width, self.info.height, config.proc_width
        )
        self.proc_w = self.processing.width
        self.proc_h = self.processing.height
        self.K, self.intrinsics_provenance = self._resolve_intrinsics()
        self.uncalibrated = not provenance_is_measured(self.intrinsics_provenance)

        self.seg_backend, self.depth_backend, self.detector_backend = (
            self._build_backends()
        )
        self.pipeline = PerceptionPipeline(
            seg=self.seg_backend,
            depth=self.depth_backend,
            detector=self.detector_backend,
            K=self.K,
            mount=self.mount,
            estimator=self.estimator,
            conformal_depth=self._load_depth_interval(),
            floor_threshold=float(config.floor_threshold),
        )

        self.db = LiveDatabase(
            config.db_path,
            enabled=bool(config.enable_db and config.db_path),
        )
        self.session_id = self.db.start_session(
            source=self.info.label,
            width=self.info.width,
            height=self.info.height,
            proc_width=self.proc_w,
            proc_height=self.proc_h,
            intrinsics={
                "fx": float(self.K.fx),
                "fy": float(self.K.fy),
                "cx": float(self.K.cx),
                "cy": float(self.K.cy),
                "width": int(self.K.width),
                "height": int(self.K.height),
            },
            intrinsics_provenance=self.intrinsics_provenance,
            mount={
                "height_m": float(self.mount.h_known),
                "pitch_deg": float(self.mount.pitch_deg),
            },
            depth_variant=str(config.depth_variant),
            profile=self.profile,
            config=_json_safe(
                {
                    "proc_width": config.proc_width,
                    "floor_threshold": config.floor_threshold,
                    "goal_text": config.goal_text,
                    "goal_class": config.goal_class,
                    "goal_interpreter": (
                        f"gemini:{self._goal_agent.model}"
                        if self._goal_agent is not None and self._goal_agent.available
                        else "local"
                    ),
                    "goal_candidates": (
                        list(self._goal_tracker.spec.candidates)
                        if self._goal_tracker is not None
                        else None
                    ),
                    "arrival_distance_m": config.arrival_distance_m,
                    "arrival_stable_frames": config.arrival_stable_frames,
                    "seg_input": self.seg_input,
                    "depth_input": self.depth_input,
                }
            ),
        )
        self.db_line = (
            f"db {Path(str(config.db_path)).name}"
            if self.session_id
            else "db off"
        )
        if self.session_id:
            self.db_line += f"#{self.session_id[:8]}"

        self.speech = None
        self._speech_enabled = bool(config.speech)
        if config.speech:
            from audio.speech import SpeechEngine

            self.speech = SpeechEngine()
        self.announcer = GuidanceAnnouncer(enabled=True)

        self._cond = threading.Condition()
        self._frame: tuple[int, float, np.ndarray] | None = None
        self._capture_done = False
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._result_lock = threading.Lock()
        self._result: FrameResult | None = None
        self._result_seq = 0
        self._processed = 0
        self._capture_index = int(config.start_frame)
        self._started_s = 0.0
        self._threads: list[threading.Thread] = []
        self._writer: cv2.VideoWriter | None = None
        self._last_written_seq = -1
        self._errors: list[str] = []

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _open_source(self) -> SourceInfo:
        info = self.source.open()
        if int(self.config.start_frame) > 0 and not info.is_camera:
            self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.config.start_frame))
        return info

    def _resolve_intrinsics(self) -> tuple[Intrinsics, str]:
        if self.config.intrinsics_json:
            try:
                K = load_intrinsics_json(self.config.intrinsics_json)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"failed to load intrinsics {self.config.intrinsics_json}: {exc}"
                ) from exc
            payload = json.loads(
                Path(self.config.intrinsics_json).read_text(encoding="utf-8")
            )
            provenance = str(payload.get("provenance", "unspecified"))
            if (K.width, K.height) != (self.info.width, self.info.height):
                raise ValueError(
                    f"intrinsics describe {K.width}x{K.height} but the source is "
                    f"{self.info.width}x{self.info.height}. Calibrate at the "
                    "capture resolution; live mode does not guess a field of view."
                )
            return scale_calibration(K, self.processing), provenance
        fov = float(self.config.hfov_deg)
        return (
            intrinsics_from_hfov(self.proc_w, self.proc_h, fov),
            "assumed_hfov_demo",
        )

    def _build_backends(self):
        missing = check_weights(self.weights, self.config.depth_variant)
        if missing:
            lines = [
                f"  - {item['model']} (expected at {item['expected_path']})"
                for item in missing
            ]
            raise RuntimeError(
                "real model weights are missing:\n"
                + "\n".join(lines)
                + "\nFetch them with: python scripts/fetch_models.py --yes"
            )
        depth_dir = (
            self.weights
            / f"depth-anything-v2-metric-{self.config.depth_variant}-small"
        )
        seg = SegFormerBackend(
            weights_dir=self.weights / "segformer-b0-ade",
            input_size=self.seg_input,
        )
        depth = DepthAnythingBackend(
            variant=self.config.depth_variant,
            weights_dir=depth_dir,
            input_size=self.depth_input,
        )
        detector = YoloDetectorBackend(weights=_yolo_weights(self.weights))
        if self.config.fusion_config_json:
            try:
                fusion = load_tuned_fusion_config(
                    self.config.fusion_config_json, allow_fake=False
                )
            except FusionConfigError as exc:
                raise ValueError(f"fusion config error: {exc}") from exc
            if fusion.enabled:
                sam = MobileSAMBackend(
                    checkpoint=self.weights / "mobile-sam" / "mobile_sam.pt"
                )
                errors = sam.preflight()
                if errors:
                    raise RuntimeError(
                        "tuned fusion is enabled but MobileSAM is not ready: "
                        + "; ".join(errors)
                    )
                seg = wrap_seg_with_fusion(seg, sam, fusion)
        return seg, depth, detector

    def _load_depth_interval(self):
        if not self.config.conformal_depth_json:
            return None
        try:
            return load_depth_interval(
                self.config.conformal_depth_json, allow_fake=False
            ).model
        except DepthIntervalConfigError as exc:
            raise ValueError(f"depth interval config error: {exc}") from exc

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _capture_camera_loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:
                    self._errors.append("camera returned no frame")
                    break
                frame = self._resize(frame)
                with self._cond:
                    self._frame = (self._capture_index + 1, time.monotonic(), frame)
                    self._capture_index += 1
                    self._cond.notify()
        finally:
            with self._cond:
                self._capture_done = True
                self._cond.notify_all()

    def _capture_file_loop(self) -> None:
        """Read a video in real time, keeping only the newest frame.

        Perception is slower than the video, so this reader stays on the wall
        clock and overwrites the pending frame instead of playing the file in
        slow motion. The analysis therefore refers to the current video time.
        """
        index = int(self.config.start_frame)
        frame_period = 1.0 / max(self.info.fps, 1e-3)
        next_ts = time.monotonic()
        try:
            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:
                    if self.config.loop:
                        self.source.rewind()
                        index = 0
                        next_ts = time.monotonic()
                        continue
                    break
                index += 1
                frame = self._resize(frame)
                with self._cond:
                    self._frame = (index, time.monotonic(), frame)
                    self._capture_index = index
                    self._cond.notify_all()
                next_ts += frame_period
                delay = next_ts - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_ts = time.monotonic()
        finally:
            with self._cond:
                self._capture_done = True
                self._cond.notify_all()

    def _perception_loop(self) -> None:
        try:
            while not self._stop.is_set():
                with self._cond:
                    while self._frame is None and not self._capture_done:
                        self._cond.wait(0.1)
                    item = self._frame
                    self._frame = None
                    if item is None:
                        break
                self._process_and_publish(item[0], item[1], item[2])
                if self._max_reached():
                    self._stop.set()
                    break
        finally:
            self._finished.set()

    def _process_file_loop(self) -> None:
        index = int(self.config.start_frame)
        frame_period = 1.0 / max(self.info.fps, 1e-3)
        try:
            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:
                    if self.config.loop:
                        self.source.rewind()
                        index = 0
                        continue
                    break
                index += 1
                started = time.monotonic()
                self._process_and_publish(index, started, self._resize(frame))
                if self._max_reached():
                    break
                if self.config.realtime:
                    elapsed = time.monotonic() - started
                    if elapsed < frame_period:
                        time.sleep(frame_period - elapsed)
        finally:
            self._finished.set()

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if (frame.shape[1], frame.shape[0]) == (self.proc_w, self.proc_h):
            return frame
        return cv2.resize(
            frame, (self.proc_w, self.proc_h), interpolation=cv2.INTER_AREA
        )

    def _max_reached(self) -> bool:
        limit = self.config.max_frames
        return limit is not None and self._processed >= int(limit)

    # ------------------------------------------------------------------
    # One processed frame
    # ------------------------------------------------------------------

    def _process_and_publish(self, index: int, capture_ts: float, frame_bgr: np.ndarray) -> None:
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs = self.pipeline.run(rgb)
        labels = None
        observations: list[ObjectAnchor] = []
        bev_grid = None
        path_cells: list[tuple[int, int]] = []
        target_cell = None
        guidance = None
        nearest: dict = {}
        plane_normal = None
        statuses = dict(outputs.statuses)
        with self._goal_lock:
            self._last_index = index
            goal_tracker = self._goal_tracker
            pending_events = self._pending_goal_events
            self._pending_goal_events = []
        for event_ts, event_kind, event_text in pending_events:
            self.db.log_event(self.session_id, index, event_ts, event_kind, event_text)
        goal_candidate: ObjectAnchor | None = None
        arrival_text: str | None = None

        if outputs.seg is not None and outputs.lift is not None:
            labels = np.asarray(outputs.seg.labels)
            certified, available = _certified_mask_for_free_space(
                labels, outputs.certified_final
            )
            statuses["certification"] = "OK" if available else "unavailable"
            vetoed = (
                None if outputs.vetoed is None else np.asarray(outputs.vetoed, dtype=bool)
            )
            bev = build_bev(
                pointmap_g=outputs.lift.point_map_g,
                labels=labels,
                certified_final=certified,
                vetoed=vetoed,
                pessimistic_pointmap=outputs.pessimistic_pointmap_g,
                cfg=self.map_cfg,
            )
            bev_grid = bev.grid
            nearest = nearest_obstacle_telemetry(
                outputs.lift.point_map_g,
                bev.obstacle_mask,
                lower_pointmap_g=outputs.pessimistic_pointmap_g,
                upper_pointmap_g=outputs.conformal_upper_pointmap_g,
            )

            tracked = assign_track_ids(
                list(outputs.detections), frame_bgr, self.tracker
            )
            observations = compute_object_anchors(
                tracked,
                outputs.lift.scaled_depth,
                self.K,
                self.mount,
                plane=outputs.lift.plane_fit,
            )

            costmap = build_costmap(bev.grid, cfg=self.map_cfg)
            start_c, start_r = self._start_cell(costmap)
            goal_unreachable = False
            if goal_tracker is not None and goal_tracker.active:
                goal_candidate = goal_tracker.select(observations)
                if goal_candidate is not None:
                    target_cell = find_anchor_target(
                        costmap, goal_candidate.position_g, (start_c, start_r)
                    )
                    goal_unreachable = target_cell is None
            if target_cell is None and not goal_unreachable:
                target_cell = find_explore_target(costmap, (start_c, start_r))
            path_res = astar_plan(
                costmap,
                (start_c, start_r),
                (start_c, start_r) if target_cell is None else target_cell,
            )
            path_cells = list(path_res.path)
            guidance = compute_guidance(
                costmap=costmap,
                path_result=path_res,
                points_g=outputs.lift.point_map_g,
                certified_floor_mask=certified,
                plane_inlier_ratio=outputs.lift.plane_fit.inlier_ratio,
                plane_status=outputs.lift.plane_fit.status.value,
                scale_status=outputs.lift.scale_result.status,
                nearest_obstacle=nearest,
            )
            if goal_unreachable:
                guidance.reasons.append("goal object is not reachable in this frame")
            plane_normal = self._plane_normal(outputs)
        else:
            statuses["frame"] = "no lift or segmentation; map unavailable"

        if goal_tracker is not None and goal_tracker.active:
            with self._goal_lock:
                if self._goal_tracker is goal_tracker:
                    arrival_text = goal_tracker.update(goal_candidate)
            if arrival_text is not None:
                self.db.log_event(
                    self.session_id, index, time.time(), "goal_arrival", arrival_text
                )
                print(f"  [Goal] {arrival_text}")
                self._speak_text(arrival_text, priority=True, bypass_cooldown=True)
        goal_line = self._goal_line(goal_tracker, goal_candidate)

        process_ms = (time.perf_counter() - started) * 1000.0
        processed_ts = time.time()
        decision = "NO_PERCEPTION" if guidance is None else guidance.decision
        heading = 0.0 if guidance is None else float(guidance.heading_deg)
        grid_counts = (
            _grid_counts(bev_grid)
            if bev_grid is not None
            else {"free": 0, "occupied": 0, "uncertain": 0, "unknown": 0}
        )
        reasons = [] if guidance is None else list(guidance.reasons)

        frame_id = self.db.log_frame(
            self.session_id,
            frame_index=index,
            capture_ts=time.time() - (time.monotonic() - float(capture_ts)),
            processed_ts=processed_ts,
            process_ms=process_ms,
            decision=decision,
            heading_deg=heading,
            nearest_nominal_m=nearest.get("distance_nominal_m"),
            nearest_lower_m=nearest.get("distance_lower_m"),
            nearest_bearing_deg=nearest.get("bearing_deg"),
            grid_counts=grid_counts,
            statuses=statuses,
            timings=outputs.timings,
            reasons=reasons,
        )
        self.db.log_objects(self.session_id, frame_id, index, observations)
        if bev_grid is not None and index % int(self.config.grid_every) == 0:
            self.db.save_bev(
                self.session_id,
                frame_id,
                index,
                processed_ts,
                self.map_cfg,
                bev_grid,
            )

        self._announce(guidance, observations, index)

        status_line = self._status_line(statuses, outputs)
        with self._result_lock:
            self._result_seq += 1
            self._result = FrameResult(
                seq=self._result_seq,
                frame_index=index,
                capture_ts=capture_ts,
                processed_ts=processed_ts,
                process_ms=process_ms,
                frame_bgr=frame_bgr,
                labels=labels,
                observations=observations,
                bev_grid=bev_grid,
                path_cells=path_cells,
                target_cell=target_cell,
                guidance=guidance,
                nearest=nearest,
                status_line=status_line,
                db_line=self.db_line,
                plane_normal=plane_normal,
                goal_line=goal_line,
                goal_candidate=goal_candidate,
                goal_complete=bool(
                    goal_tracker is not None and not goal_tracker.active
                ),
            )
            self._processed += 1

    def _start_cell(self, costmap) -> tuple[int, int]:
        start_c, start_r = world_to_cell(0.0, 0.0, self.map_cfg)
        if costmap.is_passable(start_r, start_c):
            return start_c, start_r
        for y_fwd in np.arange(0.1, 3.0, self.map_cfg.res):
            tc, tr = world_to_cell(0.0, float(y_fwd), self.map_cfg)
            if costmap.is_passable(tr, tc):
                return tc, tr
        return start_c, start_r

    def _plane_normal(self, outputs) -> np.ndarray | None:
        plane = outputs.lift.plane_fit
        normal = np.asarray(getattr(plane, "n", ()), dtype=np.float64)
        if normal.shape == (3,) and np.isfinite(normal).all():
            length = float(np.linalg.norm(normal))
            if length > 1e-9:
                return normal / length
        pitch = math.radians(float(self.mount.pitch_deg))
        return np.array([0.0, math.cos(pitch), math.sin(pitch)], dtype=np.float64)

    def _status_line(self, statuses: dict, outputs) -> str:
        parts = [
            f"seg {statuses.get('seg', '?')}",
            f"depth {statuses.get('depth', '?')}",
            f"det {statuses.get('detector', '?')}",
        ]
        if outputs.lift is not None:
            parts.append(f"plane {outputs.lift.plane_fit.status.value}")
            scale_status = getattr(
                outputs.lift.scale_result.status, "value", outputs.lift.scale_result.status
            )
            parts.append(f"scale {scale_status}")
        if statuses.get("certification") == "unavailable":
            parts.append("certification unavailable")
        parts.append(f"{self.proc_w}x{self.proc_h} {self.profile}")
        return "  |  ".join(parts)

    def _announce(self, guidance, observations: list[ObjectAnchor], index: int) -> None:
        now = time.monotonic()
        announcement = self.announcer.update(guidance, now)
        if (
            announcement is None
            and self.config.speak_objects
            and (guidance is None or guidance.decision not in ("STOP_UNCLEAR", "STOP_BLOCKED"))
        ):
            announcement = self.announcer.object_alert(observations, now)
        if announcement is None:
            return
        self.db.log_event(
            self.session_id, index, time.time(), announcement.kind, announcement.text
        )
        self._speak(announcement)

    def _speak(self, announcement) -> None:
        if self.speech is None or not self._speech_enabled:
            return
        try:
            self.speech.speak(
                announcement.text,
                priority=bool(announcement.priority),
                bypass_cooldown=bool(announcement.emergency),
                emergency=bool(announcement.emergency),
            )
        except Exception as exc:  # noqa: BLE001 - TTS must never kill perception
            self._errors.append(f"speech: {type(exc).__name__}: {exc}")

    def _speak_text(
        self,
        text: str,
        *,
        priority: bool = True,
        bypass_cooldown: bool = True,
        emergency: bool = False,
    ) -> None:
        if not text or self.speech is None or not self._speech_enabled:
            return
        try:
            self.speech.speak(
                text,
                priority=bool(priority),
                bypass_cooldown=bool(bypass_cooldown),
                emergency=bool(emergency),
            )
        except Exception as exc:  # noqa: BLE001 - TTS must never kill perception
            self._errors.append(f"speech: {type(exc).__name__}: {exc}")

    def _goal_line(
        self,
        tracker: GoalTracker | None,
        candidate: ObjectAnchor | None,
    ) -> str:
        if tracker is None:
            return ""
        if not tracker.active:
            return f"goal {tracker.spec.intent}: reached"
        if candidate is not None:
            return (
                f"goal {tracker.spec.intent}: "
                f"{candidate.class_name} {candidate.distance_m:.1f} m"
            )
        return f"goal {tracker.spec.intent}: searching {tracker.spec.candidates_text}"

    def _start_voice(self) -> None:
        if not self.config.voice or self.config.headless:
            return
        try:
            from live.voice import PushToTalkVoice
        except Exception as exc:  # noqa: BLE001
            print(f"  [Voice] Unavailable: {type(exc).__name__}: {exc}")
            return
        self._voice = PushToTalkVoice(
            on_transcript=self._on_voice_transcript,
            on_listening=self._on_voice_listening,
            on_error=self._on_voice_error,
        )
        self._voice_ready = bool(self._voice.start())
        if self._voice_ready:
            self._speak_text(
                "Voice ready. Hold S, say your goal, then release.",
                priority=True,
                bypass_cooldown=True,
            )
        else:
            print('  [Voice] You can still set a goal with --goal "I want to sit".')
        if self._goal_agent is not None and self._goal_agent.available:
            print(
                f"  [Goal] Voice understanding: Gemini ({self._goal_agent.model}), "
                "local parser fallback."
            )
        else:
            print(
                "  [Goal] Voice understanding: local parser "
                "(set GEMINI_API_KEY to use Gemini Flash)."
            )

    def _on_voice_transcript(self, text: str) -> None:
        transcript = " ".join(str(text or "").split()).strip()
        if not transcript:
            return
        if is_cancel_goal(transcript):
            self._cancel_goal(transcript)
            return
        spec, source = self._interpret_goal(transcript)
        if spec is None:
            self._speak_text(
                "I did not catch a goal. Hold S and try again.",
                priority=True,
                bypass_cooldown=True,
            )
            return
        if spec.intent == "cancel":
            self._cancel_goal(transcript)
            return
        tracker = GoalTracker(
            spec,
            arrival_distance_m=self.config.arrival_distance_m,
            stable_frames=self.config.arrival_stable_frames,
        )
        with self._goal_lock:
            self._goal_tracker = tracker
        self._queue_goal_event(
            "goal_set", f"{transcript} -> {spec.candidates_text} [{source}]"
        )
        print(f"  [Goal] Active ({source}): {spec.intent} -> {spec.candidates_text}")
        self._speak_text(
            f"Goal set. Looking for {spec.candidates_text}.",
            priority=True,
            bypass_cooldown=True,
        )

    def _interpret_goal(self, transcript: str) -> tuple[GoalSpec | None, str]:
        agent = self._goal_agent
        if agent is not None and agent.available:
            spec = agent.interpret(transcript)
            if spec is not None:
                return spec, "gemini"
            reason = agent.last_error or "no response"
            self._errors.append(f"goal agent: {reason}")
            print(f"  [Goal] Gemini unavailable ({reason}); using local parser.")
        try:
            return parse_goal_text(transcript), "local"
        except ValueError:
            return None, "none"

    def _cancel_goal(self, transcript: str) -> None:
        with self._goal_lock:
            self._goal_tracker = None
        self._queue_goal_event("goal_cancelled", transcript)
        print("  [Goal] Cancelled.")
        self._speak_text(
            "Goal cancelled. Continue exploring.",
            priority=True,
            bypass_cooldown=True,
        )

    def _queue_goal_event(self, kind: str, text: str) -> None:
        with self._goal_lock:
            self._pending_goal_events.append((time.time(), str(kind), str(text)))

    def _on_voice_listening(self, active: bool) -> None:
        self._voice_listening = bool(active)
        if self.speech is None or not self._speech_enabled:
            return
        try:
            if active:
                self.speech.duck()
            else:
                self.speech.unduck()
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"speech duck: {type(exc).__name__}: {exc}")

    def _on_voice_error(self, message: str) -> None:
        now = time.monotonic()
        last = self._last_voice_error_s.get(message, 0.0)
        if now - last < 10.0:  # TO TUNE ON REAL DATA
            return
        self._last_voice_error_s[message] = now
        self._errors.append(f"voice: {message}")
        self._speak_text(message, priority=True, bypass_cooldown=True)

    def _voice_unavailable_hint(self) -> None:
        if self._voice_hint_given:
            return
        self._voice_hint_given = True
        if self._voice is None:
            print(
                "  [Voice] Voice is disabled for this run. "
                'Start with --goal "I want to sit" to set one.'
            )
            self._speak_text(
                "Voice input is off. You can set a goal with the goal option.",
                priority=True,
                bypass_cooldown=True,
            )
            return
        detail = self._voice.error or "voice is unavailable"
        print(
            f"  [Voice] {detail}. Install pyaudio, SpeechRecognition and pynput, "
            'or start with --goal "I want to sit".'
        )
        self._speak_text(
            "Voice input is unavailable. You can set a goal with the goal option.",
            priority=True,
            bypass_cooldown=True,
        )

    def _announce_initial_goal(self) -> None:
        with self._goal_lock:
            tracker = self._goal_tracker
        if tracker is None:
            return
        self.db.log_event(
            self.session_id,
            self._last_index,
            time.time(),
            "goal_set",
            f"{tracker.spec.text} -> {tracker.spec.candidates_text}",
        )
        self._speak_text(
            f"Goal set. Looking for {tracker.spec.candidates_text}.",
            priority=True,
            bypass_cooldown=True,
        )

    # ------------------------------------------------------------------
    # Public run loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        self._started_s = time.monotonic()
        self._capture_index = int(self.config.start_frame)
        if self.config.out_video:
            out_path = Path(self.config.out_video)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_path),
                fourcc,
                float(max(self.config.video_fps, 1.0)),
                (self.proc_w + 320, self.proc_h),
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not open output video: {out_path}")
            self._writer = writer
        self._print_header()
        self._start_voice()
        self._announce_initial_goal()

        if self.info.is_camera:
            self._threads.append(
                threading.Thread(target=self._capture_camera_loop, daemon=True)
            )
            self._threads.append(
                threading.Thread(target=self._perception_loop, daemon=True)
            )
        elif self.config.realtime:
            self._threads.append(
                threading.Thread(target=self._capture_file_loop, daemon=True)
            )
            self._threads.append(
                threading.Thread(target=self._perception_loop, daemon=True)
            )
        else:
            self._threads.append(
                threading.Thread(target=self._process_file_loop, daemon=True)
            )
        for thread in self._threads:
            thread.start()
        summary: dict = {}
        try:
            if self.config.headless:
                self._headless_loop()
            else:
                self._display_loop()
        finally:
            summary = self.summary()
            self._close()
        return summary

    def _print_header(self) -> None:
        print("Visiona3D live - RGB only")
        print(f"  source:      {self.info.label} ({self.info.width}x{self.info.height} @ {self.info.fps:.1f} fps)")
        print(f"  processing:  {self.proc_w}x{self.proc_h} ({self.profile}, seg {self.seg_input}, depth {self.depth_input})")
        print(f"  intrinsics:  {self.intrinsics_provenance}")
        print(f"  mount:       height {self.mount.h_known:.2f} m, pitch {self.mount.pitch_deg:.1f} deg (TO TUNE ON REAL DATA)")
        print(f"  capture:     {cv2_backend_note()}")
        if self.info.is_camera:
            playback = "newest frame only; analysis is as fresh as the CPU allows"
        elif self.config.realtime:
            playback = "real time; frames the CPU cannot process are skipped"
        else:
            playback = "every frame, no pacing (runs slower than real time)"
        print(f"  playback:    {playback}")
        print(f"  database:    {self.db.path if self.db.enabled else 'off'}")
        if self.session_id:
            print(f"  session:     {self.session_id}")
        if self.uncalibrated:
            print("  WARNING:     intrinsics are not a physical-camera measurement; metric distances are estimates.")
        with self._goal_lock:
            tracker = self._goal_tracker
        if tracker is not None:
            print(
                f"  goal:        {tracker.spec.intent} -> "
                f"{tracker.spec.candidates_text} "
                f"(arrival {tracker.arrival_distance_m:.1f} m, TO TUNE ON REAL DATA)"
            )
        keys = "  [q] quit"
        if self.config.voice and not self.config.headless:
            keys += "   [s] hold to speak, release to set a goal"
        keys += "   [m] mute speech"
        print(keys + "\n")

    def _headless_loop(self) -> None:
        while not self._finished.is_set() or self._has_new_result():
            result = self._latest_new_result()
            if result is None:
                if self._stop.is_set():
                    break
                time.sleep(0.05)
                continue
            canvas = self._render(result)
            if self._writer is not None:
                self._writer.write(canvas)
            if self._processed % 5 == 0:
                decision = "NO_PERCEPTION" if result.guidance is None else result.guidance.decision
                print(
                    f"  frame {result.frame_index}: {result.process_ms:.0f} ms, "
                    f"{decision}, {len(result.observations)} objects"
                )
            if self._max_reached():
                break
        deadline = time.monotonic() + 5.0
        while self._finished.is_set() is False and time.monotonic() < deadline:
            time.sleep(0.05)

    def _display_loop(self) -> None:
        window = "Visiona3D Live (RGB)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            result = self._latest_result()
            if result is None:
                canvas = hud.loading_frame(
                    self.proc_w + 320,
                    self.proc_h,
                    "Loading models / waiting for first frame...",
                )
            else:
                canvas = self._render(result)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                self._toggle_speech()
            elif key == ord("s") and not self._voice_ready:
                self._voice_unavailable_hint()
            if result is not None:
                self._last_written_seq = result.seq
            if self._finished.is_set():
                if result is None or not self._has_new_result():
                    break
        cv2.destroyWindow(window)

    def _render(self, result: FrameResult) -> np.ndarray:
        view = hud.blend_segmentation(result.frame_bgr, result.labels)
        view = hud.draw_objects(
            view,
            result.observations,
            K=self.K,
            plane_normal=result.plane_normal,
            mount_height_m=self.mount.h_known,
        )
        panel = hud.render_bev_panel(
            result.bev_grid,
            path_cells=result.path_cells,
            target_cell=result.target_cell,
            cfg=self.map_cfg,
            size=(320, self.proc_h),
        )
        return hud.compose_hud(
            view,
            guidance=result.guidance,
            observations=result.observations,
            nearest=result.nearest,
            bev_panel=panel,
            status_line=result.status_line,
            goal_line=result.goal_line,
            frame_age_s=time.monotonic() - result.capture_ts,
            process_ms=result.process_ms,
            fps=self._fps(),
            db_line=result.db_line,
            uncalibrated=self.uncalibrated,
            profile=self.profile,
        )

    def _fps(self) -> float:
        elapsed = time.monotonic() - self._started_s
        return 0.0 if elapsed <= 0 else self._processed / elapsed

    def _latest_result(self) -> FrameResult | None:
        with self._result_lock:
            return self._result

    def _latest_new_result(self) -> FrameResult | None:
        with self._result_lock:
            result = self._result
            if result is None or result.seq == self._last_written_seq:
                return None
            self._last_written_seq = result.seq
            return result

    def _has_new_result(self) -> bool:
        with self._result_lock:
            return self._result is not None and self._result.seq != self._last_written_seq

    def _toggle_speech(self) -> None:
        self._speech_enabled = not self._speech_enabled
        if self.speech is not None:
            try:
                if self._speech_enabled:
                    if not self._voice_listening:
                        self.speech.unduck()
                else:
                    self.speech.duck()
            except Exception:  # noqa: BLE001
                pass
        print(f"  speech: {'on' if self._speech_enabled else 'off'}")

    def _close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        voice = self._voice
        self._voice = None
        self._voice_ready = False
        self._voice_listening = False
        if voice is not None:
            try:
                voice.stop()
            except Exception as exc:  # noqa: BLE001
                self._errors.append(f"voice stop: {type(exc).__name__}: {exc}")
        for thread in self._threads:
            thread.join(timeout=10.0)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.speech is not None:
            try:
                self.speech.stop()
            except Exception:  # noqa: BLE001
                pass
        with self._goal_lock:
            pending_events = self._pending_goal_events
            self._pending_goal_events = []
            last_index = self._last_index
        for event_ts, event_kind, event_text in pending_events:
            self.db.log_event(
                self.session_id, last_index, event_ts, event_kind, event_text
            )
        self.source.release()
        self.db.close()

    def summary(self) -> dict:
        db_summary = (
            self.db.session_summary(self.session_id)
            if self.session_id
            else {}
        )
        with self._goal_lock:
            tracker = self._goal_tracker
        goal = (
            None
            if tracker is None
            else {
                "intent": tracker.spec.intent,
                "candidates": list(tracker.spec.candidates),
                "text": tracker.spec.text,
                "complete": not tracker.active,
                "arrival_distance_m": tracker.arrival_distance_m,
                "arrival_stable_frames": tracker.stable_frames,
            }
        )
        return {
            "session_id": self.session_id,
            "database": str(self.db.path) if self.db.path else None,
            "source": self.info.label,
            "capture_size": [self.info.width, self.info.height],
            "processing_size": [self.proc_w, self.proc_h],
            "profile": self.profile,
            "goal": goal,
            "goal_interpreter": (
                f"gemini:{self._goal_agent.model}"
                if self._goal_agent is not None and self._goal_agent.available
                else "local"
            ),
            "intrinsics_provenance": self.intrinsics_provenance,
            "uncalibrated": self.uncalibrated,
            "processed": self._processed,
            "wall_seconds": round(time.monotonic() - self._started_s, 2),
            "decisions": db_summary.get("decisions", {}),
            "objects": db_summary.get("objects", 0),
            "avg_process_ms": round(db_summary.get("avg_process_ms", 0.0), 1),
            "errors": list(self._errors),
        }


__all__ = ["FrameResult", "LiveConfig", "LiveRunner", "PROFILE_SIZES", "PROFILES"]
