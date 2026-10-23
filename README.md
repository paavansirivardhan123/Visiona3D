# Visiona-Walk

**An RGB-only wearable navigation aid for blind and low-vision people.**

Visiona-Walk turns one camera feed into certified free space, metric 3D object
positions, a bird's-eye occupancy grid, an A* path and spoken guidance. It is a
single system: capture, segmentation, depth, tracking, mapping, planning,
guidance, persistence and speech run in one loop and share one frame.

> Every numeric default is `TO TUNE ON REAL DATA`. Nothing here is a validated
> accuracy or safety claim. The verified runs used route video with an assumed
> field of view, not a physically calibrated camera or blind participants.

---

## What it does

```
camera/video -> SegFormer floor/wall/obstacle masks
             -> Depth Anything V2 metric depth
             -> YOLO11n detections + track IDs
             -> 3D ground-plane anchors per object
             -> certified BEV grid (75 x 60 @ 0.1 m)
             -> conformal obstacle bounds (optional)
             -> A* path + explore/anchor target
             -> GUIDE / CAUTION / STOP guidance
             -> HUD + SQLite session DB + gated speech
```

- **Floor and walls:** SegFormer-B0 (ADE20K) merged to floor/wall/obstacle/other;
  optional MobileSAM refinement. Sky can never become floor.
- **Objects:** YOLO boxes with track ID, class, confidence, metric distance,
  lateral offset and a projected 3D anchor.
- **Map:** free space must be certified. Unknown and uncertain cells are never
  traversable.
- **Guidance:** A* over a chamfer costmap; STOP_UNCLEAR when the system cannot
  certify the floor, STOP_BLOCKED when no path exists.
- **Persistence:** every session, frame, object anchor, BEV snapshot and
  announcement is stored in `database/visiona_live.db` (SQLite, stdlib only).
- **Speech:** priority-gated announcements with cooldowns; STOP overrides.

---

## Quick start

Requirements: Python 3.10+, Windows/Linux, a webcam or a video file. CPU-only
works; a GPU is not required.

```powershell
cd E:\git_projects\Projects\Visiona3D
.\.venv\Scripts\Activate.ps1
```

Dependencies are pinned in `pyproject.toml`. If the venv is not ready:

```powershell
python -m pip install uv
uv sync
```

Fetch the model weights once (no network after that):

```powershell
python scripts\fetch_models.py --dry-run
python scripts\fetch_models.py --yes
```

### Run on a video file (works immediately)

```powershell
python scripts\live_demo.py --source sample-vid\FRONT\VID2.mp4 --hfov 70 --profile fast --max-frames 4 --headless --out-video reports\live_demo\live.mp4 --no-speech --summary-json reports\live_demo\summary.json
```

### Run on a webcam (demo intrinsics, distances are estimates)

```powershell
python scripts\live_demo.py --source 0 --hfov 70
```

`q` quits, `s` toggles speech. The red HUD banner means the field of view is
assumed, not measured.

### Run with calibrated intrinsics (metric mode)

```powershell
python scripts\capture_calibration.py --camera 0 --out calib --board 9x6 --views 15
python scripts\calibrate_camera.py --images calib --board 9x6 --square 0.018 --out reports\camera_intrinsics.json
python scripts\live_demo.py --source 0 --intrinsics reports\camera_intrinsics.json
```

Capture at the resolution you will run at. Intrinsics for a different
resolution are rejected; the runner never rescales a guessed field of view.

---

## What you see

| Region | Contents |
| --- | --- |
| Camera view | 45% blended floor/wall/obstacle masks, YOLO boxes with `#track class conf distance lateral`, projected 3D anchor points |
| Top bar | per-stage status (seg, depth, det, plane, scale, certification), processing size and profile |
| Bottom bar | decision, heading, certified nearest obstacle with bearing and optional lower bound, frame age, process time, FPS, DB session |
| Right panel | 75 x 60 BEV, forward up: free / occupied / uncertain / unknown, A* path, target cross, robot triangle |

The HUD draws only on the exact frame that produced the masks and anchors. It
shows frame age so latency is visible instead of hidden.

---

## Project layout

One repository, one entry point (`scripts/live_demo.py`), one database.

```
Visiona3D/
  scripts/live_demo.py       product entry point: live/video -> HUD -> DB -> speech
  live/                      runner, synchronisation, HUD, anchors, SQLite, speech bridge
  perception/                SegFormer, Depth Anything V2, YOLO, pipeline, fusion, tracker
  geometry/                  camera model, frames, scale, lifting
  calibration/               intrinsics, abstention, conformal depth config
  mapping/                   BEV grid, certified floor, obstacles, rolling map
  planning/                  costmap, A*, targets, smoothing, guidance
  audio/                     speech engine and priority queue
  core/                      config, detection types, memory
  agents/, services/         legacy assistant modules used by the old app
  kinematics/, evaluation/
  dataset/, audit/
  scripts/                   fetch, smoke, evaluation, calibration, demos
  tests/                     offline tests (numpy/pytest/stdlib only)
  docs/                      component and operating documentation
  main.py                    LEGACY app (original YOLO/TTS loop), kept read-only
  weights/, sample-vid/,     local only, gitignored
  data/, calib/
  database/, logs/, reports/ runtime artifacts, gitignored and safe to delete
```

The old `main.py` app is retained as a read-only reference and shares `core/`,
`perception/` and `audio/` with the live system. It is not the product entry
point.

---

## Configuration

```powershell
--profile fast|balanced|quality     model input sizes (not safety rules)
--proc-width 640                    processing width cap
--goal-class chair                  route to the nearest detected object instead
--fusion-config path.json           optional SegFormer + MobileSAM tuning
--conformal-depth path.json         optional pessimistic obstacle bounds
--no-speech --no-speak-objects      mute audio (events are still logged)
--no-db                             disable persistence
--headless --out-video out.mp4      write the annotated HUD instead of a window
--start-frame 100 --max-frames 20 --loop --no-realtime
--camera-width 1280 --camera-height 720
```

Mount defaults (`--mount-height 1.4`, `--pitch 15`) and `--floor-threshold 0.90`
are `TO TUNE ON REAL DATA`.

---

## Persistence

```powershell
python -c "import sqlite3;c=sqlite3.connect('database/visiona_live.db');print(c.execute('select class_name,round(distance_m,2),round(anchor_x,2),round(anchor_y,2),round(anchor_z,2) from object_registry order by distance_m').fetchall())"
```

Tables: `sessions`, `frames`, `objects`, `object_registry`, `bev_snapshots`,
`events`. The schema version is checked on open; a mismatch raises instead of
silently reading old columns.

---

## Performance (measured, not promised)

On the bundled 1920x1080 route video at 640x360, CPU only (12 logical cores,
torch 10 threads, no CUDA), warm frames:

| Stage | Time |
| --- | --- |
| SegFormer (256 input) | 0.16 - 0.26 s |
| Depth Anything V2 (266 input) | 0.49 - 0.80 s |
| YOLO11n | 0.13 - 0.17 s |
| ground lift | 0.41 - 0.57 s |
| **total per processed frame** | **1.23 - 2.45 s** |

That is roughly 0.5-0.8 HUD updates per second. The display stays responsive and
shows the latest processed frame; stale frames are dropped, never queued. To go
faster: `--proc-width 480 --profile fast`, or run the backends on a GPU. The
first frame additionally includes model loading.

---

## Safety guarantees

1. Free space must be certified; unknown/uncertain cells are not traversable.
2. Obstacles use the conformal lower-bound distance when an interval artifact is
   supplied; otherwise no lower bound is invented.
3. Every failed stage becomes a status and a STOP/NO_PERCEPTION decision, not an
   exception in the loop.
4. The planner recommends; it never claims that a path is safe.
5. Uncalibrated intrinsics are labelled in the HUD and in the session row.

---

## Tests and verification

No model, camera, network or TTS engine is used in tests.

```powershell
python -m pytest tests audio -q
python geometry\scripts\run_checks.py
```

Latest full run: `294 passed, 1 warning, 14 subtests passed` and all geometry,
perception, mapping and planning checks passed. Synthetic results are regenerated
by `python scripts\run_planner_eval.py --scenes 200 --seed 42`; report artifacts
are local and not committed.

---

## Limitations

- Not field-tested on a physical camera or with blind users.
- 2.5D grid, not a mesh: BEV plus one 3D anchor per object.
- No odometry fusion in the live loop; each frame produces its own grid.
- Monocular depth bias is uncorrected.
- CPU-only HUD rate is about 0.5-0.8 Hz on the test machine.
- No regulatory or safety certification.

---

## Documentation

- [`docs/live_system.md`](docs/live_system.md) - operating guide, synchronisation,
  database schema, measured latency, limitations.
- [`docs/perception_backends.md`](docs/perception_backends.md) - models, fusion
  rules, evaluation and tuning.
- [`docs/mapping_planning.md`](docs/mapping_planning.md) - grid conventions,
  certification, planner and guidance.
- [`docs/annotation_guide.md`](docs/annotation_guide.md) and
  [`docs/capture_protocol.md`](docs/capture_protocol.md) - data collection.

## Legacy app

`main.py` is the original live app (YOLO + MiDaS + TTS + LangChain assistant).
It still runs, but it is not connected to the safety stack and is kept only as a
reference:

```powershell
python main.py
```

Its optional `GROQ_API_KEY` in `.env` is not required by the live system.

---

**Author:** Paavan Siri Vardhan Narava
