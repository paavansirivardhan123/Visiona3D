# Live RGB System (connected prototype)

One process turns a webcam or video file into a single real-time HUD:

```
frame -> SegFormer floor/wall/obstacle masks -> metric depth -> 3D object anchors
      -> certified BEV grid (0.1 m) -> A* path -> guidance -> HUD + SQLite + speech
```

The safety stack (segmentation, metric depth, BEV mapping, A*, guidance and
conformal obstacle bounds) and the live loop (capture, YOLO tracking, HUD and
TTS) are one codebase. The runner owns the loop and calls the safety code in
the same process and on the same frame.

> **Provenance.** Every default below is `TO TUNE ON REAL DATA`. Nothing in
> this document is a validated accuracy or safety claim. All numbers under
> *Measured latency* are wall-clock timings from one CPU-only machine, not
> guarantees.

## Run it

```powershell
# Public entry point: choose webcam or video from the menu
python starter.py

# Real camera, calibrated intrinsics (the only measured mode)
python starter.py --source 0 --intrinsics reports/camera_intrinsics.json

# Video file, deterministic headless run with an annotated MP4
python starter.py --source sample-vid/FRONT/VID2.mp4 --hfov 70 --profile fast --start-frame 100 --max-frames 4 --headless --out-video reports/live_demo/live.mp4 --no-speech --summary-json reports/live_demo/summary.json

# Expand "I want to sit" to chair/bench/couch/bed and guide to the nearest one
python starter.py --source 0 --intrinsics camera.json --goal "I want to sit"

# The same goal on a video, without a microphone
python starter.py --source sample-vid/FRONT/VID2.mp4 --hfov 70 --goal "I want to sit" --headless --max-frames 20

# Legacy single-class goal
python starter.py --source 0 --intrinsics camera.json --goal-class chair
```

`--hfov` is an explicit demo fallback for an uncalibrated camera. It sets
`intrinsics_provenance=assumed_hfov_demo`, shows a persistent red banner in the
HUD, and makes every metric distance an estimate. Without `--intrinsics` and
without `--hfov` the CLI exits with code 2: field of view is never guessed.

### Capture calibration

```powershell
python scripts/capture_calibration.py --camera 0 --out calib --board 9x6 --views 15
python scripts/calibrate_camera.py --images calib --board 9x6 --square 0.018 --out reports/camera_intrinsics.json
```

The intrinsics must describe the camera's capture resolution. The runner
rejects a mismatch instead of rescaling a guess; processing-resolution
intrinsics are derived with `geometry.camera.scale_intrinsics` after an aspect
check (`live/imaging.py`).

## Synchronisation model

The HUD can only be misaligned if masks are drawn on a different frame than
the one they were computed from. The runner prevents that by construction:

- **Camera mode:** a capture thread stores only the newest frame. The
  perception thread takes it and sets the slot back to empty. If perception is
  slower than the camera, stale frames are dropped, never queued.
- **File mode (default, `realtime`):** a reader thread stays on the wall clock
  at the file's native FPS and overwrites the pending frame, so a slow CPU
  skips frames instead of replaying the video in slow motion. The analysis and
  any spoken response therefore refer to the current video time, not to a
  backlog. `--no-realtime` reads every frame back-to-back (exhaustive, but far
  slower than real time); `--start-frame`, `--loop` and `--max-frames` apply in
  both modes (`--max-frames` counts processed frames).
- **One frame per result:** `FrameResult` carries the exact BGR frame that
  produced the labels, anchors and BEV grid. The renderer never combines a new
  frame with old masks.
- **Latency is visible:** the HUD prints `frame age` (capture to display),
  `process` time, measured FPS and the active profile. On CPU the age is
  typically above one second; this is shown, not hidden.

## Profiles and measured latency

`--profile` changes only model input sizes, never a safety rule:

| profile | SegFormer input | depth input | measured process/frame (warm) |
| --- | --- | --- | --- |
| `fast` (default) | 256 | 266 (aspect-preserving) | 1.23 - 2.45 s |
| `balanced` | 320 | 364 | not measured on this machine |
| `quality` | 512 | 518 | not measured on this machine |

Measured on the bundled 1920x1080 / 25 fps route video at 640x360 processing,
CPU only (12 logical cores, torch 10 threads, no CUDA), after the first
frame (model load) was excluded:

```
frame 102: 1535 ms (seg 263, depth 712, det 173, lift 415)
frame 103: 1308 ms (seg 163, depth 490, det 127, lift 570)
frame 104: 1229 ms (seg 164, depth 804, det 161, lift 479)
```

SegFormer's 150-class softmax was previously computed at full processing
resolution and cost ~2 s/frame. It is now merged to the 4 classes at the
model's low output resolution and only those 4 channels are upsampled.
Depth output is still mapped back to the full frame with bilinear
interpolation, and geometry is unchanged.

A webcam will therefore update the HUD at roughly 0.5-0.8 Hz on this class of
CPU. Capture/display stay responsive and always show the latest processed
frame; they do not freeze. Lower `--proc-width`, a smaller `--profile`, or a
GPU (`device=` on the backends) raise the rate. No timing here is a promise.

## What is drawn

- **Segmentation overlay:** floor (green), wall (blue), obstacle (red) blended
  at 45% on the processed frame. Sky is never floor.
- **Object notation:** YOLO boxes with `#track class conf distance lateral`,
  colour-coded by distance (<1.5 m red, <3 m amber), plus the projected 3D
  ground-plane anchor as a filled circle.
- **BEV panel:** 75 x 60 cells at 0.1 m, forward is up. Free, occupied,
  uncertain and unknown are distinct; A* path in orange, target cross in
  magenta, robot triangle at the origin. Unknown/uncertain cells are never
  traversable.
- **Guidance strip:** GUIDE / CAUTION / STOP_UNCLEAR / STOP_BLOCKED with
  heading, reasons and the certified nearest-obstacle distance and bearing
  (lower bound when a conformal depth interval is supplied).
- **Goal strip:** the active intent, the class currently targeted (with its
  depth) or the full candidate search list, and `reached` after arrival.

## Goals, voice input and arrival

```text
hold S -> 16 kHz audio -> Google STT -> Gemini Flash (optional) or local parser
       -> candidate classes -> nearest usable 3D anchor -> A* target
       -> stable arrival check -> one priority announcement
```

- **Push-to-talk:** `live/voice.py` pre-warms PyAudio and a pynput global key
  listener. Hold `S` or `V`; release to transcribe. Speech is ducked while the
  key is held so the microphone does not hear TTS. `--no-voice` disables the
  microphone and listener. Headless runs never open either.
- **Intent expansion:** `live/goals.py` maps intents to detector classes
  ("sit" -> chair, bench, couch, bed; "water" -> bottle, cup, wine glass, sink,
  refrigerator). Synonyms are canonicalised (`sofa` -> `couch`,
  `fridge` -> `refrigerator`). Unknown goals keep the cleaned phrase and its
  head noun.
- **Unknown distances:** candidates that are not currently detected are simply
  absent. Conceptually they carry the legacy `999 m` unknown sentinel, so they
  are never selected, never become a target and never trigger arrival.
- **Nearest candidate:** every frame ranks usable anchors (status `OK`, finite
  position and positive depth) by distance across all goal classes. Ties use
  the intent's class order, then track id. The chosen anchor is passed to
  `find_anchor_target`; the normal costmap and A* safety rules are unchanged.
- **Arrival:** the same track must stay within `--arrival-distance` (default
  1.5 m) for `--arrival-frames` consecutive processed frames (default 3), both
  `TO TUNE ON REAL DATA`. Then the priority message "You have reached your goal
  destination. The {class} is {distance} meters {direction}." is spoken once,
  logged as `goal_arrival`, and the goal is marked complete. Missing frames,
  farther frames or a different track reset the streak.
- **Optional Gemini Flash understanding:** `live/goal_agent.py` sends the
  transcript to `gemini-2.5-flash` when `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
  is set. `--goal-model` or `GEMINI_MODEL` overrides the model. The request uses
  a JSON response schema and the reply is validated locally: only COCO-80
  detector names are accepted, synonyms are canonicalised, and duplicates or
  unknown classes are dropped. No key, HTTP error, timeout, blocked reply,
  invalid JSON or empty class list falls back to `parse_goal_text`;
  `--no-voice-agent` forces the local path. No external agent framework is used.
- **No LLM is required:** the local parser remains the default and the
  fallback. It needs no network, key or account. Only Google speech-to-text
  always needs a connection; Gemini needs one only when enabled. Missing
  packages, a busy microphone or an STT failure produce a spoken/printed
  status; the navigation loop keeps running.

## Persistence

SQLite (`database/visiona_live.db` by default, stdlib only) stores:

| table | contents |
| --- | --- |
| `sessions` | source, capture/processing size, intrinsics + provenance, mount, profile, config |
| `frames` | per-frame timings, decision, heading, nearest obstacle, grid counts, per-stage status, reasons |
| `objects` | per-frame 3D anchors for every tracked detection (class, confidence, distance, x/y/z, box) |
| `object_registry` | one row per `(session, track_id)`: first/last frame, latest distance and anchor, observation count |
| `bev_snapshots` | zlib-compressed occupancy grid per stored frame plus metric cell geometry |
| `events` | every announcement the gating layer produced (also when speech is muted), plus `goal_set`, `goal_cancelled` and `goal_arrival` |

The schema version is stored in `meta`; a mismatch raises instead of silently
reading old columns. `--no-db` disables all writes.

```powershell
sqlite3 database/visiona_live.db "SELECT track_id, class_name, distance_m, anchor_x, anchor_y, anchor_z FROM object_registry ORDER BY distance_m;"
```

Example from the verified run (4 frames, 8 tracks):

```
5  handbag  1.98  -1.23  1.83  0.63
1  person   1.98  -1.26  1.83  0.63
2  person   3.98  -1.41  3.99  0.71
...
```

## Speech

`live/guidance_bridge.py` converts guidance into announcements with separate
cooldowns and thresholds (`TO TUNE ON REAL DATA`): STOP overrides everything,
then CAUTION, then GUIDE, then per-object proximity alerts. Repeats are
suppressed until the heading changes by 15 deg or the distance by 0.5 m.
STOP decisions suppress object chatter. Speech failures are recorded and never
stop perception. `--no-speech` and `--no-speak-objects` only mute audio; events
are still logged. `m` toggles mute; `s` is reserved for push-to-talk when voice
is enabled. Goal confirmations and arrivals use the priority queue, so they are
not dropped behind navigation chatter.

Voice recognition sends only the held-key clip to Google STT. If
`GEMINI_API_KEY` is configured, only the transcript is sent to Gemini Flash;
frames, maps and database rows never leave the machine. Both are optional and
were not exercised by the offline test suite (no microphone, recognizer,
network or API key in tests).

## Safety status of this integration

- Certified floor only; unknown/uncertain is not free; sky cannot be floor.
- Obstacles are placed at the pessimistic lower-bound distance when a
  conformal depth interval is configured; otherwise no lower bound is invented
  and the field is absent.
- Any missing model, calibration mismatch or failed stage becomes a status
  string and a NO_PERCEPTION/STOP decision, not an exception in the loop.
- Planning is a recommendation. The system never states that a path is safe.

## Known limitations

- **2.5D, not a mesh.** The "3D object grid" is a metric ground-frame BEV
  (75 x 60 at 0.1 m) plus one 3D anchor per detection. There are no voxels or
  object meshes.
- **CPU-only latency.** ~1.2-2.5 s per processed frame on the test machine.
  The HUD is live but not video-rate.
- **Uncalibrated demo intrinsics.** Distances from `--hfov` are estimates and
  are labelled as such in the HUD and the session row.
- **No ego-motion yet.** Each frame produces its own grid; the rolling map in
  `mapping/rolling.py` is not wired into the live loop, so BEV snapshots are
  per-frame, not accumulated with odometry.
- **Monocular depth bias.** Metric depth from an indoor Small checkpoint is
  used as-is; no per-camera affine calibration has been validated.
- **Camera mode not exercised here.** The verified run used a video file; a
  physical RGB webcam still needs a real calibration and field test.
- **Voice goals are not field-tested.** Push-to-talk, Google STT, microphone
  permissions, the live Gemini API and real-room arrival distances still need
  testing on the target laptop. Only the deterministic parser, the Gemini
  request/validation/fallback path (fake transport) and the arrival state
  machine are covered offline.
- **Goal vocabulary is detector-limited.** "Sit", "water", "food", "exit",
  "wallet" and "money" expand to the classes in `live/goals.py`. A detector
  without `door` or `stairs` will never satisfy an exit goal; unknown classes
  stay unseen rather than being invented.
- **Arrival uses monocular depth.** A 1.5 m reading is an estimate, and the
  three-frame stability rule reduces but does not eliminate false arrivals.

## Offline tests

No model, camera, network or TTS engine is used:

```powershell
python -m pytest tests/test_live_resolution.py tests/test_live_db.py tests/test_live_guidance_bridge.py tests/test_live_anchors.py tests/test_live_goals.py tests/test_live_goal_agent.py tests/test_live_voice.py tests/test_live_runner_goals.py tests/test_input_size_profiles.py -q
```
