# Visiona3D

**Spoken navigation from one ordinary RGB camera.**

Visiona3D looks at the camera image, finds the floor, walls and objects, maps
them in 3D, plans a path and speaks short guidance. It needs no special camera
and no depth sensor. Navigation runs offline after setup. Spoken goals are
optional: speech-to-text uses Google's speech service, and the transcript can
be understood by Gemini Flash when `GEMINI_API_KEY` is set, or by the built-in
local parser with no key at all.

> Every number in this project is a starting value marked `TO TUNE ON REAL
> DATA`. Visiona3D is a working prototype, not a certified safety device or a
> replacement for a cane or guide dog.

---

## End-to-end workflow

One diagram for the whole runtime: what Visiona3D does when it is turned on,
how each frame is processed, and where the results go. Setup and options are
covered later in [Start here](#start-here-first-time) and
[Advanced control](#advanced-control).

# Visiona3D — End-to-End Workflow

```mermaid
flowchart TD
    A([Start]) --> B[Input & Configuration]

    B --> B1[Webcam / Video Input]
    B1 --> B2[Read Latest Frame]
    B2 --> B3[Load Camera Intrinsics, HFOV, Mount Height & Pitch]

    B3 --> C[Visual Perception]

    C --> C1[SegFormer: Floor / Wall / Obstacle Segmentation]
    C --> C2[Depth Anything V2: Depth Estimation]
    C --> C3[YOLO11n: Object Detection]
    C3 --> C4[IoU-Based Object Tracking]

    C1 --> D[3D Mapping]
    C2 --> D
    C4 --> D

    D --> D1[Ground-Plane Projection]
    D1 --> D2[Generate 3D Object Anchors]
    D2 --> D3[Build BEV Grid: 75 × 60 at 0.1 m / Cell]
    D3 --> D4[Generate Costmap]

    D4 --> E{Goal Understanding Enabled?}

    E -- No --> F[Target Selection]
    E -- Yes --> G{Goal Available?}

    G -- No --> G1[Hold S to Speak]
    G1 --> G2[Google Speech-to-Text]
    G2 --> G3{Gemini API Key Available?}

    G3 -- Yes --> G4[Gemini Flash: Resolve Goal to Object Class]
    G3 -- No --> G5[Local Deterministic Parser]

    G4 --> G6{Valid Goal?}
    G5 --> G6

    G6 -- No --> G1
    G6 -- Yes --> F

    G -- Yes --> F

    F --> F1{Active Goal Has Visible Candidate?}

    F1 -- Yes --> F2[Select Nearest Matching Goal Object]
    F1 -- No --> F3[Select Nearest Exploration Cell]

    F2 --> H[A* Path Planning]
    F3 --> H

    H --> I[Safety Validation]

    I --> I1{View Certified?}
    I1 -- No --> I2[STOP_UNCLEAR]
    I1 -- Yes --> I3{Path Found?}

    I3 -- No --> I4[STOP_BLOCKED]
    I3 -- Yes --> I5{"Clearance < 0.20 m?"}

    I5 -- Yes --> I6[CAUTION]
    I5 -- No --> I7[GUIDE]

    I2 --> J[Runtime Outputs]
    I4 --> J
    I6 --> J
    I7 --> J

    J --> J1[HUD Visualization]
    J --> J2[Speech Guidance]
    J --> J3[SQLite Session Store]
    J --> J4[Optional PNG / JSON / Video Exports]

    J1 --> K[Auxiliary Visualization]
    K --> K1[Trail Estimation]
    K --> K2[Zone Layer Display Only]

    J2 --> L{Continue?}
    J3 --> L
    J4 --> L
    K1 --> L
    K2 --> L

    L -- Yes --> B2
    L -- No --> M([End])

    style A fill:#d4edda,stroke:#28a745,color:#000
    style M fill:#f8d7da,stroke:#dc3545,color:#000
    style I2 fill:#f8d7da,stroke:#dc3545,color:#000
    style I4 fill:#f8d7da,stroke:#dc3545,color:#000
    style I6 fill:#fff3cd,stroke:#ffc107,color:#000
    style I7 fill:#d4edda,stroke:#28a745,color:#000
```

### Outputs and destinations

| Destination | Contents | Read by navigation? |
| --- | --- | --- |
| Screen (HUD) | Current frame, masks, object boxes, BEV grid, path and decision strip | It is the live view |
| Speech | Decisions, object alerts and the one-shot goal arrival sentence | No |
| `database\visiona_live.db` | Sessions, frames, object registry, BEV snapshots, events and goal records | No |
| `--map-png` | Session map with trail, zones, object markers and START/NOW | No |
| `--summary-json` | Decisions, counts, timings, errors and the `trail` block | No |
| `--out-video` | Annotated HUD recording | No |

Only the newest processed frame is displayed, so the picture and the map always
belong to the same moment. A failed stage becomes a visible status and a stop
decision, never a crash. The trail and zone layer are estimated from floor
motion: they can drift or freeze, and guidance reads only the current frame's
grid. Voice is the only part that uses the network; navigation never does.

The 0.20 m caution clearance and the 1.5 m arrival distance in the diagram are
starting values, both marked `TO TUNE ON REAL DATA` in the code and in the
option reference.

---

## Start here (first time)

You need a Windows or Linux computer, Python 3.11 or 3.12, and a webcam or a
video file. A GPU is not required.

### Step 1 - install once

Open PowerShell in the project folder:

```powershell
cd E:\git_projects\Projects\Visiona3D
python -m pip install uv
uv sync
.\.venv\Scripts\Activate.ps1
```

### Step 2 - download the models once

This is a one-time download of roughly a few hundred MB. After it finishes,
Visiona3D runs offline.

```powershell
python scripts\fetch_models.py --yes
```

### Step 3 - start

```powershell
python starter.py
```

Seven short questions appear. **Press Enter seven times.** That starts the
webcam with sensible defaults. Nothing else is required on the first run.

The first start can take up to about a minute while the models load. After
that, a window opens with the camera view, coloured floor and wall areas,
boxes around objects, a small map on the right, and spoken guidance.

To stop, click the picture and press **q**.

If you are blind or cannot read the questions, just press Enter: the webcam
starts and guidance is spoken. A helper only needs to launch the command once.
To skip the questions entirely, use this one-line start:

```powershell
python starter.py --source 0 --hfov 75
```

---

## The questions, explained

Every question is printed as `Question N of 7`, a one-line `example`, and
`Enter yours [default]`. Press Enter to accept the default.

| Question | Example answer | What it does | Default |
| --- | --- | --- | --- |
| 1. What should it look at? | `1` webcam, `2` video file | Video asks again for the path; drag the file into the window | `1` (webcam) |
| 2. How many frames? | `10`, `60`, `100` | Fewer frames = faster run; blank means the whole video | all frames |
| 3. Speed or quality? | `1` fast, `2` balanced, `3` quality | Model input-size profile | `1` (fast) |
| 4. Spoken guidance? | `yes` / `no` (or `1` / `0`) | `no` also turns the microphone off | `yes` |
| 5. Save the session map image? | `reports\session_map.png` or a folder | Writes the bird's-eye map with the walked route, walkable/wall/danger zones and detected objects | `yes`, `reports/session_map.png` |
| 6. Save an annotated video? | `reports\live\demo.mp4` | Also runs without a window; useful for recordings | `no` |
| 7. Calibration file? | `reports\camera_intrinsics.json` | Blank estimates the lens; type `?` for calibration steps | blank (estimates) |

Before the app starts, the exact command line equivalent to your answers is
printed, so you can copy it and reuse it next time. Any command-line arguments
skip the questions and are passed straight through.

Rules that apply after it starts:

| Key | Action |
| --- | --- |
| `q` | Quit (click the picture window first) |
| `s` | **Hold** to speak a destination, release to set it |
| `v` | Same as `s` (kept from the old keys) |
| `m` | Mute or unmute speech |

Guidance is spoken automatically. You do not need to press anything to hear it.

---

## Set a destination by voice

1. Hold **S** anywhere while the live window is open.
2. Wait for "Listening...", say a goal, then release.
3. Visiona3D hears the words, expands them into object classes, and plans
   toward the nearest one it can see.

Examples:

| You say | It searches for |
| --- | --- |
| "I want to sit" | chair, bench, couch, bed |
| "I need water" | bottle, cup, wine glass, sink, refrigerator |
| "I am hungry" | apple, banana, sandwich, orange, broccoli, carrot, pizza, donut, cake, refrigerator, dining table |
| "I want to leave" | door, stairs |
| "Find my backpack" | handbag, backpack |

The goal stays active across frames. When no candidate is visible it behaves
like the old `999 m` unknown distance: nothing is invented. When a candidate is
seen, the nearest usable 3D anchor becomes the A* target. When the goal is
within about **1.5 m** for **3 consecutive processed frames**, Visiona3D says
"**You have reached your goal destination...**" once and returns to normal
exploration. Both numbers are `TO TUNE ON REAL DATA`.

Say "cancel the goal" to clear it.

**How the words are understood.** By default the transcript is parsed locally
and deterministically: no API key, no account, no cloud LLM. If you set
`GEMINI_API_KEY`, the transcript is first sent to Gemini Flash
(`gemini-2.5-flash` by default). The model may only name detector classes, the
reply is validated locally, and the local parser takes over if the call fails,
times out or returns something unusable. `--no-voice-agent` forces local
parsing. Only voice uses the network; navigation never does.

```powershell
# Optional, current PowerShell window only. Get a key from Google AI Studio.
$env:GEMINI_API_KEY = "your-key"
python starter.py --source 0 --hfov 70
```

You can also put `GEMINI_API_KEY=...` in your local `.env`; the app already
loads that file at startup. `GEMINI_MODEL` overrides the model there.

If you have no microphone, set the same goal at startup. `--goal` always uses
the local parser, so it works with no key:

```powershell
python starter.py --source 0 --hfov 70 --goal "I want to sit"
```

---

## What you see and hear

| On screen | Meaning |
| --- | --- |
| Green area | Floor the system is willing to call free |
| Blue area | Wall |
| Red area | Obstacle |
| Coloured boxes | Objects, labelled with track number, class, confidence, distance and left/right offset |
| Dots in the boxes | The object's estimated position on the floor in 3D |
| Right panel | Top-down map: free, occupied, uncertain and unknown cells, plus the planned path |
| Bottom strip | The decision (`GUIDE`, `CAUTION`, `STOP_UNCLEAR`, `STOP_BLOCKED`), heading, nearest obstacle, frame age and speed |
| Red bar at the top | The camera is **not calibrated**, so all distances are estimates |

Spoken messages are short, for example "Path clear, continue forward",
"Caution, person close ahead" or "Stop, path blocked". Speech repeats are
suppressed until the situation changes.

The map is 2.5D: a top-down grid plus one 3D position per object. It is not a
full 3D mesh.

### The saved session map

`--map-png reports\session_map.png` writes a standalone bird's-eye picture
after every processed frame: the current 2.5D view around the camera, the
walked route, the accumulated ground zones, the detected objects and a scale
bar. It is a review aid, not part of navigation: guidance and safety never read
it. This map uses its own palette: here **grey** is wall and **blue** is
walkable floor, unlike the live HUD where floor is green and wall is blue.

**Walked route: START to NOW.** The route is drawn from an explicit **START**
dot to a **NOW** marker at the camera. When no step has been accepted yet, the
labels collapse into `START = NOW` and the footer prints
`NO STEP ACCEPTED YET (N gaps)`, so a frozen trail is not mistaken for a
missing feature. The footer also prints the accepted `steps` and `gaps`.

**Objects: one moving marker each.** Detected objects appear as labelled
markers in world coordinates, one colour per class (same class = same colour,
different classes differ). Each object keeps one moving marker: a re-created
tracker ID is rebound to the nearest same-class marker within a gate that grows
with elapsed time (`MAP_OBJECT_GATE_M` 0.6 m plus `MAP_OBJECT_SPEED_MPS`
1.6 m/s, capped at 8 m, all `TO TUNE ON REAL DATA`), so a person who walks
through the view moves their marker instead of leaving a trail of copies. If
two detections want one marker in the same frame, the nearest wins and the
other becomes a new marker. Person markers disappear after `MAP_OBJECT_TTL_S`
= 3 s without a detection (`TO TUNE ON REAL DATA`); static classes stay for the
session.

**Ground zones.** The picture accumulates a zone layer over the whole session:

| Colour | Meaning |
| --- | --- |
| Green | The route actually walked; its length is printed in the legend |
| Blue | Certified walkable floor; its measured area is printed |
| Grey | Wall |
| Bright red | Danger: blocked or occupied |
| Dark red band | The edge of what was never detected |

People and other detected objects only get class-coloured markers - they never
paint a red zone, so someone standing in front of you does not leave a
permanent hazard blob. Zone cells are 0.2 m (`TO TUNE ON REAL DATA`), a detected
object clears danger votes within 0.6 m (`TO TUNE ON REAL DATA`), and the
dark-red band is one cell wide (`TO TUNE ON REAL DATA`).

**How the route is estimated.** The trail and the zone layer come from
floor-plane motion between processed frames - there is no IMU or wheel odometry
- so both drift together and are labelled as an estimate on the image. A step
is accepted only if it fits a plausible walking speed (2.5 m/s) and turn rate
(60 deg/s) multiplied by the elapsed time between the two frames
(`TO TUNE ON REAL DATA`). That keeps slow processing from freezing the route
while absurd jumps are still rejected. For a video file the elapsed time is
**video time** (`frame index / source fps`), so `--no-realtime` does not starve
the gate. Baselines under 0.5 s are skipped without counting a gap, because
adjacent frames at 30-60 fps move too little to measure (`TO TUNE ON REAL
DATA`).

### Video playback

A video file plays on the wall clock. If the CPU needs
longer than one video frame, older frames are skipped rather than queued, so
the analysis and any spoken goal refer to the current video time instead of a
backlog. The picture updates at the processing rate (1.2-2.5 s per analysed
frame on CPU), and `frame age` shows the delay. `--no-realtime` processes every
frame instead: exhaustive, but much slower than real time. When a STOP is shown,
the strip also prints the failed safety check (for example `why: certified
floor fraction in corridor 0.000 < 0.300`).

### Saved outputs

Everything below is opt-in and written to the path you choose:

| Output | Flag | Contents |
| --- | --- | --- |
| Session map | `--map-png path.png` | The bird's-eye map described above, rewritten after every processed frame |
| Summary | `--summary-json path.json` | Run status plus the `trail` estimate |
| Annotated video | `--out-video path.mp4` (needs `--headless`) | The full HUD as rendered |
| SQLite session | `--db path.db` (default `database\visiona_live.db`) | Frames, objects, map snapshots and events |

The summary is the quickest way to check a run without opening anything else.
It records the source, processing size, profile, goal and goal interpreter,
calibration provenance, processed frames, wall time, decision counts, object
count, average processing time, errors, and the trail estimate:

```json
"trail": {"steps": 4, "gaps": 0, "route_m": 1.34, "x_m": -0.03, "y_m": 1.33, "heading_deg": 1.9}
```

`steps` counts accepted motion updates, `gaps` counts frames where no step
could be accepted, `route_m` is the estimated path length, `x_m` / `y_m` the
estimated position in the session frame and `heading_deg` the estimated
heading. A run with `steps: 0` and a large `gaps` value is frozen, not a
missing feature; see [If something does not work](#if-something-does-not-work).

---

## Calibration, in plain words

Without calibration, Visiona3D assumes a typical camera lens. It can still show
where the floor, walls and objects are, but the metre values are estimates; the
red bar stays on screen to say so.

Calibration is a **one-time measurement of your camera** that turns those
estimates into metres. It needs a printed checkerboard and about five minutes:

```powershell
python scripts\capture_calibration.py --camera 0 --out calib --board 9x6 --views 15
python scripts\calibrate_camera.py --images calib --board 9x6 --square 0.018 --out reports\camera_intrinsics.json
```

- `--board 9x6` means 9 by 6 **inner corners**, not squares.
- `--square 0.018` is the physical square size in metres. Measure your print
  and replace this number.
- Capture at the same resolution you will run at. A calibration made for a
  different resolution is rejected instead of being rescaled.
- Then start `python starter.py` and answer question 7 with
  `reports\camera_intrinsics.json`.

Type `?` at question 7 to print these same instructions.

---

## If something does not work

| Symptom | What to do |
| --- | --- |
| `cannot open camera` or no window | Close other apps using the camera; try `python starter.py --source 1` |
| Models are missing | Run `python scripts\fetch_models.py --yes` once |
| First start looks frozen for up to a minute | Models are loading; wait for the first frame |
| Video starts at the wrong place | `python starter.py --source your.mp4 --start-frame 100` |
| A video run ends immediately | Add `--loop` to repeat it, or check the path |
| Calibration file rejected | Recapture at the run resolution; the runner does not rescale |
| Distances look too short or too long | The camera is uncalibrated (`--hfov` demo mode), or `--mount-height` / `--pitch` need measuring |
| No speech | Press `m`; check the Windows default voice; `--no-speech` mutes it deliberately |
| Voice does not start | Check the microphone is connected and not used by another app; install the voice packages with `uv sync`; add `--no-voice` to skip voice |
| "Could not understand" or no transcript | Speak after "Listening..." appears, hold the key while talking, and check the internet connection (speech-to-text is online) |
| No goal is set | Use `--goal "I want to sit"` at startup, or check the on-screen goal strip after speaking |
| Voice goal ignores Gemini | Set `GEMINI_API_KEY` in the same PowerShell window. A timeout, quota or invalid reply prints its reason and falls back to the local parser |
| Saved map says `NO STEP ACCEPTED YET`, or the route stays at 0 m | The floor-motion fit believes the camera did not move. Check `trail.gaps` in `--summary-json`. Motion below the 0.5 s baseline, a dropped floor plane, or a scene the floor homography does not describe (fast panning, heavy night lighting) all reject steps. This is a display limit - guidance still runs |
| Saved map route or zones drift | The layer is estimated from floor-plane motion only, with no IMU or wheel odometry. Drift is expected; guidance and safety never read it |
| Everything is slow | Use `--profile fast --proc-width 480`; CPU-only runs at roughly 0.5-0.8 updates per second |

---

## Where your data goes

Everything stays on this machine by default. The only network use is voice: the
held-key audio clip goes to Google speech-to-text and, if `GEMINI_API_KEY` is
set, the transcript goes to Gemini Flash. No frames, maps or database rows are
uploaded. Use `--no-voice` to disable voice entirely, or `--no-voice-agent` to
keep speech-to-text but understand goals locally.

| Path | Contents |
| --- | --- |
| `database\visiona_live.db` | Sessions, frames, every object position, the map snapshot for each stored frame, spoken events, and `goal_set` / `goal_cancelled` / `goal_arrival` records |
| `reports\` | Calibration, videos and summaries you asked a command to write |
| `weights\`, `sample-vid\`, `data\`, `calib\` | Downloaded models and your local inputs |

Read the object registry:

```powershell
python -c "import sqlite3;c=sqlite3.connect('database/visiona_live.db');print(c.execute('select track_id,class_name,round(distance_m,2),round(anchor_x,2),round(anchor_y,2),round(anchor_z,2) from object_registry order by distance_m').fetchall())"
```

`object_registry` holds one row per tracked object per session; `bev_snapshots`
holds the compressed 75 x 60 grid for each stored frame. Delete the database
file at any time to start fresh; it is recreated on the next run.

---

## Advanced control

Every option is available directly on the same entry point. `python starter.py
--help` lists them all.

### Source and camera geometry

| Option | Meaning |
| --- | --- |
| `--source 0` | Webcam index, or a video file path |
| `--intrinsics file.json` | Use a measured calibration (accurate metres) |
| `--hfov 70` | Estimate the lens from a horizontal field of view instead (demo only; red bar stays on) |
| `--mount-height 1.4` | Camera height in metres (`TO TUNE ON REAL DATA`) |
| `--pitch 15` | Downward camera tilt in degrees (`TO TUNE ON REAL DATA`) |
| `--camera-width`, `--camera-height` | Request a specific webcam resolution |

### Speed and model choice

| Option | Meaning |
| --- | --- |
| `--profile fast` | Smallest model inputs and fastest CPU run (default) |
| `--profile balanced`, `--profile quality` | Larger inputs, slower, no change to safety rules |
| `--proc-width 640` | Processing width cap; lower is faster |
| `--weights-dir weights` | Where local model weights live |
| `--depth-variant indoor`, `--depth-variant outdoor` | Pick the depth checkpoint for the scene |

### Navigation and safety

| Option | Meaning |
| --- | --- |
| `--goal "I want to sit"` | Expand a natural-language goal and plan to its nearest candidate |
| `--goal-class chair` | Plan toward one exact detector class (legacy single-class mode) |
| `--arrival-distance 1.5` | Distance that counts as reaching the goal (`TO TUNE ON REAL DATA`) |
| `--arrival-frames 3` | Consecutive frames required before the arrival sentence (`TO TUNE ON REAL DATA`) |
| `--no-voice` | Do not open the microphone or global key listener |
| `--goal-model gemini-2.5-flash` | Gemini Flash model for voice goals (needs `GEMINI_API_KEY`) |
| `--no-voice-agent` | Force the local deterministic goal parser |
| `--floor-threshold 0.90` | Floor certification threshold (`TO TUNE ON REAL DATA`) |
| `--fusion-config file.json` | Optional tuned SegFormer + MobileSAM refinement |
| `--conformal-depth file.json` | Optional pessimistic obstacle bounds from calibration |

### Audio

| Option | Meaning |
| --- | --- |
| `--no-speech` | Start muted |
| `--no-speak-objects` | Speak navigation decisions only, not object chatter |

### Recording, automation and storage

| Option | Meaning |
| --- | --- |
| `--headless` | No window; use with `--out-video` |
| `--out-video out.mp4` | Write the annotated view to a video file |
| `--video-fps 4` | Output video frame rate |
| `--start-frame 100` | Skip ahead in a video |
| `--max-frames 20` | Stop after this many processed frames |
| `--loop` | Repeat the video |
| `--no-realtime` | Read every frame back-to-back instead of keeping the video on the wall clock (the trail still uses video time) |
| `--db path.db`, `--no-db` | Choose the database, or turn persistence off |
| `--map-png path.png` | Save a top-down session map with the estimated walked route, walkable/wall/danger zones and detected objects after every processed frame |
| `--summary-json file.json` | Write the run summary, including the `trail` block (`steps`, `gaps`, `route_m`, `x_m`, `y_m`, `heading_deg`) |

Two useful examples:

```powershell
python starter.py --source 0 --intrinsics reports\camera_intrinsics.json --goal "I want to sit"
python starter.py --source sample-vid\FRONT\VID2.mp4 --hfov 70 --profile fast --max-frames 4 --headless --out-video reports\live\demo.mp4 --no-speech --summary-json reports\live\summary.json
```

`--out-video` requires `--headless`. `--intrinsics` and `--hfov` are mutually
exclusive; the program refuses to guess silently.

### See every feature on one video

This renders the full HUD (segmentation, object boxes and IDs, 3D anchors, BEV
map, A* path and decisions) to an MP4, stores the 3D object grid in SQLite, and
speaks guidance while it runs:

```powershell
python starter.py --source sample-vid\FRONT\VID2.mp4 --hfov 70 --profile fast --start-frame 100 --max-frames 40 --goal-class person --out-video reports\full_demo\features.mp4 --headless --no-realtime --summary-json reports\full_demo\summary.json
```

For your own video, replace the `--source` path (keep the quotes if it has
spaces). Add `--no-speech` to render silently; spoken events are still logged.

Results:

- `reports\full_demo\features.mp4` - the annotated run.
- `reports\full_demo\summary.json` - decisions, object count, timings, errors.
- `database\visiona_live.db` - `frames`, `objects`, `object_registry`,
  `bev_snapshots` and `events` for that session.

On the bundled videos, `sample-vid\FRONT\VID3.mp4` is the indoor route where
the estimated trail moves (a smoke run accepted 4 steps over 1.34 m).
`VID2.mp4` is a panning night-market scene where the floor-motion fit can stay
at `START = NOW`; the HUD, tracking and guidance still run, and the map and
`trail.gaps` say the trail is frozen.

The optional SegFormer + MobileSAM fusion and conformal obstacle bounds need a
tuned config or fitted interval made from real labelled captures; they are not
enabled by this command and are not faked.

---

## How it works

The [end-to-end workflow](#end-to-end-workflow) above is the complete runtime
chain: frames are captured, segmented, lifted into 3D, certified into a BEV
grid, planned on, turned into a decision, and finally drawn, spoken and stored.
The code follows the same split, described under
[Project structure](#project-structure).

Only the newest processed frame is displayed, with its age shown, so the
picture and the map always belong to the same moment. Model input sizes change
with `--profile`; safety rules do not.

---

## Project structure

```
starter.py            the only entry point: questionnaire, arguments, run loop
live/                 live loop: runner, camera/video source, HUD, map image,
                      trail, zone layer, SQLite, goals, Gemini agent, voice
perception/           backend interfaces: SegFormer, Depth Anything V2, YOLO,
                      IoU tracker, fusion config, analytic fakes for tests
mapping/              BEV grid, rolling map, obstacle bounds
planning/             A*, costmap, path smoothing, targets, guidance
geometry/             camera model, ground plane, frames, 3D lift, scale
calibration/          intrinsics, conformal depth/floor bounds, geometric veto
audio/                speech output and push-to-talk capture
core/                 shared configuration constants and detection types
evaluation/, dataset/ offline metrics and synthetic dataset generation
scripts/              model download, calibration capture/run, demos, evaluations
tests/                offline tests: no model, camera, network or speech engine
docs/                 deeper documentation, indexed in docs/README.md
database/             SQLite session store, created on first run
reports/              maps, videos and summaries written by commands
weights/, sample-vid/, MobileSAM/   downloaded models and local inputs
yolo11n.pt            YOLO11n checkpoint downloaded to the project root
pyproject.toml, uv.lock, requirements.txt   packaging and pinned dependencies
```

The live loop is the only place that wires modules together. `perception/`,
`mapping/`, `planning/`, `geometry/` and `calibration/` stay independent of the
HUD and can be exercised offline against fakes.

---

## Safety and honesty

1. Unknown and uncertain cells are never treated as free space.
2. A failed stage becomes a visible status and a stop decision, not a crash.
3. Obstacles use a pessimistic lower bound only when a calibrated depth
   interval is supplied; otherwise no lower bound is invented.
4. The planner recommends a path; it never claims the path is safe.
5. Uncalibrated runs are labelled on screen and in the session record.
6. The verified runs used route video, not a physically calibrated camera and
   not blind participants. Nothing here is a validated accuracy claim.

---

## Performance (measured, not promised)

CPU only, 640x360 processing, bundled 1920x1080 video, warm frames:

| Stage | Time |
| --- | --- |
| Segmentation | 0.16 - 0.26 s |
| Depth | 0.49 - 0.80 s |
| Detection | 0.13 - 0.17 s |
| 3D lift | 0.41 - 0.57 s |
| **Total per processed frame** | **1.23 - 2.45 s** |

That is roughly 0.5-0.8 HUD updates per second. A warm smoke run measured
1.8 s on its second frame; its first frame took 30 s including model warm-up.
The display stays responsive and always shows the latest processed frame.

---

## Tests

No model, camera, network or speech engine is used in tests.

```powershell
python -m pytest tests audio -q
python geometry\scripts\run_checks.py
```

Latest run on the development machine: `479 passed, 1 warning, 14 subtests
passed in 197.44s` (the warning is expected synthetic-dataset stratification
output), and all M1-M13 geometry, perception, mapping and planning checks
passed. Synthetic evaluation is regenerated with
`python scripts\run_planner_eval.py --scenes 200 --seed 42`. Report files are
local and not committed.

---

## Limitations

- Not field-tested on a physical camera or with blind users.
- The map is a 2.5D grid plus one 3D anchor per object, not a mesh.
- Each frame builds its own grid for planning. The saved `--map-png` also
  accumulates BEV cells into a display-only world layer, but guidance uses only
  the current frame's grid; the saved route and zone layer are an estimate and
  are not read by guidance or safety.
- The saved route and zone layer depend on a floor-motion fit. Indoor walking
  works on the bundled route video, but fast panning or heavy night lighting
  can leave the route at 0 m even while guidance and object tracking keep
  running (observed on the night-market sample). The map and `trail.gaps`
  report this rather than hiding it.
- Monocular depth bias is uncorrected, including the 1.5 m arrival distance.
- Voice goals and arrival thresholds are tested offline only; microphone,
  speech-to-text and real-room distances still need a field test. The Gemini
  path is covered with a fake transport, not the live API.
- CPU-only HUD rate is about 0.5-0.8 Hz on the test machine.
- No regulatory or safety certification.

---

## Documentation

The [docs index](docs/README.md) lists the supporting documents with a
one-line summary each. The main ones:

- [`docs/live_system.md`](docs/live_system.md) - operating guide, synchronisation,
  session map and trail model, database schema, measured latency.
- [`docs/perception_backends.md`](docs/perception_backends.md) - models, fusion,
  evaluation and tuning.
- [`docs/mapping_planning.md`](docs/mapping_planning.md) - grid conventions,
  certification, planner and guidance.
- [`docs/annotation_guide.md`](docs/annotation_guide.md) - mask and box labels.
- [`docs/capture_protocol.md`](docs/capture_protocol.md) - real-world data
  collection.

Code-level conventions are in [`geometry/notation.md`](geometry/notation.md).

---

**Author:** Paavan Siri Vardhan Narava
