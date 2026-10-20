# Visiona AI: Senior Software & Computer-Vision Systems Audit Report

**Auditor:** Senior Software Auditor for Computer-Vision and Agentic-AI Systems  
**Date:** October 12, 2026  
**Repository:** Visiona AI (`paavansirivardhan123/Visiona3D`)  
**Commit Audited:** `ea6805a3808e90c52eb366c8460248cbadcfb691` ("removed map")  

---

## 1. Executive Verdict

**Verdict: RESTART CORE ARCHITECTURE, SELECTIVELY REUSE ISOLATED ALGORITHMS.**

Visiona AI is currently in a broken, non-executable state. The primary entry point (`main.py`) cannot even be imported due to a dangling `ModuleNotFoundError` (`from services.google_maps import get_directions` on `agents/tools/vision_tools.py:27` after `services/google_maps.py` was deleted in commit `ea6805a`). Beyond the import failure, the system is fundamentally unfit as a computer-vision or navigation foundation: it possesses zero floor/wall segmentation, zero camera calibration, zero geometric ground-plane modeling, and zero metric mapping. Metric distance is estimated via an uncalibrated relative-depth heuristic multiplied by an arbitrary `* 2.0` hardcoded factor, and the background depth cache pairs bounding boxes by raw list index rather than spatial association. While isolated modules (the max-heap priority queue, directional penalty decay, 1D Kalman filter math, and JSONL logger) contain salvageable algorithmic logic, the multi-threaded orchestration in `main.py` suffers from race conditions, coordinate frame mismatches, and severe architectural drift from the README claims. Attempting to incrementally patch this codebase into a calibrated, walking-height navigation system will accumulate technical debt; the core geometry, perception pipeline, and state orchestration should be restarted from scratch.

---

## 2. Component Status Table

| Component | Status | Evidence (File:Line or Command) |
|---|---|---|
| Entry Point & Runtime Loop | **BROKEN** | `main.py:26` -> `agents/orchestrator.py:6` -> `agents/tools/vision_tools.py:27` crashes on `from services.google_maps import get_directions` (`ModuleNotFoundError`) |
| Object Detection (YOLO) | **LIKELY WORKS** | `perception/vision.py:39`, `core/config.py:11` loads `yolo11n.pt` (not YOLOv8n). Downscales frame to 1280px (`_YOLO_WIDTH = 1280`) |
| Depth Pipeline (Async Cache) | **BROKEN** | `perception/vision.py:117-124`: cache lookup requires `len(cached) == len(detections)` and pairs by raw index with zero spatial tracking; falls back to heuristic width if count changes |
| Monocular Depth Estimation | **BROKEN** | `perception/mono_depth.py:95`: computes `(self._scale / midas_val) * 2.0`, doubling calibrated distances; `_calibrate` (line 163) hardcodes `focal_px = orig_w * 0.73` |
| Heuristic Depth Fallback | **LIKELY WORKS** | `perception/mono_depth.py:204-212`: pinhole estimation using `KNOWN_WIDTHS_CM` and `FOCAL_LENGTH_PX = 700` |
| Object Tracker ("ByteTrack") | **BROKEN / STUB** | `perception/tracker.py:69-138`: Not ByteTrack. Only single-stage greedy IoU (`_IOU_THRESHOLD = 0.3`). No low-confidence second association. Kalman filter is 1D depth only, not 2D bbox |
| 1D Kalman Filter | **LIKELY WORKS** | `kinematics/kalman.py:4-47`: Valid 1D state-space equations $[x, v]^T$, but `update()` is called with hardcoded `dt=1.0` in `tracker.py:60` regardless of actual frame time |
| Speed Estimation | **LIKELY WORKS** | `kinematics/speed.py:33-79`: Derives $\Delta d / \Delta t$ from track history, clamps static objects (`STATIC_CLASSES`), compensates for user egomotion |
| Time-to-Collision (TTC) | **LIKELY WORKS** | `kinematics/ttc.py:13-35`: Computes $d / v$ with exponential moving average (`TTC_SMOOTH_ALPHA = 0.3`) for approaching objects |
| Optical Flow Egomotion | **LIKELY WORKS** | `perception/egomotion.py:11-140`: Lucas-Kanade optical flow on static background features, estimates forward expansion and lateral panning |
| Max-Heap Priority Queue | **WORKS** | `core/priority_queue.py:40-73`: Correct heapq max-heap implementation sorting by TTC, distance, and object weight |
| Penalty Heatmap & Grouping | **WORKS** | `kinematics/heatmap.py:42-120`: Directional penalties decay at 3.0 pts/sec, increase by 30.0 pts on speech; groups $>5$ objects into "Group of..." |
| Speech Engine (TTS) | **LIKELY WORKS** | `audio/speech.py:19-212`: Triple priority queues (`_eq`, `_pq`, `_nq`), audio ducking, emergency bypass. SAPI5 COM threading is sensitive |
| Alert System (Proximity Beeps) | **LIKELY WORKS** | `audio/alert.py:19-62`: Frequency scaling via `BEEP_LEVELS`. Windows-only (`winsound.Beep`), silent on Linux/macOS |
| Voice Input (Push-To-Talk) | **BROKEN (OFFLINE)** | `audio/voice_input.py:130`: Calls `recognizer.recognize_google(audio)`. Fails without internet access. PyAudio pre-warming is implemented |
| Keyboard Listener (PTT) | **LIKELY WORKS** | `main.py:157-182`: `pynput.keyboard.Listener` binds `v` (LLM) and `r` (Remember Mode). Key `g` is NOT bound |
| Custom Recognition (FeatureDB) | **BROKEN** | `core/recognition.py:8-124`: MobileNetV3 vector embedding extractor. In `main.py:282`, crops `frame[y1:y2, x1:x2]` using display coordinates on unscaled camera frames |
| Vector Database Service | **DEAD CODE / STUB** | `services/vector_db.py:1-10`: 9-line stub file returning `"No remembered custom objects matched this signature."` |
| Visual Memory & Goals | **WORKS** | `core/memory.py:11-160`: 15-minute rolling memory buffer in `VisionMemory`, goal keyword matching in `GoalSystem` |
| LangChain Agent Orchestrator | **BROKEN** | `agents/orchestrator.py:1-192`: Fails import at line 6 due to deleted `services.google_maps`. When mocked, routes 9 tools via ChatGroq |
| Agent Tools (9 Tools) | **BROKEN / STUB** | `agents/tools/vision_tools.py:1-82`: 9 tools exist. `calculate_route` crashes; `get_objects_near` is hardcoded static text stub |
| Google Maps Service | **DEAD CODE / MISSING** | Deleted in commit `ea6805a`. Left dangling imports in `agents/tools/vision_tools.py:27` |
| Multi-Camera Feeds & HUD | **LIKELY WORKS** | `main.py:31-80, 333-366`: Synchronous OpenCV video capture for FRONT, LEFT, RIGHT, BACK with 2x2 grid assembly |
| Telemetry Logger | **WORKS** | `core/logger.py:12-51`: Correct JSONL session logging matching recorded telemetry in `logs/session_20260901_160359.jsonl` |
| Floor / Wall Segmentation | **MISSING** | No module or code exists in the repository for semantic or geometric floor/wall segmentation |
| Metric Map & Path Planner | **MISSING** | No occupancy grid, elevation map, costmap, or path planner exists in the repository |
| Safety Abstention System | **MISSING** | No abstention logic exists; depth estimates are always emitted even when uncalibrated or uncertain |

---

## 3. Claim-by-Claim Verification of the README

1. **YOLOv8n detection and 1280px resize**
   - **Verdict:** **BROKEN / CONFLICT**
   - **Evidence:** `core/config.py:11` sets `MODEL_PATH = "yolo11n.pt"` (YOLO11 nano, not YOLOv8n). `yolov8n.pt` does not exist in the workspace. Resize to 1280px is implemented in `perception/vision.py:25, 78` (`_YOLO_WIDTH = 1280`, `cv2.resize(frame, (_YOLO_WIDTH, yolo_h))`).

2. **Depth Anything V2 depth in background thread with heuristic fallback**
   - **Verdict:** **BROKEN**
   - **Evidence:** `perception/vision.py:184-188` spawns a background thread calling `MonoDepth.compute()`. However, the caching logic (`perception/vision.py:117-124`) only reuses depths if `len(cached) == len(detections)`, assigning values purely by array index with zero spatial correlation. If detection count changes between frames, it discards ML depth and falls back to `_heuristic()`.

3. **Depth calibration: `metric_depth_m = (scale / depth_value) x 2.0` and auto-calibration**
   - **Verdict:** **BROKEN (MATHEMATICAL CONTRADICTION)**
   - **Evidence:** `perception/mono_depth.py:95` implements `round((self._scale / midas_val) * 2.0, 2)`. In `_calibrate()` (`mono_depth.py:172`), `self._scale` is set to `real_dist * d_ref`. When evaluating that exact reference object, `(self._scale / d_ref) * 2.0 = real_dist * 2.0`, immediately doubling the calibrated distance due to the arbitrary hardcoded `2.0` multiplier. Furthermore, `_calibrate()` bootstraps `real_dist` using an arbitrary uncalibrated focal length formula (`focal_px = orig_w * 0.73` at line 163).

4. **"ByteTrack" tracker: is it real ByteTrack or IoU-only tracker with Kalman?**
   - **Verdict:** **STUB / MISREPRESENTED**
   - **Evidence:** `perception/tracker.py:69-138`. It is **NOT ByteTrack**. Real ByteTrack performs two-stage association with high- and low-confidence detection thresholds. `tracker.py:123-133` runs a simple single-stage greedy IoU argmax loop (`_IOU_THRESHOLD = 0.3`). The Kalman filter (`kinematics/kalman.py`) is a 1D filter tracking only scalar depth, not 2D bounding-box kinematics.

5. **1D Kalman filter per track, speed, motion class, TTC, EMA smoothing, threat score formula**
   - **Verdict:** **LIKELY WORKS (WITH dt FLAW)**
   - **Evidence:** `kinematics/kalman.py` implements a 1D constant-velocity filter, but `perception/tracker.py:60` invokes `self.kf.update(det.distance_m)` without passing `dt`, defaulting to `dt=1.0` regardless of frame rate. Speed and motion classification (`kinematics/speed.py:33-79`) and EMA-smoothed TTC (`kinematics/ttc.py:13-35`, `TTC_SMOOTH_ALPHA = 0.3`) operate as described. `core/detection.py:38-52` implements the exact threat score formula, but `Config.TTC_WARN_THRESHOLD` is `3.0` in `core/config.py:96` instead of `3.5` claimed in the README.

6. **Optical-flow egomotion and motion compensation**
   - **Verdict:** **LIKELY WORKS**
   - **Evidence:** `perception/egomotion.py:11-140` runs Lucas-Kanade optical flow on non-dynamic background features, calculates radial expansion `med_exp`, and estimates user forward speed (`min(2.5, med_exp * 0.4)`). `kinematics/speed.py:61` compensates object speed with `true_vel = relative_vel + egomotion_speed_mps`.

7. **Priority queue, penalty heatmap, grouping messages ("Group of persons ahead")**
   - **Verdict:** **WORKS**
   - **Evidence:** `core/priority_queue.py:40-73` (max-heap). `kinematics/heatmap.py:58-60` decays directional penalties by `Config.PENALTY_DECAY * dt` (3.0 pts/s) and applies `Config.PENALTY_APPLY` (30.0 pts) on announcement. `heatmap.py:133` formats groups as `"Group of {plural} {dir_str}"`.

8. **Cooldowns: README says both 7 s and 5 s in different places; real values in config and code**
   - **Verdict:** **BROKEN / INCONSISTENT**
   - **Evidence:** In `core/config.py:58-65`: `SPEECH_COOLDOWN = 2.5`, `SEMANTIC_COOLDOWN = 5.0`, `FRONT_CAMERA_COOLDOWN = 3.0`, `GOAL_SAME_PANEL_COOLDOWN = 5.0`, `GOAL_CONSTANT_COOLDOWN = 7.0`. However, in `main.py:537`, cooldowns are hardcoded as literals: `cooldown_time = 7.0 if dir_str == "FRONT" else 3.0`. The config values `FRONT_CAMERA_COOLDOWN`, `GOAL_SAME_PANEL_COOLDOWN`, and `GOAL_CONSTANT_COOLDOWN` are completely unused dead code.

9. **Emergency speech bypass and audio ducking**
   - **Verdict:** **LIKELY WORKS**
   - **Evidence:** `audio/speech.py:22-24, 83, 130`. Three queues: emergency (`_eq`), priority (`_pq`), normal (`_nq`). Emergency items bypass ducking and enforce `volume = 1.0`. `duck()` sets `_interrupt_requested = True` and mutes volume to 0.0.

10. **Proximity beep alerts**
    - **Verdict:** **LIKELY WORKS (WINDOWS ONLY)**
    - **Evidence:** `audio/alert.py:19-62` calls `winsound.Beep(freq, dur)` in a background thread based on `Config.BEEP_LEVELS`. On Linux/macOS, `winsound` fails to import and beeps are silently disabled.

11. **Push-to-talk keys: README lists V, R, and mentions G; find the real bindings**
    - **Verdict:** **BROKEN / INCONSISTENT**
    - **Evidence:** `main.py:160-182`: `on_press` binds only `v` (LLM mode) and `r` (Remember mode). Key `g` is **NOT BOUND** anywhere in `main.py`, despite README line 215 ("Hold V / G") and `audio/voice_input.py:57` claiming "hold G for GPS".

12. **"Zero-latency" mic pre-warming**
    - **Verdict:** **LIKELY WORKS**
    - **Evidence:** `audio/voice_input.py:46-55`: PyAudio stream opens at `VoiceInputEngine` initialization with `start=True` and continuously reads PCM frames into an internal discard loop until `start_recording()` activates buffer storage.

13. **LangChain agent: README says "9 tools" but lists 4; count, list, and test real tools**
    - **Verdict:** **BROKEN (TOTAL 9 TOOLS)**
    - **Evidence:** `agents/tools/vision_tools.py` defines exactly 9 tools: (1) `set_persistent_goal`, (2) `mark_goal_completed`, (3) `lower_goal_candidate_priority`, (4) `query_past_detections`, (5) `summarize_scene`, (6) `set_search_intent`, (7) `calculate_route`, (8) `save_object_signature`, (9) `get_objects_near`. Direct execution fails due to `from services.google_maps import get_directions` (line 27). When `services.google_maps` is mocked, all 9 tool functions execute successfully (verified in headless smoke test Stage C). `get_objects_near` is an empty string stub.

14. **Confirmation system, goal system, arrival detection (<= 1 step), step-based tracking**
    - **Verdict:** **PARTIALLY BROKEN (CONFIRMATION SYSTEM MISSING)**
    - **Evidence:** 
      - A "step" is strictly defined in `core/config.py:16` as `METERS_PER_STEP = 0.75` meters. In `main.py:641`, `current_steps = max(1, round(distance_m / 0.75))`.
      - Arrival detection (`main.py:643`) triggers when `current_steps <= 1`, prompting contextual arrival speech.
      - **Confirmation system is completely MISSING**: `agents/orchestrator.py:93-185` has no interactive confirmation dialog, state machine, or "Asking for [intent]?" mechanism.

15. **15-minute visual memory and custom object labels**
    - **Verdict:** **WORKS**
    - **Evidence:** `core/memory.py:11-82`: `VisionMemory(max_minutes=15)` buffers formatted scene descriptions in a `deque`, evicting entries older than 900 seconds. `label_object()` stores aliases in `_custom_objects`.

16. **MobileNetV3 embedding label swap (FeatureDB): exists, wired in, similarity threshold**
    - **Verdict:** **BROKEN**
    - **Evidence:** `core/recognition.py:8-124`: `FeatureDB` uses `torchvision.models.mobilenet_v3_small` with cosine similarity threshold `threshold = 0.82` (line 90). It is wired into `main.py:284` inside `_processing_loop`. However, it crops `frame[y1:y2, x1:x2]` using display coordinates (`Config.DISPLAY_W = 640`, `DISPLAY_H = 480`) against full camera frames (e.g., 1280x720 or 1920x1080), causing spatial bounding box distortion. Furthermore, running MobileNetV3 synchronously per detection per frame severely degrades CPU throughput.

17. **Remember Mode (R key)**
    - **Verdict:** **BROKEN (REQUIRES CLOUD LLM)**
    - **Evidence:** `main.py:167-170, 903-925`. When R is released, speech is routed to `agent_engine.extract_memory_label(text)` which calls Groq cloud LLM to parse `alias` and `base_class`. If offline or key missing, it fails with "The reasoning engine is offline". If successful, it captures 50 frames to `database/{alias}/img_N.jpg`.

18. **Multi-camera (4 feeds, 2x2 HUD)**
    - **Verdict:** **LIKELY WORKS**
    - **Evidence:** `main.py:116, 333-366`: `CameraFeed` instances for FRONT, LEFT, RIGHT, BACK. If $\ge 3$ active feeds, `_build_grid` stacks them into a 2x2 grid via `np.vstack((top, bottom))`.

19. **Session logging to `logs/session_*.jsonl`**
    - **Verdict:** **WORKS**
    - **Evidence:** `core/logger.py:12-51`. Output verified in `logs/session_20260901_160359.jsonl`. Emits valid JSONL records with timestamps, directions, object attributes, and speech events.

20. **"Offline-first": which components actually need network**
    - **Verdict:** **BROKEN / FALSE CLAIM**
    - **Evidence:** The system is **NOT offline-first**:
      - Speech recognition requires Google Cloud STT (`audio/voice_input.py:130`: `recognizer.recognize_google`).
      - Agent reasoning requires Groq Cloud API (`agents/orchestrator.py:32`: `ChatGroq`).
      - Initialization requires internet to download HuggingFace Depth Anything V2 weights (~100MB) and torchvision MobileNetV3 weights (~10MB) if not locally cached.

21. **Navigation ("take me to the pharmacy"): real routing/localization or LLM text?**
    - **Verdict:** **MISSING / BROKEN**
    - **Evidence:** Commit `ea6805a` completely deleted `services/google_maps.py`. There is zero indoor/outdoor localization, no visual odometry, no SLAM, no map, and no route planning. Calling `calculate_route` crashes with `NameError` / `ModuleNotFoundError`.

22. **Performance claims (30+ FPS, ~50 ms YOLO, ~300 ms depth): measured or claimed?**
    - **Verdict:** **UNVERIFIED / CLAIM ONLY (CONTRADICTED BY CODE)**
    - **Evidence:** No benchmarks, profiling harnesses, or performance telemetry exist in the repository. The numbers ~50ms and ~300ms appear only as hardcoded string comments in `perception/vision.py:5-6`. Crucially, `main.py:329-331` explicitly throttles processing to **~6.7 FPS** (`target_cycle_time = 0.15s; time.sleep(target_cycle_time - elapsed)`), directly contradicting the claim of 30+ FPS.

---

## 4. Integration Map and Broken Connections

### Main Pipeline Trace
```
CameraFeed (main.py:31)
  │ (Raw BGR frame)
  ▼
VisionSystem.detect (perception/vision.py:65)
  ├─► cv2.resize to 1280px (_YOLO_WIDTH)
  ├─► YOLOv8/11 forward pass (cpu) ─────────────► Raw BBoxes
  ├─► _maybe_start_depth (thread) ──────────────► MonoDepth.compute (perception/mono_depth.py:56)
  │     └─► [BROKEN CACHE] Reused only if len(cached) == len(detections)
  ├─► ObjectTracker.update (perception/tracker.py:78) ──► Greedy IoU Matching + 1D Kalman
  ├─► OpticalFlowEgomotion.update (perception/egomotion.py:38) ──► Lucas-Kanade background flow
  ├─► SpeedEstimator.update (kinematics/speed.py:33) ──► ΔDistance / Δt (compensated)
  └─► TTCCalculator.compute (kinematics/ttc.py:15) ──► EMA TTC
  │ (List[Detection])
  ▼
FeatureDB Label Swap (main.py:275-288)
  │ [BROKEN COORDINATES] Crops frame[y1:y2, x1:x2] using display coords on camera frame
  ▼
DetectionPriorityQueue (core/priority_queue.py:40)
  │ Max-heap drain by TTC -> Distance -> Weight
  ▼
Kinematics Heatmap & Grouping (kinematics/heatmap.py:42)
  │ Directional penalty decay + count phrases
  ▼
SpeechEngine (audio/speech.py:19) + AlertSystem (audio/alert.py:19)
  │ Output spoken guidance + winsound beeps
  ▼
SessionLogger (core/logger.py:12) ──► logs/session_*.jsonl
```

### Broken Connections & Dead Code
1. **Broken Import in Agent Tools:**  
   `agents/tools/vision_tools.py:27` imports `from services.google_maps import get_directions`. `services/google_maps.py` was deleted in commit `ea6805a`. Importing `vision_tools`, `orchestrator`, or `main` raises `ModuleNotFoundError`.
2. **Disconnected Depth Association:**  
   `perception/vision.py:117-124` indexes `cached[i]` directly to `detections[i]`. If the order or count of detections changes during the 300ms background inference, depths are misassigned to wrong objects or completely discarded.
3. **Dead Config Settings:**  
   `core/config.py` defines 7 unused keys: `FRAME_BUDGET_MS`, `FRONT_CAMERA_COOLDOWN`, `GOAL_CONSTANT_COOLDOWN`, `GOAL_SAME_PANEL_COOLDOWN`, `LLM_COOLDOWN`, `MAPS_API_KEY`, `WINDOW_NAME`.
4. **Dead Database Backend:**  
   `services/vector_db.py` contains 9 lines of uncalled stub functions. `FeatureDB` in `core/recognition.py` maintains its own independent storage in `database/`.
5. **Dangling 'G' Key Binding:**  
   `audio/voice_input.py:57` and README claim key `G` activates GPS navigation. No key handler for `G` exists in `main.py:_on_press`.

---

## 5. Top 10 Problems Ranked by Risk

1. **Safety Risk: Entry Point Import Crash (`main.py` Unrunnable)**  
   *Evidence:* `agents/tools/vision_tools.py:27`. Running `python main.py` immediately crashes with `ModuleNotFoundError: No module named 'services.google_maps'`. The entire assistive system cannot be started by a user.
2. **Safety Risk: Arbitrary Distance Doubling in Depth Calculation**  
   *Evidence:* `perception/mono_depth.py:95`. Hardcoded `* 2.0` multiplier doubles distance estimates over calibrated values, causing approaching hazards (cars, stairs, obstacles) to be reported twice as far away as they actually are.
3. **Safety Risk: Spatial Desynchronization in Async Depth Cache**  
   *Evidence:* `perception/vision.py:117-124`. Cached depth values from 300ms prior are assigned to newly detected bounding boxes purely by list index order (`zip(detections, depths)`). A car approaching at 1m can inherit the depth of a distant tree at 8m.
4. **Safety Risk: Cloud STT Dependency in Offline Navigation**  
   *Evidence:* `audio/voice_input.py:130`. Uses Google Cloud STT (`recognize_google`). In an outdoor or low-connectivity environment, all user voice commands and emergency queries fail with `sr.RequestError`.
5. **Correctness Risk: Coordinate Mismatch in FeatureDB Crop Extraction**  
   *Evidence:* `main.py:277-284`. `Detection.box` coordinates are scaled to display resolution (`640x480`), but `frame[y1:y2, x1:x2]` crops the raw camera input (e.g. `1280x720` or `1920x1080`). Crops extract wrong image patches, corrupting recognition embeddings.
6. **Correctness Risk: Unsynchronized Multi-Threading & Race Conditions**  
   *Evidence:* `main.py:141, 157, 257, 289`. `self._latest_frames`, `self._all_dets`, and `self._state` are mutated across the GUI thread, background AI loop, and keyboard listener thread with zero mutex locks.
7. **Correctness Risk: Tracker is Naive IoU Argmax, Not ByteTrack**  
   *Evidence:* `perception/tracker.py:123-132`. Lacks two-stage association for low-confidence detections; tracks frequently drop and re-identify under occlusion, creating severe track ID fragmentation.
8. **Research Validity: Zero Ground-Plane Modeling or Floor Segmentation**  
   *Evidence:* Workspace audit. The system lacks any floor/wall segmentation or 3D geometry representation. It cannot distinguish between a clear floor, a step down, a wall, or traversable space.
9. **Research Validity: Unverified and Throttled Frame Rates**  
   *Evidence:* `main.py:329-331`. Code enforces a sleep cycle targeting $\sim 6.7$ FPS (`target_cycle_time = 0.15s`), directly disproving the claimed 30+ FPS real-time performance.
10. **Research Validity: Complete Absence of Ground Truth or Calibration Data**  
    *Evidence:* `sample-vid/`. All video files are uncalibrated raw MP4s. No camera intrinsic matrices, ground-truth metric distances, depth sensor logs, or evaluation scripts exist anywhere in the repository.

---

## 6. Reuse versus Rewrite Table for the 7-Phase Plan

| Phase | Description | Reusability Verdict | Specific Modules & Rationale |
|---|---|---|---|
| **Phase 1** | Geometry Core (intrinsics, ground plane, ray casting, 3D projections) | **REWRITE** | Existing `perception/mono_depth.py` uses heuristic hacks (`focal = 0.73 * w`, `* 2.0` multiplier). Completely missing camera intrinsic matrices ($K$), ground-plane equations, and 3D back-projection. |
| **Phase 2** | Dataset & Evaluation Harness (ground-truth distances, masks, benchmarks) | **REWRITE (BUILD FROM SCRATCH)** | Nothing exists. `sample-vid/` contains only raw uncalibrated MP4s. Must construct a benchmark dataset with known metric distances, segmentation masks, and error metric evaluation scripts (RMSE, AbsRel, mIoU). |
| **Phase 3** | Perception Wrappers (floor/wall segmentation, metric depth, 3D boxes) | **REUSE WITH CHANGES** | Reuse `perception/vision.py` wrapper structure for YOLO object detection. Replace `MonoDepth` with true metric depth (Metric3D or Depth Anything V2 Metric). Add a dedicated semantic segmentation model for floors and walls. Replace naive IoU tracker with real ByteTrack / 3D Kalman tracker. |
| **Phase 4** | Calibration & Abstention (camera height, pitch, scale calibration, uncertainty) | **REWRITE** | Existing calibration (`mono_depth.py:147`) is an uncalibrated bootstrap on the first detected object that contradicts downstream math. Must implement floor plane fitting for camera height/pitch and an abstention module when uncertainty exceeds thresholds. |
| **Phase 5** | Rolling Map, Planner & Abstention (occupancy grid, traversability, local paths) | **REWRITE (BUILD FROM SCRATCH)** | Completely missing. Deleted `services/google_maps.py` was an external API call. Must build a 2D/2.5D local rolling occupancy grid, traversability costmap from floor masks, local path planner, and safety halt abstention. |
| **Phase 6** | Output & Language Layer (audio alerts, TTS queue, goal system, speech cues) | **REUSE WITH CHANGES** | Reuse `audio/speech.py` (triple-queue TTS, ducking, bypass), `core/priority_queue.py` (max-heap), `kinematics/heatmap.py` (penalty decay), and `core/memory.py` (goal keyword mapping). Fix import in `agents/tools/vision_tools.py`, replace Google Cloud STT with local offline Whisper/Vosk, and implement real confirmation state machine. |
| **Phase 7** | Integration & Study Readiness (telemetry, profiling, test harness) | **REUSE WITH CHANGES** | Reuse `core/logger.py` (JSONL session telemetry is well-structured). Rewrite `main.py` entry point from a monolithic 1,031-line script into modular, lock-protected, study-ready execution pipelines with headless evaluation modes. |

---

## 7. Unverified Items and Requirements to Verify Them

| Item | Status | Exact Requirements to Verify |
|---|---|---|
| Ultralytics YOLO inference on live video | **UNVERIFIABLE** | Requires connected webcam hardware (`cv2.VideoCapture(0)`) or running on video files (prohibited by Prompt 0 rules). Model weights `yolo11n.pt` verified present on disk (5.6MB). |
| Depth Anything V2 ML inference speed & accuracy | **UNVERIFIABLE** | Requires downloading ~100MB HuggingFace model weights (`depth-anything/Depth-Anything-V2-Small-hf`) over network and GPU/CPU profiling. |
| PyAudio microphone pre-warming & capture | **UNVERIFIABLE** | Requires physical audio input hardware (microphone device) connected to host system. |
| Google Speech-to-Text accuracy & latency | **UNVERIFIABLE** | Requires active internet connection and live spoken audio stream. |
| Groq LLaMA 3.3 70B reasoning & tool routing | **UNVERIFIABLE** | Requires active Groq API key (`GROQ_API_KEY` in `.env`) and outbound internet access to `api.groq.com`. |
| pyttsx3 / Windows SAPI5 audio playback | **UNVERIFIABLE** | Requires audio output hardware (speakers/headphones). Mocked in headless testing. |
| End-to-end multi-camera FPS | **UNVERIFIABLE** | Requires running live video decoders across 4 parallel streams on target hardware. |

---

## 8. Recommended Changes to the 7-Phase Plan

1. **Decouple 2D Object Detection from Metric Spatial Guidance:**  
   The existing code attempts to treat 2D bounding boxes as the primary spatial primitive. Phase 1 & 3 should establish a ground-plane coordinate system first; object bounding boxes should be projected onto the estimated ground plane rather than computing pseudo-distances in pixel space.
2. **Mandate Offline Local STT (Whisper.cpp / Vosk):**  
   Remove Google Cloud STT (`speech_recognition.recognize_google`) immediately. For an assistive navigation system, network dropouts must not silence user voice control. Replace with local quantized Whisper or Vosk.
3. **Drop Ad-Hoc Relative Depth Calibration in Favor of Metric Depth:**  
   Do not attempt to calibrate uncalibrated inverse-depth models (like MiDaS / relative Depth Anything) using bounding box heuristics. Modern zero-shot metric depth models (e.g. Metric3D v2, Depth Anything V2 Metric) directly predict metric depth in meters and should be paired with geometric ground plane checks.
4. **Replace Monolithic `main.py` with an Event-Driven or Node Pipeline:**  
   The current architecture interweaves GUI display, background AI processing, keyboard hooks, and speech in shared variables without synchronization. Transition to an explicit queue-based or publisher-subscriber architecture (e.g., lightweight internal message bus or ROS2-style nodes).
5. **Add Strict Ground-Plane Abstention Rule in Phase 4 & 5:**  
   Prioritize the abstention mechanism: if the floor segmentation confidence drops below threshold or camera pitch/roll cannot be verified via ground-plane fitting, the system must verbally abstain ("Terrain unverified, proceed with caution") rather than guessing.
