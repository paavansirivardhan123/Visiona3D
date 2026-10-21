# Perception backends and tools

Research-only perception interfaces for Visiona-Walk. Real model wrappers are
swappable; every automated test uses analytic fakes. No test loads a model,
opens a camera, reads a video file or touches the network.

## Models

| Role | Model ID | Expected size | Target folder |
|------|----------|---------------|---------------|
| Segmentation | `nvidia/segformer-b0-finetuned-ade-512-512` | ~15 MB | `weights/segformer-b0-ade/` |
| Metric depth (indoor) | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | ~99 MB | `weights/depth-anything-v2-metric-indoor-small/` |
| Metric depth (outdoor) | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | ~99 MB | `weights/depth-anything-v2-metric-outdoor-small/` |
| Detection | `yolo11n.pt` (Ultralytics YOLO11-nano) | ~5.6 MB | repository root or `weights/yolo11n.pt` |

The depth variant is selected with `DepthAnythingBackend(variant="indoor" |
"outdoor")` or the `VISIONA_DEPTH_VARIANT` environment variable (default
`indoor`). The old application's `core/config.py` is read-only; the flag lives
here rather than in that file.

## Fetching and smoke testing

```bash
python scripts/fetch_models.py --dry-run   # default; lists everything, no network
python scripts/fetch_models.py --yes       # downloads into weights/
python scripts/real_models_smoke.py --images path/to/stills --out reports
python scripts/real_models_smoke.py --images path/to/stills --fake
```

`fetch_models.py --check` asks the Hugging Face API whether model IDs resolve;
it is network-using and is not run by agents. The real backends call
`from_pretrained(..., local_files_only=True)`: cached/local weights load
normally, missing weights raise a clear error naming the model and the fetch
command instead of downloading. `real_models_smoke.py` exits with code 2 and
prints exactly what to fetch when weights are absent. Fake mode needs no
weights and is covered by `tests/test_pipeline.py`.

## Default class map

`perception/class_map.py` builds `{source_id: mask_id}` from the model's
`id2label` **names**, never from hard-coded indices.

| Mask class | Default names |
|------------|---------------|
| floor (1) | floor, road, sidewalk, path, rug, carpet, grass, earth, field, sand |
| wall (2) | wall, building, house, skyscraper, fence |
| obstacle (3) | person, car, chair, table, pole, tree, stairs, step, door, windowpane, and similar named objects (see `DEFAULT_RULES`) |
| other (4) | every unknown name; `sky` is also mapped to other and reported via `sky_mask` |

Safety rule: an unknown or unmapped name can never map to floor. A name claimed
by two rules raises an error. Rules are overridable through
`build_class_map(id2label, rules=...)`; a name's target changes with the rule.

## Tensor conventions

- RGB `uint8` images `(H, W, 3)`.
- Labels `uint8` `(H, W)` using `core.config` indices.
- Probabilities `float32` `(H, W, 4)` in `[floor, wall, obstacle, other]`
  order, summing to 1; a separate boolean sky mask excludes sky pixels.
- Depth `float32` `(H, W)` metric z-depth in camera frame C, NaN for invalid.
- Intrinsics scale with any resize/letterbox (`scale_intrinsics`); outputs are
  mapped back to the original resolution before use.

See `geometry/notation.md` ("Perception conventions") for the canonical rules.

## Modules

| Module | Purpose |
|--------|---------|
| `perception/interfaces.py` | `Det2D`, `SegOutput`, `DepthOutput`, `StageTiming`, `FrameOutputs`, Protocols and `validate()` |
| `perception/class_map.py` | name-based class mapping and default rules |
| `perception/postprocess.py` | stable softmax, class-probability merge, letterbox/resize, NaN-aware map-back, depth sanitising |
| `perception/fakes.py` | analytic fake segmentation/depth/detector with seeded corruptions |
| Fake label directions | `floor_as_wall_fraction` = floor→wall (safe); `wall_as_floor_fraction` = wall→floor (unsafe, pollutes the plane fit) |
| `perception/seg_backend.py` | SegFormer-B0 ADE20K wrapper (lazy torch/transformers) |
| `perception/depth_backend.py` | Depth Anything V2 metric indoor/outdoor wrapper (lazy) |
| `perception/detector_backend.py` | YOLO11-nano wrapper (lazy ultralytics) |
| `perception/pipeline.py` | synchronous non-raising pipeline with stage timings and status codes |
| `scripts/calibrate_camera.py` | checkerboard intrinsics; also accepts ready-made correspondences |
| `scripts/fetch_models.py` | model manifest, dry run, download and check |
| `scripts/real_models_smoke.py` | user-run real/fake smoke test with overlays and JSON |

## Calibration

```bash
python scripts/calibrate_camera.py --images calib/ --board 9x6 --square 0.025 \
    --out reports/camera_intrinsics.json --device "phone rear camera"
```

The JSON contains `fx`, `fy`, `cx`, `cy`, distortion coefficients, image size,
RMS reprojection error, per-view errors, board config, date and device label.
Fewer than 10 usable views is refused with a clear error. OpenCV is imported
lazily; `calibrate_from_correspondences` is testable without image files.

## Testing

```bash
python -m pytest tests audio -q
python geometry/scripts/run_checks.py
python geometry/scripts/negative_controls.py
```

The combined runner prints one table with geometry checks T1-T10 and perception
checks P1-P13. The negative-control script verifies every deliberate bug,
including source-level mutations applied to temporary copies of the package.
