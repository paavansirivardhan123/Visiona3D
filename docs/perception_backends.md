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
| Promptable floor refinement | `mobile_sam.pt` (MobileSAM ViT-T) | ~40 MB | `weights/mobile-sam/mobile_sam.pt` |

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
python scripts/eval_frames.py --fake --images path/to/frames --out reports/frame_eval
```

MobileSAM is listed by `fetch_models.py` but not downloaded by it: install the
third-party package from https://github.com/ChaoningZhang/MobileSAM and place
`mobile_sam.pt` at `weights/mobile-sam/mobile_sam.pt`. `--yes` prints that
manual step instead of claiming the checkpoint was fetched.

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

## Promptable floor refinement (SegFormer + MobileSAM)

`perception/seg_fusion.py` wraps a segmentation backend with a promptable-mask
refiner. The default wiring is SegFormer-B0 + MobileSAM ViT-T:

1. Run the base segmenter.
2. Select positive prompts from floor pixels with `p_floor >= min_floor_prob`,
   eroded by `erosion_px` so prompts avoid boundaries, and spread over distinct
   grid cells. Selection is deterministic.
3. Query the promptable backend with the prompt points.
4. Accept the mask only if `IoU(mask, base_floor) >= min_accept_iou`; otherwise
   return the base segmentation with a `fallback:*` reason.
5. On acceptance, replace the floor labels with the mask minus sky and minus
   pixels whose base non-floor probability is at least `veto_prob`.
6. Return the base probabilities and sky mask unchanged.

Safety asymmetry:

- The refiner changes labels only. `probs` and `sky_mask` stay SegFormer-only,
  so a promptable mask can never create conformal-certified free space.
- Missing weights, a runtime error, an empty mask or a malformed mask are
  fallbacks, never exceptions and never silent acceptance.
- Promptable refinement cannot verify a mask it never sees; the module does not
  claim the fused floor is safe.

Defaults are `0.60` floor probability, `2` px erosion, `8` prompts, `32` px
minimum spacing, `0.50` acceptance IoU and `0.60` non-floor veto probability.
Every default is TO TUNE ON REAL DATA. `MobileSAMBackend.preflight()` reports a
missing checkpoint or package without importing either.

## Frame evaluation harness

```bash
python scripts/eval_frames.py --fake --images path/to/frames --out reports/frame_eval
python scripts/eval_frames.py --images path/to/frames \
    --labels-dir path/to/labels --depth-dir path/to/depth \
    --out reports/frame_eval
```

Outputs `frame_eval.json` and `frame_eval.csv`. Variants are `segformer`, `sam`
(floor-only, one fixed bottom-centre prompt; not a semantic segmenter) and
`fusion`.

Real-mode ground truth is optional: `<frame stem>.npz/.npy/.png` masks in
`--labels-dir` using ids 0 (unlabeled), 1 floor, 2 wall, 3 obstacle, 4 other,
255 ignore, and `<frame stem>.npz/.npy` metric depth in `--depth-dir`. Frames
without a sidecar get null label/depth metrics. Fake mode renders analytic
ground truth and needs no weights.

| Metric | Definition |
|--------|------------|
| `floor_iou`, `wall_iou`, `mean_iou` | per-class IoU over valid GT pixels |
| `wall_as_floor_rate` | predicted floor on GT wall / all GT wall pixels (unsafe) |
| `unsafe_free_rate` | predicted floor that is non-floor in GT / all predicted floor |
| `floor_recall` | predicted floor on GT floor / all GT floor |
| `floor_fraction` | predicted floor pixels / image pixels (availability) |
| `depth_absrel` | mean abs(pred-gt)/gt on finite GT depth (depth backend only) |
| `latency_ms` | wall-clock per variant call; fusion includes base + refinement |

Synthetic results exercise the harness only. Real accuracy claims require real
frames with supplied masks; this repository reports no real-model numbers.
Timings are machine-specific.

## Fusion threshold tuning

`scripts/tune_fusion.py` chooses the acceptance IoU and non-floor veto
probability on labelled frames. The sweep never selects on held-out data.

```bash
python scripts/tune_fusion.py --fake --images path/to/frames --out reports/fusion_tuning
python scripts/tune_fusion.py --images path/to/frames --labels-dir path/to/labels \
    --route-map path/to/routes.json --calibration-fraction 0.5 --seed 0 \
    --out reports/fusion_tuning
```

The tool runs SegFormer and MobileSAM once per frame, caches the mask, and
re-applies `perception.seg_fusion.apply_floor_mask` — the same function
`SegSAMFusion.fuse` uses — for every threshold pair. Defaults are `0.30` to
`0.80` in steps of `0.10` for `min_accept_iou`, `0.30` to `0.90` in steps of
`0.10` for `veto_prob`, a `0.5` calibration fraction, split seed `0`, `1000`
bootstrap resamples and `0.25` fake wall-as-floor pollution. Every default is
TO TUNE ON REAL DATA.

Splitting is deterministic. Without `--route-map` frames are shuffled by
`--seed` and split into calibration and held-out sets. With `--route-map`
(a JSON object of frame stem to route, covering every frame) whole routes are
assigned to one side; a subset-sum search picks the calibration route set
closest to the target fraction while leaving at least one frame held out. Both
modes require at least `--min-frames` frames (default `6`, TO TUNE ON REAL
DATA).

Selection uses calibration rows only. A configuration is eligible only if its
floor IoU is at least SegFormer-only and its `unsafe_free_rate` and
`wall_as_floor_rate` are not above SegFormer-only. Eligible configurations are
ordered by lowest `unsafe_free_rate`, lowest `wall_as_floor_rate`, highest floor
IoU, highest floor recall, higher acceptance IoU, higher veto probability, then
higher accepted rate. If nothing is eligible, or the best eligible
configuration does not improve on SegFormer-only, the selected variant is
SegFormer-only.

Outputs are `fusion_tuning.json` and `fusion_tuning.csv`. The JSON contains the
split, grid, policy, per-configuration calibration table, the selected
configuration and reason, per-frame summaries and held-out bootstrap 95% CIs.
Held-out numbers are descriptive: they can show `fusion_default` or
SegFormer-only beating the selected configuration, and they are not a safety
guarantee. Fake mode exercises the workflow only; real accuracy still requires
real labelled frames and real weights.

The fake scenes include clean masks, floor-plus-wall spill, empty masks and SAM
errors. With `--fake-wall-as-floor 0` there is nothing for fusion to fix, so a
correct run selects SegFormer-only rather than forcing a change.

```bash
python -m pytest tests/test_tune_fusion.py -q
```

## Runtime fusion configuration

`perception/fusion_config.py` turns a tuning report into a validated runtime
decision. It never fits or changes thresholds.

```bash
python scripts/real_models_smoke.py --images path/to/stills \
    --fusion-config reports/fusion_tuning/fusion_tuning.json \
    --out reports/smoke
```

The end-to-end planner demo consumes the same validated report:

```bash
python scripts/plan_demo.py frame.png --intrinsics camera.json \
    --weights-dir weights --depth-variant indoor \
    --fusion-config reports/fusion_tuning/fusion_tuning.json
```

`plan_demo` passes `--floor-threshold` (default 0.90, TO TUNE ON REAL DATA) to
`PerceptionPipeline`, so free-space certification uses the base probabilities
and geometric veto, never the fused labels. When certification is unavailable
it fails closed (no FREE cells) and warns instead of promoting labels. Passing
`--conformal-depth` loads a validated split-conformal `depth_interval.json`
from `scripts/run_calibration_real.py` and activates the lower-bound obstacle
layer; the loader refuses a fake-mode artifact in a real run and rejects
unbounded or inconsistent quantiles. The pipeline lifts both bounds
(`pessimistic_pointmap_g`, `conformal_upper_pointmap_g`), and the demo reports
nearest-obstacle nominal/lower/upper distances; without an artifact no lower
bound is fabricated and the abstention interval check is skipped.
`run_summary.json` records the fusion config, the per-frame fusion decision,
the depth-interval provenance, certified/vetoed pixel counts and backend
statuses. A real fusion run also preflights MobileSAM and exits `2` with fetch
instructions when it is missing.

For a fake end-to-end check, pass `--fake` together with a report produced by a
fake tuning run. The loader refuses a fake report in a real run, so synthetic
thresholds cannot silently configure real capture.

Validation is about provenance and ranges only:

- schema version `1`, a `real` or `fake` mode, and a selection written by the
  tool;
- for `fusion`, reason `calibration_safety_first_winner`, `eligible=true`,
  `improved=true`, and both thresholds finite in `[0, 1]` and present in the
  report's swept grid;
- for `segformer_only`, one of the tool's two fallback reasons. The returned
  config is disabled, the base segmenter is kept, and no promptable backend is
  needed.

An enabled real run preflights the local MobileSAM checkpoint and package and
exits `2` with fetch instructions if either is unavailable. The smoke report
records `fusion_config` plus a per-image decision (`accepted`, `reason`, IoU,
prompt count, added/removed floor pixels and timings). Fused labels feed the
overlay; probabilities and the sky mask still come from SegFormer, so
`PerceptionPipeline`'s conformal certification is unchanged.

The runtime path is:

```python
from perception.fusion_config import load_tuned_fusion_config, wrap_seg_with_fusion

config = load_tuned_fusion_config("reports/fusion_tuning/fusion_tuning.json")
seg = wrap_seg_with_fusion(SegFormerBackend(weights_dir=...), MobileSAMBackend(...), config)
pipeline = PerceptionPipeline(seg=seg, depth=..., detector=..., K=..., mount=..., estimator=...)
```

Loading a configuration does not validate accuracy or safety, and a
SegFormer-only selection is a valid outcome. Every selected threshold remains
TO TUNE ON REAL DATA.

```bash
python -m pytest tests/test_fusion_config.py -q
```

## Tensor conventions

- RGB `uint8` images `(H, W, 3)`.
- Labels `uint8` `(H, W)` using `core.config` indices.
- Probabilities `float32` `(H, W, 4)` in `[floor, wall, obstacle, other]`
  order, summing to 1; a separate boolean sky mask excludes sky pixels.
- Depth `float32` `(H, W)` metric z-depth in camera frame C, NaN for invalid.
- Intrinsics scale with any resize/letterbox (`scale_intrinsics`); outputs are
  mapped back to the original resolution before use.
- After `SegSAMFusion`, `labels` may be a refined floor map while `probs`
  remain the base segmenter's. Probability-based consumers (conformal
  certification) must keep using `probs`; label consumers see the refinement.
  The two are deliberately not forced to agree.

See `geometry/notation.md` ("Perception conventions") for the canonical rules.

## Modules

| Module | Purpose |
|--------|---------|
| `perception/interfaces.py` | `Det2D`, `SegOutput`, `DepthOutput`, `StageTiming`, `FrameOutputs`, Protocols and `validate()` |
| `perception/class_map.py` | name-based class mapping and default rules |
| `perception/postprocess.py` | stable softmax, class-probability merge, letterbox/resize, NaN-aware map-back, depth sanitising |
| `perception/fakes.py` | analytic fake segmentation/depth/detector and promptable mask with seeded corruptions |
| Fake label directions | `floor_as_wall_fraction` = floor→wall (safe); `wall_as_floor_fraction` = wall→floor (unsafe, pollutes the plane fit) |
| `FakePromptMaskBackend` modes | `floor`, `floor_plus_wall`, `wall`, `empty`, `raise` |
| `perception/seg_backend.py` | SegFormer-B0 ADE20K wrapper (lazy torch/transformers) |
| `perception/seg_fusion.py` | SegFormer + promptable-mask floor refinement, IoU gate, veto and fallbacks |
| `perception/fusion_config.py` | validated runtime selection from a tuning report; fake-report guard and SegFormer-only handling |
| `perception/depth_backend.py` | Depth Anything V2 metric indoor/outdoor wrapper (lazy) |
| `perception/detector_backend.py` | YOLO11-nano wrapper (lazy ultralytics) |
| `perception/pipeline.py` | synchronous non-raising pipeline with stage timings and status codes |
| `scripts/calibrate_camera.py` | checkerboard intrinsics; also accepts ready-made correspondences |
| `scripts/generate_synthetic_calibration.py` | offline synthetic checkerboard grid + intrinsics, tagged `provenance: synthetic` |
| `scripts/fetch_models.py` | model manifest, dry run, download, check and manual MobileSAM notice |
| `scripts/real_models_smoke.py` | user-run real/fake smoke test with overlays, JSON and optional tuned fusion config |
| `scripts/eval_frames.py` | fake/real frame evaluation with JSON + CSV |
| `scripts/tune_fusion.py` | deterministic calibration/held-out split and safety-first fusion threshold sweep with JSON + CSV |

## Calibration

```bash
python scripts/calibrate_camera.py --images calib/ --board 9x6 --square 0.025 \
    --out reports/camera_intrinsics.json --device "phone rear camera"
```

The JSON contains `fx`, `fy`, `cx`, `cy`, distortion coefficients, image size,
RMS reprojection error, per-view errors, board config, date and device label.
Fewer than 10 usable views is refused with a clear error. OpenCV is imported
lazily; `calibrate_from_correspondences` is testable without image files.

### Synthetic calibration (offline only)

When no physical checkerboard capture is available, `scripts/generate_synthetic_calibration.py`
renders a checkerboard grid at a requested resolution, calibrates those views
and writes `camera_intrinsics.json`:

```bash
python scripts/generate_synthetic_calibration.py --width 1042 --height 500 \
    --hfov-deg 70 --out reports/synthetic_calibration/1042x500
```

The output is tagged `"provenance": "synthetic"` and contains a warning.
`plan_demo` emits a `UserWarning` when it loads such a file, and records the
provenance in `run_summary.json`. This exercises the calibration path and
unblocks offline or placeholder demos; it is **not** a measurement of a
physical camera, and metric distances computed from real frames are not
trustworthy. The default 70° horizontal FOV is assumed, not validated.
Regenerate the JSON for each capture resolution: `plan_demo` requires an exact
width and height match.

## Testing

```bash
python -m pytest tests audio -q
python -m pytest tests/test_seg_fusion.py tests/test_eval_frames.py tests/test_tune_fusion.py tests/test_fusion_config.py -q
python -m pytest tests/test_calibrate_camera.py tests/test_synthetic_calibration.py -q
python geometry/scripts/run_checks.py
python geometry/scripts/negative_controls.py
```

The fusion, tuning and harness tests use analytic fakes only: no MobileSAM
package, checkpoint, camera, video or network.

The combined runner prints one table with geometry checks T1-T10 and perception
checks P1-P13. The negative-control script verifies every deliberate bug,
including source-level mutations applied to temporary copies of the package.
