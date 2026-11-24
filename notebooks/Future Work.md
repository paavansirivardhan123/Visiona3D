# Future Work: one YOLO26 model instead of YOLO11n + SegFormer

Status date: 2026-11-24. **Measured** = measured on the development laptop (Intel i5-1235U, Iris Xe, about 7.7 GB RAM, CPU only). **Estimate** = my estimate. Nothing here has been field-tested with a camera on a person.

## 1. Goal

Visiona3D today runs two separate perception models next to depth estimation:

- **SegFormer-B0** for floor / wall / obstacle segmentation, and **YOLO11n** for object detection (plus Depth Anything V2 for depth and MobileSAM for floor refinement).
- That is **the old Visiona**. It works, it is tested (see the root `README.md`), and it is kept unchanged as the fallback.

The research goal is **one model** that does both jobs, so the pipeline becomes **YOLO26-based model + Depth Anything V2**:

- an **instance head** (boxes + instance masks, the 80 COCO classes), and
- a **semantic head** for 8 environment classes: `floor, walkable_path, road, wall, glass, staircase, door, sky`.

Target: at least 10 FPS end to end on this CPU-only laptop. **That target is not met and is not claimed** (section 8).

The unified model replaces the old pair **only if it beats it by measurement** (section 7). Otherwise the old pair stays.

## 2. Method (what is implemented)

Code: `visiona_mtl/` (library) and `visiona_unified_perception.ipynb` (thin notebook that calls `visiona_mtl/workflow.py`).

**Architecture** (`model.py`; measured with Ultralytics 8.4.123):

- Shared **YOLO26n-seg trunk**: 24 layers, 3,126,280 parameters. Backbone = layers 0 to 10, neck = layers 11 to 22, original instance head = layer 23, unmodified.
- New **semantic head**: 157,200 parameters, reads the P3 (stride 8) and P4 (stride 16) feature maps, one **sigmoid per class** (8 outputs). Unified model: **3,283,480 parameters**.
- Input: RGB float 0..1, `(B,3,H,W)`, H and W multiples of 32. Train at 320 (local) or 640; deployment candidates 320 / 384 / 416 / 480.
- Freezing experiments: **E1** backbone (layers 0 to 10) frozen, 1,918,008 trainable parameters; **E2** first 6 layers frozen, 3,060,920 trainable; **E3** nothing frozen. Frozen layers also keep their BatchNorm statistics fixed.

**Labels** (`taxonomy.py`, `convert.py`): every dataset is converted to one PNG per image with values `0..7` = the eight classes, **`254`** = labeled but none of the eight (a real negative), **`255`** = unknown (ignored). Each image also records **which classes it can supervise** (a glass-only image supervises only `glass`; a stairs-only image only `staircase`; COCO-Stuff, ADE20K and the sidewalk set supervise all eight). A missing label is therefore never taught as "not that class".

**Loss** (`loss.py`): per-class sigmoid binary cross-entropy + Dice, restricted to the classes each image supervises, at 1/4 resolution, with an auxiliary head (weight 0.4). Total loss = instance loss + semantic loss. Sigmoid outputs are per-pixel scores, **not calibrated probabilities**; the safety layer must be re-calibrated before use.

**Training** (`data.py`, `train.py`): two data streams through one trunk. A COCO batch (Ultralytics loader, mosaic etc.) drives the instance loss; a semantic batch (own loader, letterbox + flip + colour jitter only) drives the semantic loss. AdamW, trunk learning rate x0.1, warm-up + cosine, EMA, checkpoint every epoch (`last_*`, `best_*`, a snapshot every 5 epochs), automatic resume, early stopping on a mean-mIoU validation score.

**Evaluation** (`evaluate.py`): per-class IoU, Boundary IoU, **false-walkable** (share of predicted floor/walkable whose ground truth is labeled non-walkable), glass and staircase miss rates; instance mAP is scored by Ultralytics against the pretrained `yolo26n-seg`. Metrics are unit-tested with known answers.

## 3. Data

All local, converted to the 8 classes, 0 unreadable masks. Class shares are **measured** on a 400-image sample per source.

| Source | Images | walkable_path | road | Notes |
|---|---|---|---|---|
| COCO (objects, 80 classes) | 22,408 train images with polygon labels; 1,000-image validation list | | | chosen for environment content, not a random sample |
| COCO-Stuff | 22,408 train, 1,000 val maps | 4.3% | 4.8% | floor 4.5%, wall 16%, sky 10%, glass 2.0% |
| ADE20K | 20,210 train, 2,000 val | 2.1% | 4.5% | floor 6.4%, wall 15%, glass 1.8%, staircase 0.8% |
| Sidewalk-semantic (Hugging Face, **CC BY-NC**, gated) | 1,000 (900 train, 100 val) | **34%** | **12%** | the main source of walkable_path and road |
| GDD glass | 2,980 train, 936 test | | | glass only; glass fills about 50% of each image (close-ups) |
| Stairs (KAIST, CC BY 4.0) | 788 train, 190 val | | | staircase only; stock photos, some watermarks and near-duplicates |
| Own frames | 194 extracted | | | **not labeled** |

Not used: Sidewalk_1 (Roboflow; boxes only, no usable masks).

**Still missing (and why it matters for a blind user):**

- **Classes with no data and no label yet:** curb, step-down, pothole / open hole, pole, bollard, traffic cone, barrier, tactile paving. COCO has none of these. Object-like ones also need new instance-head classes.
- **More door, glass and stair variety** (each has one narrow source).
- **A labeled test set from your own camera and routes**: 100+ frames. Without it nobody can say the model works for you.
- Candidate sources found by web search (**licence, size and format unverified**): SideGuide (access after agreeing to terms), VIDVIP, Roboflow door and staircase sets (CC BY 4.0 per their listings), StairNet (labels, not masks). From general knowledge, not checked: Cityscapes, Mapillary Vistas, BDD100K, NYU Depth V2, SUN RGB-D, Trans10K.
- **Licences:** the sidewalk set (CC BY-NC) and ADE20K (non-commercial) forbid commercial use; Ultralytics is AGPL-3.0.

## 4. First run (E1): results

Setup (the first, deliberately small run): COCO subset of 3,000 images, 1,754 COCO-Stuff + 1,000 ADE20K + GDD + stairs for the semantic stream, **no sidewalk data**, 320 px, batch 4, backbone frozen, 10,000 steps (20 epochs of about 21 minutes, about 7 hours, **measured**). Raw numbers: `results/E1_first_run_evaluation.json`.

**Objects** (1,000 COCO validation images, 640 px):

| | Box mAP50-95 | Mask mAP50-95 |
|---|---|---|
| Pretrained yolo26n-seg | 0.394 | 0.329 |
| E1 | 0.342 | 0.294 |

Objects got **worse** (mask -0.036, box -0.052).

**Environment** (300 validation images per set, 416 px):

| Class | COCO-Stuff IoU | ADE20K IoU |
|---|---|---|
| floor | 0.32 | 0.61 |
| wall | 0.46 | 0.51 |
| sky | 0.47 | 0.71 |
| door | 0.12 | 0.12 |
| glass | 0.04 | 0.19 |
| staircase | 0.00 | 0.05 |
| walkable_path | 0.00 | 0.00 |
| road | 0.00 | 0.00 |

- Mean IoU: COCO-Stuff 0.18, ADE20K 0.27. Glass-only set (GDD): glass IoU 0.45. Stairs set: staircase IoU 0.83 (single class, easy).
- **Safety numbers:** false-walkable 28% (COCO-Stuff) and 10% (ADE20K) of what the model called walkable; glass pixels missed 83% / 49%; staircase pixels missed 100% / 93%.
- Validation score (mean mIoU over four sets) rose from 0.26 (epoch 0) to about 0.41 and flattened from epoch 9 on.
- **Model-forward speed** (this laptop, busy; unified vs plain yolo26n-seg): 219 vs 243 ms at 320 px, 305 vs 312 ms at 416 px, 323 vs 312 ms at 480 px. The environment head adds almost nothing, but about 300 ms per forward pass is about 3 FPS before depth and the rest of the app.

**Verdict:** E1 does **not** meet the bar for replacing YOLO11n + SegFormer. No pipeline file was changed.

**Likely causes (reasoned, not proven):** almost no walkable_path / road examples in that run (0.1 to 2.5% of pixels), a tiny COCO subset (3,000 of 118,000), 320 px images, and a fully frozen backbone.

## 5. Next experiments (data is already installed)

1. **E1b**: the same recipe with the full data of section 3 (22,408 COCO images, all of ADE20K, the sidewalk set). The notebook uses experiment name `E1b_frozen_backbone_sidewalk`. About 20 minutes per 500 steps (**measured earlier**), so a 30,000-step run is about 15 to 21 hours (**estimate**). Set `epochs=60, time_budget_hours=30`.
2. **E2** (first 6 layers frozen) if E1b's semantic head is still weak. It trains about 60% more parameters, so each step is slower (**not measured**). E3 only if E2 clearly helps.
3. Check **why walkable_path and road scored exactly 0**: data scarcity is the leading explanation, but class mapping and sampling weights should be inspected too.
4. Decide with the project owner how crosswalk, cycling lane, parking driveway, rail track and curb from the sidewalk set are mapped (currently ignored).
5. Add data for the missing classes of section 3, and label 100+ own frames.

## 6. How to run (this computer only)

1. Close heavy apps, plug in the charger, disable sleep. Free RAM was only about 450 MB in my runs; running out of memory is the main risk.
2. Open `visiona_unified_perception.ipynb` in VS Code, kernel **Python 3.14** (`C:\Python314`, Ultralytics 8.4.123).
3. Set `RUN_DOWNLOADS = True` only if datasets are missing; set `RUN_TRAINING = True` to train. Run All. Training resumes from `last_resume.pt` after an interruption.
4. Outputs: `work/checkpoints/<name>/evaluation.json` and `work/final/` (`.pt` files + one ONNX file). Both folders are git-ignored.

## 7. Bar the model must clear before integration (proposal, adjustable)

Judged on **your own labeled frames and the validation sets**, never on training data:

- Object mask mAP50-95 within 0.01 of the pretrained baseline (0.329).
- floor and walkable_path IoU above 0.6, road IoU above 0.5.
- False-walkable below 5%; glass pixel miss below 20%; staircase pixel miss below 15%.

## 8. Integration into Visiona (only after section 7 is met)

1. Add the unified model as a **new backend** next to `perception/seg_backend.py` and `perception/detector_backend.py` (the interfaces are in `perception/interfaces.py`; the runner wires them in `live/runner.py`), selected by a setting. The old SegFormer + YOLO11n path stays as the default fallback.
2. Map the 8 classes to the four labels the safety layer uses (floor, wall, obstacle, sky) and **re-calibrate** the safety layer on the new outputs.
3. Run the existing tests (`tests/`, `audio/`, `geometry/scripts/run_checks.py`), add tests for the new backend, and benchmark the **whole application** on this laptop.
4. Remove the old models only after a full real-world test.

**On 10 FPS:** before any change, the whole application ran at **0.36 FPS** (mean 2.77 s per frame, 180 frames, **measured on a loaded, memory-starved laptop**; see the root `README.md`). SegFormer + YOLO were only about 24% of that time; Depth Anything V2 was about 42% and the floor-plane fit about 16%. Even a free replacement would give about 0.47 FPS. Reaching 10 FPS needs a much faster depth step, cheaper geometry, and probably a lower input size or a smarter frame schedule in addition to the new model. A clean re-run of the whole-app benchmark on an unloaded laptop is also still pending.

## 9. Folder map

- `visiona_unified_perception.ipynb`: the thin notebook. `visiona_mtl/`: model, losses, data, evaluation, converters, `workflow.py`.
- `results/`: saved first-run evaluation, baselines, architecture facts, dataset statistics. `analysis/`: COCO-Stuff label table used by the converters.
- `Datasets/`, `work/`: local data, checkpoints and outputs (git-ignored). Large COCO training images may live on another drive behind a Windows junction.
