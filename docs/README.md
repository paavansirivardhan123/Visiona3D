# Visiona3D documentation

Start with the root [README](../README.md) for installation, the seven startup
questions, the keys and the full option reference. The documents here go
deeper.

| Document | What it covers |
| --- | --- |
| [`live_system.md`](live_system.md) | The one-process live loop: stages, frame/time synchronisation, HUD, saved session map and trail estimation, zone layer, goals and voice, SQLite schema, measured latency |
| [`perception_backends.md`](perception_backends.md) | Segmentation, depth and detection backends, model IDs and folders, fake backends, SegFormer + MobileSAM refinement, evaluation and tuning |
| [`mapping_planning.md`](mapping_planning.md) | Frame and grid conventions, BEV certification, costmap, A*, path smoothing, guidance decisions, synthetic results |
| [`annotation_guide.md`](annotation_guide.md) | Dataset labelling: mask class indices, per-class rules, special cases, frame sampling, distance ground truth |
| [`capture_protocol.md`](capture_protocol.md) | Physical capture setup: camera height and pitch, checkerboard calibration, routes, recording format, split rules, privacy |
| [`../geometry/notation.md`](../geometry/notation.md) | Frame definitions and sign conventions shared by every module |

Suggested reading order for a new contributor: this page, the root README,
`live_system.md`, then `mapping_planning.md` and `perception_backends.md` for
the layer you are changing.

Every numeric default in these documents is a starting point marked
`TO TUNE ON REAL DATA`. None is a validated accuracy claim, and the automated
results in `mapping_planning.md` are synthetic unless stated otherwise.
