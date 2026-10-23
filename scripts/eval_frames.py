"""Frame-level segmentation comparison: SegFormer, SAM-only and fusion.

Usage:
    python scripts/eval_frames.py --fake --images path/to/frames --out reports/frame_eval
    python scripts/eval_frames.py --images path/to/frames \
        --labels-dir path/to/labels --depth-dir path/to/depth \
        --out reports/frame_eval

Fake mode renders the analytic scene, so every metric has synthetic ground
truth and no weights are needed. Real mode loads local/cached weights only and
exits with code 2 (printing what to fetch) when they are absent.

Variants:
    segformer  SegFormer-B0 labels.
    sam        Prompted MobileSAM floor mask only (no semantics); one fixed
               bottom-centre prompt. This is a floor-only, prompt-dependent
               baseline, not a semantic segmenter.
    fusion     SegFormer + promptable floor refinement (SegSAMFusion).

Real accuracy claims require real frames with supplied ground-truth masks.
Synthetic runs exercise the harness, not the models.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    Intrinsics,
    MASK_FLOOR,
    MASK_OBSTACLE,
    MASK_OTHER,
    MASK_WALL,
)
from evaluation.metrics_depth import compute_depth_metrics  # noqa: E402
from evaluation.metrics_seg import compute_segmentation_metrics  # noqa: E402
from perception.interfaces import SegOutput  # noqa: E402
from perception.seg_fusion import MobileSAMBackend, SegSAMFusion  # noqa: E402
from scripts.real_models_smoke import IMAGE_SUFFIXES, load_image  # noqa: E402

VARIANTS = ("segformer", "sam", "fusion")
FAKE_BOXES = ((-0.25, 0.25, 2.4, 2.8, 0.35, 0.85),)
DEFAULT_WEIGHTS = REPO_ROOT / "weights"
LABEL_SUFFIXES = (".npz", ".npy", ".png")
DEPTH_SUFFIXES = (".npz", ".npy")
METRIC_FIELDS = (
    "floor_iou",
    "wall_iou",
    "mean_iou",
    "unsafe_free_rate",
    "wall_as_floor_rate",
    "floor_recall",
    "floor_fraction",
    "depth_absrel",
)
CSV_FIELDS = (
    "frame",
    "variant",
    "floor_iou",
    "wall_iou",
    "mean_iou",
    "unsafe_free_rate",
    "wall_as_floor_rate",
    "floor_recall",
    "floor_fraction",
    "floor_pixels",
    "depth_absrel",
    "latency_ms",
    "accepted",
    "reason",
    "prompt_count",
)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _empty_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {field: None for field in METRIC_FIELDS}
    metrics["floor_pixels"] = 0
    return metrics


def label_metrics(pred_labels: np.ndarray, gt_labels: np.ndarray | None) -> dict[str, Any]:
    """Return floor/wall IoU and the two unsafe directions for one frame."""
    if gt_labels is None:
        return _empty_metrics()
    pred = np.asarray(pred_labels)
    gt = np.asarray(gt_labels)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction shape {pred.shape} != ground-truth shape {gt.shape}")

    valid = (gt != 0) & (gt != 255)
    if not valid.any():
        return _empty_metrics()

    metrics = compute_segmentation_metrics(pred, gt)
    gt_wall = valid & (gt == MASK_WALL)
    gt_floor = valid & (gt == MASK_FLOOR)
    pred_floor = pred == MASK_FLOOR
    wall_as_floor = (
        float(np.count_nonzero(pred_floor & gt_wall) / np.count_nonzero(gt_wall))
        if gt_wall.any()
        else None
    )
    floor_recall = (
        float(np.count_nonzero(pred_floor & gt_floor) / np.count_nonzero(gt_floor))
        if gt_floor.any()
        else None
    )
    return {
        "floor_iou": _num(metrics["per_class_iou"]["floor"]),
        "wall_iou": _num(metrics["per_class_iou"]["wall"]),
        "mean_iou": _num(metrics["mean_iou"]),
        "unsafe_free_rate": _num(metrics["false_walkable_rate"]),
        "wall_as_floor_rate": _num(wall_as_floor),
        "floor_recall": _num(floor_recall),
        "floor_fraction": float(np.count_nonzero(pred_floor) / pred.size),
        "depth_absrel": None,
        "floor_pixels": int(np.count_nonzero(pred_floor)),
    }


def _floor_only_output(mask: np.ndarray) -> SegOutput:
    """Wrap a promptable floor mask as a floor-only segmentation output."""
    mask_arr = np.asarray(mask, dtype=bool)
    labels = np.full(mask_arr.shape, MASK_OTHER, dtype=np.uint8)
    labels[mask_arr] = MASK_FLOOR
    probs = np.zeros(mask_arr.shape + (4,), dtype=np.float32)
    probs[..., 3] = 1.0
    probs[mask_arr] = 0.0
    probs[mask_arr, 0] = 1.0
    return SegOutput(
        labels=labels,
        probs=probs,
        sky_mask=np.zeros(mask_arr.shape, dtype=bool),
    )


def _bottom_centre_prompt(h: int, w: int) -> np.ndarray:
    return np.array([[(w - 1) * 0.5, (h - 1) * 0.88]], dtype=np.int64)


def _fake_scene(image: np.ndarray):
    from tests.synthetic import render_scene

    h, w = image.shape[:2]
    K = Intrinsics(
        fx=0.9 * w,
        fy=0.9 * w,
        cx=w / 2.0,
        cy=h / 2.0,
        width=w,
        height=h,
    )
    params = dict(
        K=K,
        H=h,
        W=w,
        h=1.4,
        pitch_deg=15.0,
        wall_distance=4.0,
        boxes=FAKE_BOXES,
        seed=7,
    )
    depth, floor, wall, boxes, _gt = render_scene(
        K, h, w, 1.4, 15.0, 4.0, FAKE_BOXES, seed=7
    )
    gt_labels = np.zeros((h, w), dtype=np.uint8)
    gt_labels[floor] = MASK_FLOOR
    gt_labels[wall] = MASK_WALL
    for box_mask in boxes:
        gt_labels[box_mask] = MASK_OBSTACLE
    return params, depth.astype(np.float32), gt_labels


def _find_sidecar(folder: Path, stem: str, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_labels(path: Path) -> np.ndarray:
    from dataset.loader import load_mask

    labels = np.asarray(load_mask(path))
    if labels.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D label mask, got {labels.shape}")
    allowed = {0, MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER, 255}
    unexpected = sorted(set(int(v) for v in np.unique(labels)) - allowed)
    if unexpected:
        raise ValueError(f"{path.name}: label ids {unexpected} outside {sorted(allowed)}")
    return labels


def _load_depth(path: Path) -> np.ndarray:
    from dataset.loader import load_depth

    depth = np.asarray(load_depth(path))
    if depth.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D depth map, got {depth.shape}")
    return depth


def _check_weights(weights_dir: Path, need_depth: bool) -> list[tuple[str, Path]]:
    checks = [
        ("SegFormer-B0 ADE20K", weights_dir / "segformer-b0-ade" / "config.json"),
        ("MobileSAM checkpoint", weights_dir / "mobile-sam" / "mobile_sam.pt"),
    ]
    if need_depth:
        checks.append(
            (
                "Depth Anything V2 metric indoor",
                weights_dir / "depth-anything-v2-metric-indoor-small" / "config.json",
            )
        )
    return [(name, path) for name, path in checks if not path.exists()]


def _aggregate(frames: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [frame["variants"][variant] for frame in frames]
        entry: dict[str, Any] = {}
        for field in METRIC_FIELDS:
            values = [_num(row.get(field)) for row in rows]
            present = [value for value in values if value is not None]
            entry[field] = float(np.mean(present)) if present else None
        latencies = [_num(row.get("latency_ms")) for row in rows]
        latencies = [value for value in latencies if value is not None]
        entry["latency_p50_ms"] = (
            float(np.percentile(latencies, 50)) if latencies else None
        )
        entry["latency_p95_ms"] = (
            float(np.percentile(latencies, 95)) if latencies else None
        )
        entry["n_frames"] = len(rows)
        entry["n_label_metrics"] = sum(
            1 for row in rows if row.get("floor_iou") is not None
        )
        if variant == "fusion":
            accepted = [bool(row.get("accepted")) for row in rows]
            entry["accepted_rate"] = (
                float(sum(accepted) / len(accepted)) if accepted else None
            )
        aggregate[variant] = entry
    return aggregate


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for frame in frames:
            for variant in VARIANTS:
                row = frame["variants"][variant]
                writer.writerow(
                    {
                        "frame": frame["name"],
                        "variant": variant,
                        "floor_iou": _csv_value(row.get("floor_iou")),
                        "wall_iou": _csv_value(row.get("wall_iou")),
                        "mean_iou": _csv_value(row.get("mean_iou")),
                        "unsafe_free_rate": _csv_value(row.get("unsafe_free_rate")),
                        "wall_as_floor_rate": _csv_value(row.get("wall_as_floor_rate")),
                        "floor_recall": _csv_value(row.get("floor_recall")),
                        "floor_fraction": _csv_value(row.get("floor_fraction")),
                        "floor_pixels": _csv_value(row.get("floor_pixels")),
                        "depth_absrel": _csv_value(frame.get("depth_absrel")),
                        "latency_ms": _csv_value(row.get("latency_ms")),
                        "accepted": _csv_value(row.get("accepted")),
                        "reason": _csv_value(row.get("reason")),
                        "prompt_count": _csv_value(row.get("prompt_count")),
                    }
                )


def _print_frame(frame: dict[str, Any]) -> None:
    parts = [f"{frame['name']}:"]
    for variant in VARIANTS:
        row = frame["variants"][variant]
        parts.append(
            f"{variant} floor_iou={_csv_value(row.get('floor_iou')) or '-'} "
            f"unsafe={_csv_value(row.get('unsafe_free_rate')) or '-'}"
        )
    fusion = frame["variants"]["fusion"]
    parts.append(f"accepted={fusion.get('accepted')} reason={fusion.get('reason')}")
    print(" | ".join(parts))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate SegFormer, SAM-only and fusion on a folder of frames"
    )
    parser.add_argument("--images", required=True, help="folder of still frames")
    parser.add_argument("--out", default="reports/frame_eval", help="output folder")
    parser.add_argument("--fake", action="store_true", help="use analytic fake backends")
    parser.add_argument(
        "--fake-wall-as-floor",
        type=float,
        default=0.0,
        help="fake mode only: relabel this fraction of wall pixels as floor",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="real mode: folder of <frame>.npz/.npy/.png ground-truth masks",
    )
    parser.add_argument(
        "--depth-dir",
        default=None,
        help="real mode: folder of <frame>.npz/.npy ground-truth depth maps",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(DEFAULT_WEIGHTS),
        help="real mode: local weights folder",
    )
    args = parser.parse_args(argv)

    image_dir = Path(args.images)
    if not image_dir.is_dir():
        print(f"ERROR: image folder not found: {image_dir}")
        return 2
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(
            f"ERROR: no frames found in {image_dir} "
            f"({', '.join(sorted(IMAGE_SUFFIXES))})"
        )
        return 2

    if not (0.0 <= float(args.fake_wall_as_floor) <= 1.0):
        print("ERROR: --fake-wall-as-floor must be in [0, 1]")
        return 2
    if not args.fake and float(args.fake_wall_as_floor) != 0.0:
        print("ERROR: --fake-wall-as-floor only applies to --fake")
        return 2
    if args.fake and (args.labels_dir or args.depth_dir):
        print("ERROR: --labels-dir/--depth-dir are for real mode; --fake renders ground truth")
        return 2

    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    weights_dir = Path(args.weights_dir)
    for name, folder in (("labels", labels_dir), ("depth", depth_dir)):
        if folder is not None and not folder.is_dir():
            print(f"ERROR: {name} folder not found: {folder}")
            return 2

    seg_backend: Any
    sam_backend: Any
    depth_backend: Any | None = None
    fusion: SegSAMFusion
    if args.fake:
        from perception.fakes import (
            FakeDepthBackend,
            FakePromptMaskBackend,
            FakeSegBackend,
        )
    else:
        missing = _check_weights(weights_dir, need_depth=depth_dir is not None)
        if missing:
            print("ERROR: real model weights are missing:")
            for name, expected in missing:
                print(f"  - {name} (expected at {expected})")
            print("\nFetch them with:")
            print("  python scripts/fetch_models.py --dry-run")
            print("  python scripts/fetch_models.py --yes")
            print("  MobileSAM is a manual download; see docs/perception_backends.md")
            return 2

        from perception.depth_backend import DepthAnythingBackend
        from perception.seg_backend import SegFormerBackend

        seg_backend = SegFormerBackend(weights_dir=weights_dir / "segformer-b0-ade")
        sam_backend = MobileSAMBackend(
            checkpoint=weights_dir / "mobile-sam" / "mobile_sam.pt"
        )
        sam_errors = sam_backend.preflight()
        if sam_errors:
            print("ERROR: MobileSAM is not ready:")
            for error in sam_errors:
                print(f"  - {error}")
            return 2
        if depth_dir is not None:
            depth_backend = DepthAnythingBackend(
                weights_dir=weights_dir / "depth-anything-v2-metric-indoor-small"
            )
        fusion = SegSAMFusion(seg=seg_backend, sam=sam_backend)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    n_with_labels = 0
    n_with_depth = 0

    for path in paths:
        image = load_image(path)
        h, w = image.shape[:2]
        if args.fake:
            params, gt_depth, gt_labels = _fake_scene(image)
            seg_backend = FakeSegBackend(
                **params, wall_as_floor_fraction=float(args.fake_wall_as_floor)
            )
            sam_backend = FakePromptMaskBackend(**params)
            fusion = SegSAMFusion(seg=seg_backend, sam=sam_backend)
            depth_backend = FakeDepthBackend(**params)
        else:
            gt_labels = None
            gt_depth = None
            label_path = (
                _find_sidecar(labels_dir, path.stem, LABEL_SUFFIXES)
                if labels_dir is not None
                else None
            )
            if label_path is not None:
                gt_labels = _load_labels(label_path)
                if gt_labels.shape != (h, w):
                    print(
                        f"ERROR: {label_path.name}: mask shape {gt_labels.shape} "
                        f"!= image {(h, w)}"
                    )
                    return 2
            depth_path = (
                _find_sidecar(depth_dir, path.stem, DEPTH_SUFFIXES)
                if depth_dir is not None
                else None
            )
            if depth_path is not None:
                gt_depth = _load_depth(depth_path)
                if gt_depth.shape != (h, w):
                    print(
                        f"ERROR: {depth_path.name}: depth shape {gt_depth.shape} "
                        f"!= image {(h, w)}"
                    )
                    return 2

        depth_absrel: float | None = None
        if depth_backend is not None and gt_depth is not None:
            try:
                depth_out = depth_backend.predict(image)
            except Exception as exc:  # noqa: BLE001 - report the failing stage
                print(f"ERROR: depth backend failed on {path.name}: {exc}")
                return 2
            valid = (
                np.isfinite(gt_depth)
                & (gt_depth > 0.0)
                & np.isfinite(depth_out.depth_m)
                & (depth_out.depth_m > 0.0)
            )
            depth_absrel = _num(
                compute_depth_metrics(
                    gt_depth, depth_out.depth_m, valid_mask=valid
                )["AbsRel"]
            )

        try:
            seg_start = time.perf_counter()
            seg_output = seg_backend.predict(image)
            seg_ms = (time.perf_counter() - seg_start) * 1000.0
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: segmentation failed on {path.name}: {exc}")
            return 2

        try:
            sam_start = time.perf_counter()
            sam_output = sam_backend.predict(image, _bottom_centre_prompt(h, w))
            sam_ms = (time.perf_counter() - sam_start) * 1000.0
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: SAM-only variant failed on {path.name}: {exc}")
            return 2

        fusion_result = fusion.fuse(image, seg_output)

        variants: dict[str, dict[str, Any]] = {}
        variants["segformer"] = {
            **label_metrics(seg_output.labels, gt_labels),
            "latency_ms": seg_ms,
        }
        variants["sam"] = {
            **label_metrics(_floor_only_output(sam_output.mask).labels, gt_labels),
            "latency_ms": sam_ms,
        }
        variants["fusion"] = {
            **label_metrics(fusion_result.seg.labels, gt_labels),
            "latency_ms": seg_ms + fusion_result.timings_ms["total"],
            "accepted": bool(fusion_result.accepted),
            "reason": fusion_result.reason,
            "prompt_count": int(fusion_result.prompt_count),
            "sam_iou": _num(fusion_result.iou),
            "added_floor_pixels": int(fusion_result.added_floor_pixels),
            "removed_floor_pixels": int(fusion_result.removed_floor_pixels),
        }
        for row in variants.values():
            row["depth_absrel"] = depth_absrel

        if gt_labels is not None:
            n_with_labels += 1
        if gt_depth is not None:
            n_with_depth += 1
        frame = {
            "name": path.name,
            "image_shape": list(image.shape),
            "ground_truth": "synthetic" if args.fake else ("provided" if gt_labels is not None else None),
            "depth_absrel": depth_absrel,
            "variants": variants,
        }
        frames.append(frame)
        _print_frame(frame)

    aggregate = _aggregate(frames)
    report = {
        "schema_version": 1,
        "generated": datetime.datetime.now().isoformat(),
        "mode": "fake" if args.fake else "real",
        "ground_truth": (
            "synthetic"
            if args.fake
            else ("provided" if n_with_labels else "none")
        ),
        "n_frames": len(frames),
        "n_frames_with_labels": n_with_labels,
        "n_frames_with_depth": n_with_depth,
        "variants": list(VARIANTS),
        "weights_dir": None if args.fake else str(weights_dir),
        "aggregate": aggregate,
        "frames": frames,
        "notes": [
            "Synthetic runs exercise the harness; they are not real-world accuracy.",
            "SAM-only is a floor-only, prompt-dependent baseline, not a semantic segmenter.",
            "Latency is wall-clock on this machine and includes per-call image encoding for SAM.",
            "Every numeric default in the perception code is TO TUNE ON REAL DATA.",
        ],
    }
    json_path = out_dir / "frame_eval.json"
    csv_path = out_dir / "frame_eval.csv"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(csv_path, frames)

    print("\nAggregate (mean over frames with finite values):")
    for variant in VARIANTS:
        entry = aggregate[variant]
        print(
            f"  {variant:<9} floor_iou={_csv_value(entry['floor_iou']) or '-'} "
            f"wall_iou={_csv_value(entry['wall_iou']) or '-'} "
            f"unsafe_free={_csv_value(entry['unsafe_free_rate']) or '-'} "
            f"wall_as_floor={_csv_value(entry['wall_as_floor_rate']) or '-'} "
            f"floor_recall={_csv_value(entry['floor_recall']) or '-'} "
            f"p50={_csv_value(entry['latency_p50_ms']) or '-'} ms"
        )
    print(f"Wrote {json_path} and {csv_path}")
    if not args.fake and n_with_labels == 0:
        print("NOTE: no ground-truth masks matched; label metrics are null.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
