"""User-run smoke test for the real perception backends.

    python scripts/real_models_smoke.py --images path/to/stills --out reports
    python scripts/real_models_smoke.py --images path/to/stills --fake

Real mode loads local/cached weights only and exits with code 2 (printing what
to fetch) when weights are missing. Fake mode needs no model and is covered by
the test suite. Overlay PNGs and a JSON summary are written to the output dir.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import struct
import sys
import time
import zlib
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import Intrinsics  # noqa: E402
from perception.interfaces import SegOutput  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}
DEFAULT_WEIGHTS = REPO_ROOT / "weights"

SEGMENTATION_COLORS = {
    0: (0, 0, 0),        # unlabeled / sky
    1: (40, 180, 70),    # floor  (green)
    2: (60, 90, 220),    # wall   (blue)
    3: (220, 60, 50),    # obstacle (red)
    4: (140, 140, 140),  # other  (gray)
}


def _parse_ppm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        raise ValueError(f"{path.name}: only binary P6 PPM is supported by the stdlib loader")
    tokens: list[bytes] = []
    index = 2
    while len(tokens) < 3:
        while index < len(data) and data[index : index + 1].isspace():
            index += 1
        if index < len(data) and data[index : index + 1] == b"#":
            while index < len(data) and data[index : index + 1] != b"\n":
                index += 1
            continue
        start = index
        while index < len(data) and not data[index : index + 1].isspace():
            index += 1
        tokens.append(data[start:index])
    while index < len(data) and data[index : index + 1].isspace():
        index += 1
    width, height, maxval = (int(token) for token in tokens)
    if maxval != 255:
        raise ValueError(f"{path.name}: only 8-bit PPM is supported")
    pixels = np.frombuffer(data[index : index + width * height * 3], dtype=np.uint8)
    if pixels.size != width * height * 3:
        raise ValueError(f"{path.name}: truncated PPM payload")
    return pixels.reshape(height, width, 3).copy()


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".ppm":
        return _parse_ppm(path)
    from PIL import Image  # lazy, user-run tool only

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an 8-bit RGB PNG using only the standard library."""
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _yolo_weights(weights_dir: Path) -> Path:
    """Prefer the managed weights folder, then the repository-root checkpoint."""
    managed = weights_dir / "yolo11n.pt"
    if managed.exists():
        return managed
    return REPO_ROOT / "yolo11n.pt"


def check_weights(weights_dir: Path) -> list[dict[str, str]]:
    """Return missing-weight entries for the real backends."""
    missing = []
    checks = [
        ("nvidia/segformer-b0-finetuned-ade-512-512", weights_dir / "segformer-b0-ade" / "config.json"),
        (
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            weights_dir / "depth-anything-v2-metric-indoor-small" / "config.json",
        ),
        ("yolo11n.pt", _yolo_weights(weights_dir)),
    ]
    for model_id, expected in checks:
        if not expected.exists():
            missing.append({"model": model_id, "expected_path": str(expected)})
    return missing


def _make_fakes(image: np.ndarray):
    from perception.fakes import FakeDepthBackend, FakeDetector, FakeSegBackend

    height, width = image.shape[:2]
    K = Intrinsics(
        fx=0.9 * width,
        fy=0.9 * width,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )
    boxes = [(-0.25, 0.25, 2.4, 2.8, 0.35, 0.85)]
    common = dict(K=K, H=height, W=width, h=1.4, pitch_deg=15.0, wall_distance=4.0, boxes=boxes, seed=7)
    return FakeSegBackend(**common), FakeDepthBackend(**common), FakeDetector(**common)


def _make_real(weights_dir: Path):
    from perception.depth_backend import DepthAnythingBackend
    from perception.detector_backend import YoloDetectorBackend
    from perception.seg_backend import SegFormerBackend

    seg = SegFormerBackend(weights_dir=weights_dir / "segformer-b0-ade")
    depth = DepthAnythingBackend(weights_dir=weights_dir / "depth-anything-v2-metric-indoor-small")
    detector = YoloDetectorBackend(weights=_yolo_weights(weights_dir))
    return seg, depth, detector


def _overlay(image: np.ndarray, seg: SegOutput, detections: list) -> np.ndarray:
    overlay = image.copy()
    labels = np.asarray(seg.labels)
    for class_id, color in SEGMENTATION_COLORS.items():
        mask = labels == class_id
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.asarray(color)).astype(np.uint8)
    for det in detections:
        x1, y1, x2, y2 = np.asarray(det.box_xyxy).astype(int).tolist()
        h, w = overlay.shape[:2]
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        overlay[y1 : max(y1 + 1, y2), x1 : max(x1 + 2, x2)] = (255, 220, 0)
        overlay[y1 : max(y1 + 1, y2), x1 : max(x1 + 2, x1 + 2)] = (255, 220, 0)
        overlay[max(0, y2 - 2) : max(1, y2), x1 : max(x1 + 2, x2)] = (255, 220, 0)
    return overlay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test real or fake perception backends")
    parser.add_argument("--images", required=True, help="folder of still images")
    parser.add_argument("--out", default="reports", help="output folder for JSON + overlays")
    parser.add_argument("--fake", action="store_true", help="use fake analytic backends")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS), help="local weights folder")
    args = parser.parse_args(argv)

    image_dir = Path(args.images)
    if not image_dir.is_dir():
        print(f"ERROR: image folder not found: {image_dir}")
        return 2
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"ERROR: no images found in {image_dir} ({', '.join(sorted(IMAGE_SUFFIXES))})")
        return 2

    weights_dir = Path(args.weights_dir)
    if not args.fake:
        missing = check_weights(weights_dir)
        if missing:
            print("ERROR: real model weights are missing:")
            for item in missing:
                print(f"  - {item['model']} (expected at {item['expected_path']})")
            print("\nFetch them with:")
            print("  python scripts/fetch_models.py --dry-run")
            print("  python scripts/fetch_models.py --yes")
            print("Model ids: nvidia/segformer-b0-finetuned-ade-512-512,")
            print("  depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf,")
            print("  depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf, yolo11n.pt")
            return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated": datetime.datetime.now().isoformat(),
        "fake": bool(args.fake),
        "images": [],
    }

    for path in paths:
        image = load_image(path)
        height, width = image.shape[:2]
        if args.fake:
            seg_backend, depth_backend, detector_backend = _make_fakes(image)
        else:
            seg_backend, depth_backend, detector_backend = _make_real(weights_dir)

        timings: dict[str, float] = {}
        start = time.perf_counter()
        seg = seg_backend.predict(image)
        timings["seg_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        depth = depth_backend.predict(image)
        timings["depth_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        detections = detector_backend.predict(image)
        timings["detector_ms"] = (time.perf_counter() - start) * 1000.0

        labels = np.asarray(seg.labels)
        histogram = {
            name: int(np.sum(labels == class_id))
            for class_id, name in ((1, "floor"), (2, "wall"), (3, "obstacle"), (4, "other"))
        }
        depth_arr = np.asarray(depth.depth_m)
        finite = depth_arr[np.isfinite(depth_arr)]
        depth_min = float(np.min(finite)) if finite.size else None
        depth_max = float(np.max(finite)) if finite.size else None

        overlay_path = out_dir / f"{path.stem}_overlay.png"
        write_png(overlay_path, _overlay(image, seg, detections))

        entry = {
            "name": path.name,
            "image_shape": list(image.shape),
            "seg_shape": list(labels.shape),
            "depth_shape": list(depth_arr.shape),
            "depth_min_m": depth_min,
            "depth_max_m": depth_max,
            "class_histogram": histogram,
            "n_detections": len(detections),
            "timings_ms": timings,
            "overlay": overlay_path.name,
        }
        report["images"].append(entry)
        print(
            f"{path.name}: image={image.shape} seg={labels.shape} depth={depth_arr.shape} "
            f"range=[{depth_min}, {depth_max}] m classes={histogram} dets={len(detections)} "
            f"timings={timings}"
        )

    json_path = out_dir / "real_models_smoke.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {json_path} and {len(paths)} overlay PNG(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
