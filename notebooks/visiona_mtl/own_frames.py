"""Checks for the user's own labeled frames (the held-out real-world test set). Read-only."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import taxonomy as T

VALID_VALUES = set(range(T.N_SEM)) | {T.OTHER_LABELED, T.IGNORE}


def validate_own_labels(root: Path, label_dir: str = "labels_sem") -> dict:
    """Pair every image in <root>/images with <root>/<label_dir>/<stem>.png and check the encoding.

    Expected mask: single channel, same size as the image, values only in 0..7, 254 (labeled other), 255 (ignore).
    Reports counts, mismatches, illegal values and how much of each class is present (so gaps are visible).
    """
    root = Path(root)
    images = sorted((root / "images").glob("*.jpg"))
    report = dict(images=len(images), labeled=0, missing=[], size_mismatch=[], illegal_values=[], unreadable=[],
                  class_pixels={n: 0 for n in T.SEM_CLASSES}, other_pixels=0, ignore_pixels=0, total_pixels=0)
    for img in images:
        mask_path = root / label_dir / (img.stem + ".png")
        if not mask_path.exists():
            report["missing"].append(img.name)
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            report["unreadable"].append(img.name)
            continue
        if mask.ndim == 3:
            report["illegal_values"].append(f"{img.name}: mask has {mask.shape[2]} channels, expected 1")
            continue
        shape = cv2.imread(str(img)).shape[:2]
        if mask.shape != shape:
            report["size_mismatch"].append(f"{img.name}: image {shape} vs mask {mask.shape}")
            continue
        values = set(int(v) for v in np.unique(mask))
        if not values <= VALID_VALUES:
            report["illegal_values"].append(f"{img.name}: {sorted(values - VALID_VALUES)}")
            continue
        counts = np.bincount(mask.ravel(), minlength=256)
        for c, n in enumerate(T.SEM_CLASSES):
            report["class_pixels"][n] += int(counts[c])
        report["other_pixels"] += int(counts[T.OTHER_LABELED])
        report["ignore_pixels"] += int(counts[T.IGNORE])
        report["total_pixels"] += int(mask.size)
        report["labeled"] += 1
    t = max(report["total_pixels"], 1)
    report["class_share"] = {n: v / t for n, v in report["class_pixels"].items()}
    report["ok"] = report["labeled"] > 0 and not (report["size_mismatch"] or report["illegal_values"] or report["unreadable"])
    return report
