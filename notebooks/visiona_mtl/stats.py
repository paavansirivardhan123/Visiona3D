"""Dataset analysis for the converted semantic masks: class pixel shares, image presence, overlays."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import taxonomy as T

PALETTE = {
    0: (80, 200, 80), 1: (60, 220, 200), 2: (60, 60, 230), 3: (230, 140, 60), 4: (230, 230, 60),
    5: (200, 60, 200), 6: (60, 140, 230), 7: (200, 200, 200), T.OTHER_LABELED: (70, 70, 70), T.IGNORE: (0, 0, 0),
}


def summarize(records, base: Path, sample: int | None = None, seed: int = 0):
    """Read converted masks and summarize. `sample` limits how many are read (random, reproducible)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    if sample is not None and sample < len(records):
        idx = np.sort(rng.choice(len(records), size=sample, replace=False))
    pix = np.zeros(256, dtype=np.int64)
    present = np.zeros(256, dtype=np.int64)
    area_when_present = [[] for _ in range(T.N_SEM)]
    sizes = []
    unreadable = 0
    for i in idx:
        sem = cv2.imread(str(Path(base) / records[i]["sem_mask"]), cv2.IMREAD_UNCHANGED)
        if sem is None:
            unreadable += 1
            continue
        counts = np.bincount(sem.ravel(), minlength=256)
        pix += counts
        present += (counts > 0)
        sizes.append(sem.shape[:2])
        for c in range(T.N_SEM):
            if counts[c] > 0:
                area_when_present[c].append(counts[c] / sem.size)
    n = len(sizes)
    total = max(int(pix.sum()), 1)
    out = dict(
        images_read=n, images_total=len(records), unreadable_masks=unreadable,
        supervised_sets=sorted({tuple(r["supervised"]) for r in records}),
        objects_labeled_fraction=float(np.mean([r["objects_labeled"] for r in records])) if records else 0.0,
        image_size_min=list(map(int, np.min(sizes, axis=0))) if sizes else None,
        image_size_max=list(map(int, np.max(sizes, axis=0))) if sizes else None,
        pixel_share_other_labeled=float(pix[T.OTHER_LABELED] / total),
        pixel_share_ignore=float(pix[T.IGNORE] / total),
        classes={},
    )
    for c, name in enumerate(T.SEM_CLASSES):
        out["classes"][name] = dict(
            pixel_share=float(pix[c] / total),
            images_with_class=int(present[c]),
            images_with_class_fraction=float(present[c] / max(n, 1)),
            mean_area_when_present=float(np.mean(area_when_present[c])) if area_when_present[c] else 0.0,
        )
    return out


def overlay(image_bgr: np.ndarray, sem: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend class colours onto the image. 254 is dark grey, 255 (ignore) is left untouched."""
    colour = np.zeros_like(image_bgr)
    for k, rgb in PALETTE.items():
        colour[sem == k] = rgb[::-1]
    mask = (sem != T.IGNORE)[..., None]
    return np.where(mask, (alpha * colour + (1 - alpha) * image_bgr).astype(np.uint8), image_bgr)


def montage(records, base: Path, out_path: Path, n: int = 6, width: int = 320, seed: int = 1):
    """Save a grid of image | overlay pairs for visual quality control."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(records), size=min(n, len(records)), replace=False)
    rows = []
    for i in pick:
        r = records[i]
        img = cv2.imread(str(Path(base) / r["image"]))
        sem = cv2.imread(str(Path(base) / r["sem_mask"]), cv2.IMREAD_UNCHANGED)
        if img is None or sem is None:
            continue
        if img.shape[:2] != sem.shape[:2]:
            sem = cv2.resize(sem, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        h = int(img.shape[0] * width / img.shape[1])
        a = cv2.resize(img, (width, h))
        b = cv2.resize(overlay(img, sem), (width, h))
        rows.append(np.hstack([a, b]))
    h_min = min(r.shape[0] for r in rows)
    rows = [r[:h_min] for r in rows]
    cols = 2
    grid = [np.hstack(rows[i:i + cols]) for i in range(0, len(rows) - len(rows) % cols, cols)]
    sheet = np.vstack(grid)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return sheet.shape
