"""Convert each source dataset to the project's semantic PNG encoding (see taxonomy.py) plus coverage metadata.

Images are never copied. Each converter writes only
    <out_dir>/sem/<stem>.png        one uint8 class id per pixel (0..7, 254 = labeled other, 255 = ignore)
    <out_dir>/coverage.jsonl        one record per image: image path, mask path, supervised classes, source, flags
so disk use stays small (an 8-class PNG is a few KB).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from . import taxonomy as T


# ------------------------------------------------------------------------------------------------
# lookup tables
# ------------------------------------------------------------------------------------------------
def _lut(default: int) -> np.ndarray:
    return np.full(256, default, dtype=np.uint8)


def cocostuff_lut(labels_txt: str | Path) -> np.ndarray:
    """COCO-Stuff stuffthingmaps pixel value p -> project id. p = label id - 1; 255 = unlabeled in the source."""
    names = {}
    for line in Path(labels_txt).read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition(":")
        if k.strip().isdigit():
            names[int(k)] = v.strip()
    lut = _lut(T.OTHER_LABELED)                # things and unmapped stuff: labeled, none of the 8
    lut[255] = T.IGNORE
    for pixel in range(0, 182):
        name = names.get(pixel + 1)
        if name == "unlabeled":
            lut[pixel] = T.IGNORE
        elif name in T.COCOSTUFF_TO_SEM:
            lut[pixel] = T.SEM_ID[T.COCOSTUFF_TO_SEM[name]]
    return lut


def ade20k_lut() -> np.ndarray:
    """ADE20K annotation value k (1..150) -> project id. 0 = unlabeled in the source."""
    lut = _lut(T.OTHER_LABELED)
    lut[0] = T.IGNORE
    for k, name in enumerate(T.ADE20K_NAMES, start=1):
        if name in T.ADE20K_TO_SEM:
            lut[k] = T.SEM_ID[T.ADE20K_TO_SEM[name]]
    return lut


def hf_sidewalk_lut() -> np.ndarray:
    """Hugging Face sidewalk-semantic id (0..34) -> project id."""
    lut = _lut(T.OTHER_LABELED)
    for k, name in enumerate(T.HF_SIDEWALK_NAMES):
        if name in T.HF_SIDEWALK_IGNORE or name in T.HF_SIDEWALK_REVIEW:
            lut[k] = T.IGNORE                  # unknown / undecided: neither positive nor negative
        elif name in T.HF_SIDEWALK_TO_SEM:
            lut[k] = T.SEM_ID[T.HF_SIDEWALK_TO_SEM[name]]
    lut[35:] = T.IGNORE
    return lut


def apply_lut(label_img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if label_img.ndim == 3:                    # some PNG exports repeat the label in all channels
        label_img = label_img[..., 0]
    return lut[label_img]


# ------------------------------------------------------------------------------------------------
# writers
# ------------------------------------------------------------------------------------------------
def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(str(path), str(base)).replace("\\", "/")


def mask_path(record: dict, base: Path) -> Path:
    """Absolute path of a record's semantic mask. Records store it RELATIVE to the dataset root, so they stay valid
    when the folder moves (different drive, Colab) and when the working directory changes."""
    return Path(base) / record["sem_mask"]


def _record(source: str, image: Path, mask: Path, base: Path, supervised, objects_labeled: bool, note: str = ""):
    return dict(
        source=source, image=_rel(image, base), sem_mask=_rel(mask, base),
        supervised=list(supervised), objects_labeled=bool(objects_labeled), note=note,
    )


def write_coverage(path: Path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def read_coverage(path: Path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def convert_png_masks(source: str, images_dir: Path, masks_dir: Path, out_dir: Path, lut: np.ndarray, base: Path,
                      image_ext=(".jpg", ".jpeg", ".png"), mask_ext=".png", limit: int | None = None,
                      objects_labeled: bool = False):
    """Convert a dataset that ships one label PNG per image (COCO-Stuff, ADE20K, HF export). Returns (records, pixel_counts)."""
    images_dir, masks_dir, out_dir = Path(images_dir), Path(masks_dir), Path(out_dir)
    (out_dir / "sem").mkdir(parents=True, exist_ok=True)
    records, counts = [], np.zeros(256, dtype=np.int64)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in image_ext)
    for n, img in enumerate(files):
        if limit is not None and n >= limit:
            break
        src = masks_dir / (img.stem + mask_ext)
        if not src.exists():
            continue
        label = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if label is None:
            continue
        sem = apply_lut(label, lut)
        dst = out_dir / "sem" / (img.stem + ".png")
        cv2.imwrite(str(dst), sem)
        counts += np.bincount(sem.ravel(), minlength=256)
        records.append(_record(source, img, dst, base, T.SUPERVISED[source], objects_labeled))
    return records, counts


def convert_gdd(source: str, root: Path, out_dir: Path, base: Path, limit: int | None = None):
    """GDD: glass mask (255 = glass). Glass -> id 4. Everything else -> 254 (NOT glass; other classes unknown).

    Only the glass class is supervised in these images (SUPERVISED[source] == [glass]).
    """
    root, out_dir = Path(root), Path(out_dir)
    (out_dir / "sem").mkdir(parents=True, exist_ok=True)
    glass = T.SEM_ID["glass"]
    records, counts = [], np.zeros(256, dtype=np.int64)
    for n, img in enumerate(sorted((root / "image").glob("*.jpg"))):
        if limit is not None and n >= limit:
            break
        src = root / "mask" / (img.stem + ".png")
        mask = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        sem = np.where(mask > 0, glass, T.OTHER_LABELED).astype(np.uint8)
        dst = out_dir / "sem" / (img.stem + ".png")
        cv2.imwrite(str(dst), sem)
        counts += np.bincount(sem.ravel(), minlength=256)
        records.append(_record(source, img, dst, base, T.SUPERVISED[source], False,
                               "only glass is labeled; non-glass pixels are negatives for glass only"))
    return records, counts


def convert_yolo_polygons(source: str, images_dir: Path, labels_dir: Path, out_dir: Path, class_to_sem: dict, base: Path,
                          limit: int | None = None):
    """Roboflow/YOLO polygon labels -> semantic PNG. Polygon pixels get their class id, the rest 254 (negative for the
    supervised class only). Rows that are not polygons (boxes) are counted and skipped, never guessed."""
    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    (out_dir / "sem").mkdir(parents=True, exist_ok=True)
    records, counts, skipped_rows = [], np.zeros(256, dtype=np.int64), 0
    for n, img in enumerate(sorted(images_dir.glob("*.jpg"))):
        if limit is not None and n >= limit:
            break
        arr = cv2.imread(str(img))
        if arr is None:
            continue
        h, w = arr.shape[:2]
        sem = np.full((h, w), T.OTHER_LABELED, dtype=np.uint8)
        lab = labels_dir / (img.stem + ".txt")
        if lab.exists():
            for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 7 or len(parts) % 2 == 0:          # a polygon is 1 + 2n numbers with n >= 3
                    skipped_rows += 1
                    continue
                cls = int(float(parts[0]))
                if cls not in class_to_sem:
                    continue
                pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                cv2.fillPoly(sem, [np.round(pts).astype(np.int32)], T.SEM_ID[class_to_sem[cls]])
        dst = out_dir / "sem" / (img.stem + ".png")
        cv2.imwrite(str(dst), sem)
        counts += np.bincount(sem.ravel(), minlength=256)
        records.append(_record(source, img, dst, base, T.SUPERVISED[source], False,
                               "only the listed classes are labeled; the rest are negatives for them"))
    return records, counts, skipped_rows
