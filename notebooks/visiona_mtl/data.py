"""Data streams for the unified model.

Two streams feed one shared trunk (nothing here trains anything):

* instance stream: the standard Ultralytics COCO loader (objects fully labeled, all Ultralytics augmentations)
* semantic stream: this module. Image + 8-class semantic PNG + the list of classes the image supervises.

They are kept separate on purpose: mosaic / perspective augmentation would have to be applied identically to an
image, its polygons and its semantic mask. Separate streams avoid that, and avoid ever treating an image whose
objects are not fully labeled as a "no objects" example.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from . import taxonomy as T


def letterbox_pair(img: np.ndarray, sem: np.ndarray | None, size: int, train: bool, rng: np.random.Generator):
    """Resize (keeping aspect) and pad image + mask to size x size. Mask is resized with nearest and padded with 255."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if train:
        scale *= float(rng.uniform(0.6, 1.3))
    nh, nw = max(int(round(h * scale)), 8), max(int(round(w * scale)), 8)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if sem is not None:
        sem = cv2.resize(sem, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas_sem = np.full((size, size), T.IGNORE, dtype=np.uint8)
    # place: centred when evaluating, random offset (crop if larger than the canvas) when training
    def offsets(n, m):
        if n <= m:
            return (int(rng.integers(0, m - n + 1)) if train else (m - n) // 2), 0, n
        o = int(rng.integers(0, n - m + 1)) if train else (n - m) // 2
        return 0, o, m
    y0, sy, hh = offsets(nh, size)
    x0, sx, ww = offsets(nw, size)
    canvas[y0:y0 + hh, x0:x0 + ww] = img[sy:sy + hh, sx:sx + ww]
    if sem is not None:
        canvas_sem[y0:y0 + hh, x0:x0 + ww] = sem[sy:sy + hh, sx:sx + ww]
    return canvas, canvas_sem


class SemanticDataset(Dataset):
    """Images with a converted semantic mask. `records` come from coverage.jsonl (see convert.py)."""

    def __init__(self, records, base: Path, imgsz: int = 640, augment: bool = True, seed: int = 0):
        self.records = list(records)
        self.base = Path(base)
        self.imgsz = int(imgsz)
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        rng = np.random.default_rng((self.seed, i, int(torch.initial_seed() % 2**31)))
        img = cv2.imread(str(self.base / r["image"]), cv2.IMREAD_COLOR)
        sem = cv2.imread(str(self.base / r["sem_mask"]), cv2.IMREAD_UNCHANGED)
        if img is None or sem is None:
            raise FileNotFoundError(f"unreadable pair: {r['image']} / {r['sem_mask']}")
        if sem.shape[:2] != img.shape[:2]:
            sem = cv2.resize(sem, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, sem = letterbox_pair(img, sem, self.imgsz, self.augment, rng)
        if self.augment:
            if rng.random() < 0.5:                                  # horizontal flip only: left/right are symmetric,
                img, sem = img[:, ::-1], sem[:, ::-1]               # up/down is NOT (floor is below, sky above)
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.6, 1.4), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.6, 1.4), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        sup = torch.zeros(T.N_SEM, dtype=torch.bool)
        sup[list(r["supervised"])] = True
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous(),   # uint8 (3,H,W)
            "sem": torch.from_numpy(np.ascontiguousarray(sem)),                                  # uint8 (H,W)
            "supervised": sup,
            "source": r["source"],
        }


def collate_semantic(batch):
    return {
        "img": torch.stack([b["img"] for b in batch]),
        "sem": torch.stack([b["sem"] for b in batch]),
        "supervised": torch.stack([b["supervised"] for b in batch]),
        "source": [b["source"] for b in batch],
    }


def load_coverage(path: Path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def make_semantic_loader(sources: dict, base: Path, batch: int, imgsz: int = 640, augment: bool = True,
                         workers: int = 0, seed: int = 0, num_samples: int | None = None):
    """sources: {name: {"records": [...], "weight": float}}. `weight` is the share of the sampling mass the SOURCE gets,
    so rare sources (stairs, glass) can be oversampled without copying files. Returns a DataLoader."""
    datasets, per_item = [], []
    for name, spec in sources.items():
        ds = SemanticDataset(spec["records"], base, imgsz, augment, seed)
        datasets.append(ds)
        per_item.append(np.full(len(ds), spec["weight"] / max(len(ds), 1), dtype=np.float64))
    full = ConcatDataset(datasets)
    weights = torch.from_numpy(np.concatenate(per_item))
    sampler = WeightedRandomSampler(weights, num_samples=num_samples or len(full), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    return DataLoader(full, batch_size=batch, sampler=sampler, num_workers=workers, collate_fn=collate_semantic,
                      drop_last=True, persistent_workers=workers > 0)


def make_instance_loader(data_yaml: str, imgsz: int, batch: int, mode: str = "train", workers: int = 0,
                         fraction: float = 1.0, overrides: dict | None = None):
    """The standard Ultralytics COCO segmentation loader (objects fully labeled). `data_yaml` must not contain a
    `download:` block (so nothing is ever fetched implicitly)."""
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    cfg = get_cfg(DEFAULT_CFG, overrides={"task": "segment", "imgsz": imgsz, "batch": batch, "fraction": fraction,
                                          "workers": workers, **(overrides or {})})
    data = check_det_dataset(data_yaml, autodownload=False)
    split = data["train"] if mode == "train" else data["val"]
    ds = build_yolo_dataset(cfg, split, batch, data, mode=mode, rect=False, stride=32)
    loader = build_dataloader(ds, batch, workers, shuffle=(mode == "train"))
    return loader, data
