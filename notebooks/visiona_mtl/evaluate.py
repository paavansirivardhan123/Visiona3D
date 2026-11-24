"""Semantic evaluation: per-class IoU, Boundary IoU and the navigation-safety rates.

Ground-truth encoding (taxonomy.py): 0..7 classes, 254 = labeled but none of the 8, 255 = ignore.
Predictions come from loss.compose(): 0..7, or 255 = "no class is confident".

Definitions (all counted only on pixels where the image supervises the class and the ground truth is not ignore):
* IoU_c            = TP / (TP + FP + FN)
* Boundary IoU_c   = IoU between the GT and predicted boundary bands (width = 2% of the image diagonal)
* false-walkable   = of the pixels predicted floor or walkable_path, the share whose GT is labeled and NOT walkable
                     (road, wall, glass, staircase, door, sky or 254). This is the safety-critical number.
* false-walkable rate on GT negatives = the same count divided by all labeled non-walkable GT pixels
* miss rate_c      = 1 - pixel recall of class c; and image-level: images where GT has >= min_area of the class but the
                     prediction covers < min_cover of that GT area
"""
from __future__ import annotations

import cv2
import numpy as np

from . import taxonomy as T

WALKABLE = (T.SEM_ID["floor"], T.SEM_ID["walkable_path"])


def _boundary(mask: np.ndarray, d: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    inner = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=d)
    return (m - inner).astype(bool)


def boundary_iou(gt_c: np.ndarray, pr_c: np.ndarray, ratio: float = 0.02) -> float | None:
    d = max(1, int(round(ratio * float(np.hypot(*gt_c.shape)))))
    g, p = _boundary(gt_c, d), _boundary(pr_c, d)
    union = (g | p).sum()
    return None if union == 0 else float((g & p).sum() / union)


class SemanticMeter:
    """Accumulate over images, then call .summary()."""

    def __init__(self, n_classes: int = T.N_SEM, min_area: float = 0.01, min_cover: float = 0.5):
        self.n = n_classes
        self.tp = np.zeros(n_classes, np.int64)
        self.fp = np.zeros(n_classes, np.int64)
        self.fn = np.zeros(n_classes, np.int64)
        self.bi_sum = np.zeros(n_classes)
        self.bi_n = np.zeros(n_classes, np.int64)
        self.img_present = np.zeros(n_classes, np.int64)
        self.img_missed = np.zeros(n_classes, np.int64)
        self.walk_pred = 0
        self.walk_fp = 0
        self.neg_gt = 0
        self.images = 0
        self.min_area, self.min_cover = min_area, min_cover

    def update(self, pred: np.ndarray, gt: np.ndarray, supervised) -> None:
        """pred, gt: (H,W) uint8. supervised: iterable of class ids this image labels (both yes and no)."""
        valid = gt != T.IGNORE
        self.images += 1
        for c in supervised:
            g = (gt == c) & valid
            p = (pred == c) & valid
            self.tp[c] += (g & p).sum()
            self.fp[c] += (p & ~g).sum()
            self.fn[c] += (g & ~p).sum()
            if g.any():
                bi = boundary_iou(g, p)
                if bi is not None:
                    self.bi_sum[c] += bi
                    self.bi_n[c] += 1
                if g.mean() >= self.min_area:
                    self.img_present[c] += 1
                    if (g & p).sum() / g.sum() < self.min_cover:
                        self.img_missed[c] += 1
        # false-walkable: only meaningful where BOTH walkable classes are supervised (fully labeled images)
        if all(c in set(supervised) for c in WALKABLE):
            walk_pred = np.isin(pred, WALKABLE) & valid
            walk_gt = np.isin(gt, WALKABLE)
            self.walk_pred += int(walk_pred.sum())
            self.walk_fp += int((walk_pred & ~walk_gt).sum())
            self.neg_gt += int((valid & ~walk_gt).sum())

    def summary(self) -> dict:
        iou = {}
        for c, name in enumerate(T.SEM_CLASSES[: self.n]):
            denom = self.tp[c] + self.fp[c] + self.fn[c]
            rec_den = self.tp[c] + self.fn[c]
            iou[name] = {
                "iou": None if denom == 0 else float(self.tp[c] / denom),
                "pixel_recall": None if rec_den == 0 else float(self.tp[c] / rec_den),
                "pixel_miss_rate": None if rec_den == 0 else float(self.fn[c] / rec_den),
                "boundary_iou": None if self.bi_n[c] == 0 else float(self.bi_sum[c] / self.bi_n[c]),
                "image_miss_rate": None if self.img_present[c] == 0 else float(self.img_missed[c] / self.img_present[c]),
                "images_with_class": int(self.img_present[c]),
            }
        ious = [v["iou"] for v in iou.values() if v["iou"] is not None]
        return {
            "images": self.images,
            "miou": None if not ious else float(np.mean(ious)),
            "per_class": iou,
            "false_walkable_of_predicted_walkable": None if self.walk_pred == 0 else self.walk_fp / self.walk_pred,
            "false_walkable_rate_on_nonwalkable_gt": None if self.neg_gt == 0 else self.walk_fp / self.neg_gt,
            "predicted_walkable_pixels": self.walk_pred,
        }
