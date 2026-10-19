"""Segmentation metrics for Visiona3D.

Computes per-class IoU, mean IoU, precision/recall, and FALSE-WALKABLE RATE.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.config import (
    MASK_FLOOR,
    MASK_WALL,
    MASK_OBSTACLE,
    MASK_OTHER,
    MASK_UNLABELED,
    MASK_IGNORE,
    MASK_EVAL_CLASSES,
    RANGE_BIN_EDGES,
    range_bin_index,
)


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    depth: Optional[np.ndarray] = None,
    range_bin: Optional[str] = None,
) -> dict[str, Any]:
    """Compute segmentation metrics.

    Args:
        pred_mask: Predicted mask of shape (H, W) with class indices.
        gt_mask: Ground truth mask of shape (H, W) with class indices.
        depth: Optional depth map of shape (H, W) for range-bin restricted metrics.
        range_bin: Optional range bin name ("0-2m", "2-4m", "4-6m", "6+m") to restrict evaluation.

    Returns:
        Dict with keys:
        - "per_class_iou": dict mapping class_name -> IoU
        - "mean_iou": mean IoU over present classes
        - "precision": dict class_name -> precision
        - "recall": dict class_name -> recall
        - "false_walkable_rate": overall FW rate
        - "false_walkable_rate_by_bin": dict bin_name -> FW rate (if depth provided)
    """
    assert pred_mask.shape == gt_mask.shape, "Pred and GT mask shapes must match"
    H, W = pred_mask.shape

    # Valid pixels: not unlabeled (0) and not ignore (255) in GT
    valid_gt = (gt_mask != MASK_UNLABELED) & (gt_mask != MASK_IGNORE)

    # Range bin restriction
    if depth is not None and range_bin is not None:
        bin_idx = {"0-2m": 0, "2-4m": 1, "4-6m": 2, "6+m": 3}.get(range_bin)
        if bin_idx is not None:
            bin_mask = np.zeros_like(depth, dtype=bool)
            for i, edge in enumerate(RANGE_BIN_EDGES[:-1]):
                if i == bin_idx:
                    bin_mask = (depth >= edge) & (depth < RANGE_BIN_EDGES[i + 1])
                    break
            if bin_idx == 3:  # 6+m
                bin_mask = depth >= 6.0
            valid_gt = valid_gt & bin_mask

    # Compute per-class metrics
    per_class_iou = {}
    precision = {}
    recall = {}

    # Classes to evaluate: floor, wall, obstacle, other (indices 1, 2, 3, 4)
    for cls_idx in MASK_EVAL_CLASSES:
        cls_name = {1: "floor", 2: "wall", 3: "obstacle", 4: "other"}[cls_idx]

        pred_cls = (pred_mask == cls_idx) & valid_gt
        gt_cls = (gt_mask == cls_idx) & valid_gt

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        pred_count = pred_cls.sum()
        gt_count = gt_cls.sum()

        if union > 0:
            per_class_iou[cls_name] = float(intersection / union)
        else:
            per_class_iou[cls_name] = float("nan")

        if pred_count > 0:
            precision[cls_name] = float(intersection / pred_count)
        else:
            precision[cls_name] = float("nan")

        if gt_count > 0:
            recall[cls_name] = float(intersection / gt_count)
        else:
            recall[cls_name] = float("nan")

    # Mean IoU over present classes (classes with GT pixels)
    present_ious = [v for k, v in per_class_iou.items() if not np.isnan(v)]
    mean_iou = float(np.mean(present_ious)) if present_ious else float("nan")

    # FALSE-WALKABLE RATE: fraction of predicted-floor pixels whose truth is wall, obstacle, or other
    pred_floor = (pred_mask == MASK_FLOOR) & valid_gt
    if pred_floor.sum() > 0:
        # Among predicted floor, count how many are actually wall, obstacle, or other in GT
        false_walkable = np.logical_and(
            pred_floor,
            (gt_mask == MASK_WALL) | (gt_mask == MASK_OBSTACLE) | (gt_mask == MASK_OTHER)
        ).sum()
        false_walkable_rate = float(false_walkable / pred_floor.sum())
    else:
        false_walkable_rate = float("nan")

    result = {
        "per_class_iou": per_class_iou,
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "false_walkable_rate": false_walkable_rate,
    }

    # Per-bin false walkable rate if depth provided
    if depth is not None:
        fw_by_bin = {}
        for bin_name in ["0-2m", "2-4m", "4-6m", "6+m"]:
            bin_idx = {"0-2m": 0, "2-4m": 1, "4-6m": 2, "6+m": 3}[bin_name]
            bin_mask = np.zeros_like(depth, dtype=bool)
            if bin_idx < 3:
                bin_mask = (depth >= RANGE_BIN_EDGES[bin_idx]) & (depth < RANGE_BIN_EDGES[bin_idx + 1])
            else:
                bin_mask = depth >= RANGE_BIN_EDGES[3]
            valid_bin = valid_gt & bin_mask
            pred_floor_bin = (pred_mask == MASK_FLOOR) & valid_bin
            if pred_floor_bin.sum() > 0:
                false_walkable_bin = np.logical_and(
                    pred_floor_bin,
                    (gt_mask == MASK_WALL) | (gt_mask == MASK_OBSTACLE) | (gt_mask == MASK_OTHER)
                ).sum()
                fw_by_bin[bin_name] = float(false_walkable_bin / pred_floor_bin.sum())
            else:
                fw_by_bin[bin_name] = float("nan")
        result["false_walkable_rate_by_bin"] = fw_by_bin

    return result


def compute_segmentation_metrics_per_frame(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    depths: Optional[list[np.ndarray]] = None,
) -> dict[str, Any]:
    """Compute segmentation metrics aggregated over multiple frames.

    Args:
        pred_masks: List of predicted masks (H, W).
        gt_masks: List of ground truth masks (H, W).
        depths: Optional list of depth maps for per-bin metrics.

    Returns:
        Aggregated metrics dict with same structure as compute_segmentation_metrics
        but averaged over frames.
    """
    n_frames = len(pred_masks)
    assert n_frames == len(gt_masks)
    if depths is not None:
        assert n_frames == len(depths)

    # Accumulate counts
    class_names = ["floor", "wall", "obstacle", "other"]
    class_indices = [MASK_FLOOR, MASK_WALL, MASK_OBSTACLE, MASK_OTHER]

    # Global counts
    intersection = {c: 0 for c in class_names}
    union = {c: 0 for c in class_names}
    pred_count = {c: 0 for c in class_names}
    gt_count = {c: 0 for c in class_names}

    pred_floor_total = 0
    false_walkable_total = 0

    # Per-bin counts
    bin_names = ["0-2m", "2-4m", "4-6m", "6+m"]
    pred_floor_bin = {b: 0 for b in bin_names}
    false_walkable_bin = {b: 0 for b in bin_names}

    for i in range(n_frames):
        pred = pred_masks[i]
        gt = gt_masks[i]
        valid_gt = (gt != MASK_UNLABELED) & (gt != MASK_IGNORE)

        depth = depths[i] if depths is not None else None

        for cname, cidx in zip(class_names, class_indices):
            pred_c = (pred == cidx) & valid_gt
            gt_c = (gt == cidx) & valid_gt
            inter = np.logical_and(pred_c, gt_c).sum()
            un = np.logical_or(pred_c, gt_c).sum()
            intersection[cname] += inter
            union[cname] += un
            pred_count[cname] += pred_c.sum()
            gt_count[cname] += gt_c.sum()

        # False walkable overall
        pred_floor = (pred == MASK_FLOOR) & valid_gt
        pred_floor_total += pred_floor.sum()
        false_walkable_total += np.logical_and(
            pred_floor,
            (gt == MASK_WALL) | (gt == MASK_OBSTACLE) | (gt == MASK_OTHER)
        ).sum()

        # Per-bin
        if depth is not None:
            for b_idx, bname in enumerate(bin_names):
                if b_idx < 3:
                    bin_mask = (depth >= RANGE_BIN_EDGES[b_idx]) & (depth < RANGE_BIN_EDGES[b_idx + 1])
                else:
                    bin_mask = depth >= RANGE_BIN_EDGES[3]
                valid_bin = valid_gt & bin_mask
                pred_floor_b = (pred == MASK_FLOOR) & valid_bin
                pred_floor_bin[bname] += pred_floor_b.sum()
                false_walkable_bin[bname] += np.logical_and(
                    pred_floor_b,
                    (gt == MASK_WALL) | (gt == MASK_OBSTACLE) | (gt == MASK_OTHER)
                ).sum()

    # Compute final metrics
    per_class_iou = {}
    precision = {}
    recall = {}
    for cname in class_names:
        if union[cname] > 0:
            per_class_iou[cname] = intersection[cname] / union[cname]
        else:
            per_class_iou[cname] = float("nan")
        if pred_count[cname] > 0:
            precision[cname] = intersection[cname] / pred_count[cname]
        else:
            precision[cname] = float("nan")
        if gt_count[cname] > 0:
            recall[cname] = intersection[cname] / gt_count[cname]
        else:
            recall[cname] = float("nan")

    present_ious = [v for v in per_class_iou.values() if not np.isnan(v)]
    mean_iou = float(np.mean(present_ious)) if present_ious else float("nan")

    false_walkable_rate = (
        float(false_walkable_total / pred_floor_total) if pred_floor_total > 0 else float("nan")
    )

    result = {
        "per_class_iou": per_class_iou,
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "false_walkable_rate": false_walkable_rate,
    }

    if depths is not None:
        fw_by_bin = {}
        for bname in bin_names:
            if pred_floor_bin[bname] > 0:
                fw_by_bin[bname] = false_walkable_bin[bname] / pred_floor_bin[bname]
            else:
                fw_by_bin[bname] = float("nan")
        result["false_walkable_rate_by_bin"] = fw_by_bin

    return result