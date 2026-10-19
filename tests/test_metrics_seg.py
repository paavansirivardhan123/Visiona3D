"""Test D5: Segmentation metrics with hand-made 4x4 masks."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import (
    MASK_FLOOR,
    MASK_WALL,
    MASK_OBSTACLE,
    MASK_OTHER,
    MASK_UNLABELED,
    MASK_IGNORE,
)
from evaluation.metrics_seg import compute_segmentation_metrics, compute_segmentation_metrics_per_frame


def test_d5_hand_made_iou():
    """D5: Hand-made 4x4 masks give hand-computed IoU."""
    # 4x4 test masks
    # GT:  floor(1) at top-left 2x2, wall(2) at top-right 2x2, obstacle(3) at bottom-left 2x2, other(4) at bottom-right 2x2
    gt = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.uint8)

    # Pred: perfect match
    pred = gt.copy()

    metrics = compute_segmentation_metrics(pred, gt)
    assert metrics["mean_iou"] == 1.0
    for cls in ["floor", "wall", "obstacle", "other"]:
        assert metrics["per_class_iou"][cls] == 1.0
        assert metrics["precision"][cls] == 1.0
        assert metrics["recall"][cls] == 1.0


def test_d5_partial_overlap():
    """D5: Partial overlap gives correct IoU."""
    gt = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.uint8)

    # Pred: floor shifted right by 1
    pred = np.array([
        [0, 1, 1, 2],
        [0, 1, 1, 2],
        [0, 3, 3, 4],
        [0, 3, 3, 4],
    ], dtype=np.uint8)

    metrics = compute_segmentation_metrics(pred, gt)

    # Floor: GT has 4 pixels, Pred has 4 pixels (cols 1-2), Intersection = 2 (top-middle 2x1), Union = 6
    assert metrics["per_class_iou"]["floor"] == pytest.approx(2/6)
    # Wall: GT has 4 pixels (cols 2-3), Pred has 2 pixels (col 3 only), Intersection = 2, Union = 4
    assert metrics["per_class_iou"]["wall"] == pytest.approx(2/4)
    # Obstacle: GT has 4, Pred has 4 (cols 1-2), Intersection = 2, Union = 6
    assert metrics["per_class_iou"]["obstacle"] == pytest.approx(2/6)
    # Other: GT has 4, Pred has 4 (cols 1-2), Intersection = 2, Union = 6
    assert metrics["per_class_iou"]["other"] == pytest.approx(2/6)


def test_d5_unlabeled_ignore_never_count():
    """D5: Labels 0 (unlabeled) and 255 (ignore) never count in metrics."""
    gt = np.array([
        [1, 1, 0, 255],
        [1, 1, 0, 255],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.uint8)

    pred = np.array([
        [1, 1, 1, 2],  # Predicts floor on unlabeled and ignore
        [1, 1, 1, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.uint8)

    metrics = compute_segmentation_metrics(pred, gt)

    # Floor IoU: GT floor = 4 pixels, Pred floor = 6 pixels, Intersection = 4 (only where GT is floor)
    # Union = 6 (4 GT + 2 extra pred on unlabeled/ignore don't count since GT not valid there)
    # Wait - unlabeled and ignore are excluded from valid_gt, so pred on those shouldn't affect
    # Valid GT = where gt != 0 and gt != 255
    # GT floor valid = 4 pixels
    # Pred floor valid = 4 pixels (only where valid_gt)
    # Intersection = 4, Union = 4
    assert metrics["per_class_iou"]["floor"] == 1.0


def test_d5_false_walkable_rate():
    """D5: False-walkable rate equals hand-computed fraction."""
    # GT: wall at top, floor at bottom
    gt = np.array([
        [2, 2, 2, 2],  # wall
        [2, 2, 2, 2],
        [1, 1, 1, 1],  # floor
        [1, 1, 1, 1],
    ], dtype=np.uint8)

    # Pred: predicts floor everywhere
    pred = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.uint8)

    metrics = compute_segmentation_metrics(pred, gt)

    # Predicted floor = 16 pixels
    # False walkable = predicted floor where GT is wall/obstacle/other = 8 pixels (top half)
    # FW rate = 8/16 = 0.5
    assert metrics["false_walkable_rate"] == pytest.approx(0.5)


def test_d5_false_walkable_rate_range_restricted():
    """D5: Range-restricted false-walkable rate matches hand computation."""
    gt = np.array([
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.uint8)

    pred = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.uint8)

    # Depth: top half at 1m (bin 0-2m), bottom half at 5m (bin 4-6m)
    depth = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0],
    ], dtype=np.float64)

    metrics = compute_segmentation_metrics(pred, gt, depth=depth)

    # Overall FW rate = 0.5
    assert metrics["false_walkable_rate"] == pytest.approx(0.5)

    # Bin 0-2m: top half only (wall in GT, pred floor)
    # Pred floor in bin = 8 pixels, false walkable = 8
    assert metrics["false_walkable_rate_by_bin"]["0-2m"] == pytest.approx(1.0)

    # Bin 4-6m: bottom half only (floor in GT, pred floor)
    # Pred floor in bin = 8 pixels, false walkable = 0
    assert metrics["false_walkable_rate_by_bin"]["4-6m"] == pytest.approx(0.0)


def test_d5_per_frame_aggregation():
    """D5: Per-frame aggregation matches manual computation."""
    gt1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    pred1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    gt2 = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    pred2 = np.array([[1, 1], [2, 2]], dtype=np.uint8)

    metrics = compute_segmentation_metrics_per_frame([pred1, pred2], [gt1, gt2])

    # Each frame perfect -> overall perfect
    assert metrics["mean_iou"] == 1.0
    for cls in ["floor", "wall", "obstacle", "other"]:
        assert metrics["per_class_iou"][cls] == 1.0