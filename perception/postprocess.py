"""Numpy-only post-processing for perception backends.

Contains stable softmax, class-map probability merging, letterbox/resize
bookkeeping, map-back to original resolution and depth sanitising. No heavy
backend libraries are imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ResizeInfo:
    """Geometry of a resize/letterbox operation.

    Attributes:
        src_h, src_w: Original image size (K of record).
        dst_h, dst_w: Model input size.
        scale: Uniform scale applied to the source.
        pad_x, pad_y: Left/top pad offsets in the model input.
        new_h, new_w: Scaled source size before padding.
    """

    src_h: int
    src_w: int
    dst_h: int
    dst_w: int
    scale: float
    pad_x: int
    pad_y: int
    new_h: int
    new_w: int


def letterbox_info(
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> ResizeInfo:
    """Compute uniform scale and padding for letterboxing src into dst."""
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError("letterbox_info requires positive image dimensions")
    scale = min(float(dst_h) / float(src_h), float(dst_w) / float(src_w))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    pad_y = (int(dst_h) - new_h) // 2
    pad_x = (int(dst_w) - new_w) // 2
    return ResizeInfo(
        src_h=int(src_h),
        src_w=int(src_w),
        dst_h=int(dst_h),
        dst_w=int(dst_w),
        scale=scale,
        pad_x=int(pad_x),
        pad_y=int(pad_y),
        new_h=new_h,
        new_w=new_w,
    )


def plain_resize_info(src_h: int, src_w: int, dst_h: int, dst_w: int) -> ResizeInfo:
    """Resize info for a non-letterboxed, possibly anisotropic resize."""
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError("plain_resize_info requires positive image dimensions")
    return ResizeInfo(
        src_h=int(src_h),
        src_w=int(src_w),
        dst_h=int(dst_h),
        dst_w=int(dst_w),
        scale=float(dst_w) / float(src_w),
        pad_x=0,
        pad_y=0,
        new_h=int(dst_h),
        new_w=int(dst_w),
    )


def _nearest_indices(in_size: int, out_size: int) -> np.ndarray:
    positions = (np.arange(out_size, dtype=np.float64) + 0.5) * (float(in_size) / float(out_size))
    return np.clip(np.floor(positions).astype(np.int64), 0, in_size - 1)


def resize_nearest(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resize for 2D label/mask arrays."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"resize_nearest expects a 2D array, got {arr.shape}")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("resize_nearest requires positive output dimensions")
    rows = _nearest_indices(arr.shape[0], int(out_h))
    cols = _nearest_indices(arr.shape[1], int(out_w))
    return arr[np.ix_(rows, cols)]


def _bilinear_weights(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = (np.arange(out_size, dtype=np.float64) + 0.5) * (float(in_size) / float(out_size)) - 0.5
    positions = np.clip(positions, 0.0, float(in_size - 1))
    i0 = np.floor(positions).astype(np.int64)
    i1 = np.minimum(i0 + 1, in_size - 1)
    frac = positions - i0
    return i0, i1, frac


def resize_bilinear_nanaware(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize for 2D float maps; NaN pixels do not leak into neighbours."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"resize_bilinear_nanaware expects a 2D array, got {arr.shape}")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("resize_bilinear_nanaware requires positive output dimensions")

    y0, y1, wy = _bilinear_weights(arr.shape[0], int(out_h))
    x0, x1, wx = _bilinear_weights(arr.shape[1], int(out_w))

    finite = np.isfinite(arr)
    values = np.where(finite, arr, 0.0)

    wy_col = wy[:, None]
    wx_row = wx[None, :]
    corner_weights = (
        (1.0 - wy_col) * (1.0 - wx_row),
        (1.0 - wy_col) * wx_row,
        wy_col * (1.0 - wx_row),
        wy_col * wx_row,
    )
    corners = (
        (y0, x0),
        (y0, x1),
        (y1, x0),
        (y1, x1),
    )

    numerator = np.zeros((out_h, out_w), dtype=np.float64)
    denominator = np.zeros((out_h, out_w), dtype=np.float64)
    for weight, (rows, cols) in zip(corner_weights, corners):
        numerator += weight * values[np.ix_(rows, cols)]
        denominator += weight * finite[np.ix_(rows, cols)]

    result = np.full((out_h, out_w), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result.astype(np.float32)


def map_back(
    mask_or_depth: np.ndarray,
    resize_info: ResizeInfo,
    kind: str = "label",
) -> np.ndarray:
    """Crop letterbox padding and resize an output back to the original image.

    Args:
        mask_or_depth: 2D array at model output resolution.
        resize_info: Geometry used when the image was resized/letterboxed.
        kind: ``"label"`` (nearest neighbour) or ``"depth"`` (NaN-aware bilinear).

    Returns:
        Array of shape ``(resize_info.src_h, resize_info.src_w)``.
    """
    arr = np.asarray(mask_or_depth)
    if arr.ndim != 2:
        raise ValueError(f"map_back expects a 2D array, got {arr.shape}")
    if arr.shape != (resize_info.dst_h, resize_info.dst_w):
        raise ValueError(
            "map_back input shape does not match resize_info dst size: "
            f"{arr.shape} vs {(resize_info.dst_h, resize_info.dst_w)}"
        )

    cropped = arr[
        resize_info.pad_y : resize_info.pad_y + resize_info.new_h,
        resize_info.pad_x : resize_info.pad_x + resize_info.new_w,
    ]
    if kind == "label":
        return resize_nearest(cropped, resize_info.src_h, resize_info.src_w)
    if kind == "depth":
        return resize_bilinear_nanaware(cropped, resize_info.src_h, resize_info.src_w)
    raise ValueError(f"kind must be 'label' or 'depth', got {kind!r}")


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax; finite logits up to +-1e4 stay finite."""
    values = np.asarray(logits, dtype=np.float64)
    row_max = np.max(values, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(row_max), row_max, 0.0)
    shifted = values - safe_max
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exp_values = np.exp(shifted)
    denominator = np.sum(exp_values, axis=axis, keepdims=True)
    out = np.full_like(exp_values, 1.0 / exp_values.shape[axis])
    np.divide(exp_values, denominator, out=out, where=denominator > 0.0)
    # If an entire row was non-finite, fall back to a uniform distribution.
    bad = ~np.isfinite(out)
    if bad.any():
        out = np.where(bad, 1.0 / exp_values.shape[axis], out)
    total = np.sum(out, axis=axis, keepdims=True)
    np.divide(out, total, out=out, where=total > 0.0)
    return out.astype(np.float32)


def merge_class_probs(
    logits: np.ndarray,
    class_map: Mapping[int, int],
) -> np.ndarray:
    """Softmax source logits and merge channels into [floor, wall, obstacle, other].

    Args:
        logits: float array (..., C) of model logits in source class order.
        class_map: Mapping source class id -> mask id (1..4).

    Returns:
        float32 (..., 4) probabilities summing to 1 along the last axis.
    """
    logits_arr = np.asarray(logits)
    if logits_arr.ndim < 1:
        raise ValueError("merge_class_probs expects an array with a class axis")
    n_classes = logits_arr.shape[-1]
    probs = stable_softmax(logits_arr, axis=-1)

    merged = np.zeros(logits_arr.shape[:-1] + (4,), dtype=np.float64)
    for source_id, mask_id in class_map.items():
        source_id = int(source_id)
        mask_id = int(mask_id)
        if source_id < 0 or source_id >= n_classes:
            raise ValueError(f"class_map source id {source_id} outside 0..{n_classes - 1}")
        if mask_id not in (1, 2, 3, 4):
            raise ValueError(f"class_map target mask id {mask_id} outside 1..4")
        merged[..., mask_id - 1] += probs[..., source_id]

    total = np.sum(merged, axis=-1, keepdims=True)
    np.divide(merged, total, out=merged, where=total > 0.0)
    return merged.astype(np.float32)


def labels_from_probs(probs: np.ndarray) -> np.ndarray:
    """Return uint8 mask labels (1..4) as argmax of merged channel probabilities."""
    probs_arr = np.asarray(probs)
    if probs_arr.shape[-1] != 4:
        raise ValueError(f"expected 4 probability channels, got {probs_arr.shape}")
    return (np.argmax(probs_arr, axis=-1) + 1).astype(np.uint8)


def sanitize_depth(
    depth: np.ndarray,
    max_range_m: float,
    sky_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return float32 depth with NaN for every invalid or excluded pixel.

    Invalid includes non-finite values (NaN, inf), values <= 0, values beyond
    ``max_range_m`` and sky pixels. Valid pixels are left untouched.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        raise ValueError(f"sanitize_depth expects a 2D array, got {depth_arr.shape}")
    max_range = float(max_range_m)
    if not np.isfinite(max_range) or max_range <= 0.0:
        raise ValueError("max_range_m must be a positive finite number")

    invalid = ~np.isfinite(depth_arr) | (depth_arr <= 0.0) | (depth_arr > max_range)
    if sky_mask is not None:
        sky = np.asarray(sky_mask, dtype=bool)
        if sky.shape != depth_arr.shape:
            raise ValueError(
                f"sky_mask shape {sky.shape} does not match depth shape {depth_arr.shape}"
            )
        invalid |= sky

    result = depth_arr.copy()
    result[invalid] = np.nan
    return result


__all__ = [
    "ResizeInfo",
    "letterbox_info",
    "plain_resize_info",
    "resize_nearest",
    "resize_bilinear_nanaware",
    "map_back",
    "stable_softmax",
    "merge_class_probs",
    "labels_from_probs",
    "sanitize_depth",
]
