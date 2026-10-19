"""Dataset loading utilities.

Provides functions to load frames, routes, and iterate over datasets.
Supports masks from .npz or .npy files; optional PNG via lazy cv2 import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Optional
import json
import numpy as np

from dataset.schema import FrameRecord, DistancePoint, MountInfo, Tags, validate


def _load_npz_or_npy(path: Path) -> np.ndarray:
    """Load array from .npz (first array) or .npy file."""
    if path.suffix == ".npz":
        with np.load(path) as data:
            # Return first array
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"NPZ file {path} contains no arrays")
            return data[keys[0]]
    elif path.suffix == ".npy":
        return np.load(path)
    else:
        raise ValueError(f"Unsupported mask/depth file format: {path.suffix}. Use .npz or .npy")


def _load_mask_from_npz(path: Path) -> np.ndarray:
    """Load mask from NPZ file, preferring 'mask' or 'floor_mask' key."""
    with np.load(path) as data:
        keys = list(data.keys())
        # Prefer explicit mask keys
        for preferred in ["mask", "floor_mask", "segmentation", "semantic"]:
            if preferred in keys:
                return data[preferred]
        # Fall back to first non-depth key
        for k in keys:
            if k != "depth":
                return data[k]
        # Fall back to first key
        return data[keys[0]]


def _try_load_png(path: Path) -> Optional[np.ndarray]:
    """Try to load mask as PNG using lazy cv2 import. Returns None if cv2 not available or fails."""
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return img
    except Exception:
        return None


def load_mask(mask_path: Path) -> np.ndarray:
    """Load segmentation mask from file.

    Supports .npz, .npy, and optionally .png (via cv2).
    Returns integer array of shape (H, W) with class indices.
    """
    if mask_path.suffix.lower() == ".png":
        arr = _try_load_png(mask_path)
        if arr is not None:
            return arr
        # Fall through to error
    if mask_path.suffix == ".npz":
        arr = _load_mask_from_npz(mask_path)
    else:
        arr = _load_npz_or_npy(mask_path)
    if arr.dtype != np.uint8 and arr.dtype != np.int32 and arr.dtype != np.int64:
        # Convert to integer class indices
        arr = arr.astype(np.int64)
    return arr


def load_depth(depth_path: Path) -> np.ndarray:
    """Load depth map from file.

    Supports .npz, .npy.
    Returns float array of shape (H, W) with depth in metres.
    """
    arr = _load_npz_or_npy(depth_path)
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return arr


def load_frame(dataset_root: Union[str, Path], frame_record: FrameRecord) -> dict[str, Any]:
    """Load all data for a single frame.

    Args:
        dataset_root: Root directory of the dataset.
        frame_record: FrameRecord with paths relative to dataset_root.

    Returns:
        Dict with keys: 'rgb_path', 'depth', 'mask', 'intrinsics', 'mount', 'tags', 'distance_points', 'valid_mask'.
    """
    root = Path(dataset_root)

    # Load depth
    depth = load_depth(root / frame_record.depth_path)

    # Load mask
    mask = load_mask(root / frame_record.mask_path)

    # Validate shapes match
    if depth.shape != mask.shape:
        raise ValueError(
            f"Depth shape {depth.shape} != mask shape {mask.shape} for frame {frame_record.frame_id}"
        )

    # Valid mask: finite depth > 0
    valid_mask = np.isfinite(depth) & (depth > 0.0)

    # Extract distance points as arrays
    dp_u = np.array([dp.u for dp in frame_record.distance_points], dtype=np.float64)
    dp_v = np.array([dp.v for dp in frame_record.distance_points], dtype=np.float64)
    dp_dist = np.array([dp.distance_m for dp in frame_record.distance_points], dtype=np.float64)

    # Check distance points are within image bounds
    H, W = depth.shape
    for i, (u, v) in enumerate(zip(dp_u, dp_v)):
        if not (0 <= u < W and 0 <= v < H):
            raise ValueError(
                f"Frame {frame_record.frame_id}: distance_point {i} "
                f"at ({u:.1f}, {v:.1f}) outside image bounds ({W}x{H})"
            )

    return {
        "rgb_path": str(root / frame_record.rgb_path),
        "depth": depth,
        "mask": mask,
        "intrinsics": frame_record.intrinsics,
        "mount": frame_record.mount,
        "tags": frame_record.tags,
        "distance_points": {
            "u": dp_u,
            "v": dp_v,
            "distance_m": dp_dist,
        },
        "valid_mask": valid_mask,
    }


def load_route(dataset_root: Union[str, Path], route_dir: Path) -> list[dict[str, Any]]:
    """Load all frames in a route directory.

    Expects a route_manifest.json listing frame records.

    Args:
        dataset_root: Root directory of the dataset.
        route_dir: Directory containing route manifest and frame files.

    Returns:
        List of frame data dicts (as returned by load_frame).
    """
    manifest_path = route_dir / "route_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Route manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []

    for frame_data in manifest["frames"]:
        record = FrameRecord.from_dict(frame_data)
        frames.append(load_frame(dataset_root, record))

    return frames


def iterate_dataset(
    dataset_root: Union[str, Path],
    route_ids: Optional[list[str]] = None,
) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """Iterate over all frames in a dataset.

    Args:
        dataset_root: Root directory containing route subdirectories.
        route_ids: Optional list of route IDs to include. If None, include all.

    Yields:
        Tuple of (route_id, frame_data_dict).
    """
    root = Path(dataset_root)

    if route_ids is None:
        # Find all subdirectories with route_manifest.json
        route_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "route_manifest.json").exists()]
    else:
        route_dirs = [root / rid for rid in route_ids if (root / rid / "route_manifest.json").exists()]

    for route_dir in sorted(route_dirs):
        route_id = route_dir.name
        try:
            frames = load_route(dataset_root, route_dir)
            for frame_data in frames:
                yield route_id, frame_data
        except Exception as e:
            raise RuntimeError(f"Failed to load route {route_id}: {e}") from e