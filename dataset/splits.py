"""Dataset splitting utilities.

Provides route-level splitting with stratification by tags and leakage prevention.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional
import json
import warnings
import numpy as np


def _load_route_tags(dataset_root: Path, route_id: str) -> dict[str, str]:
    """Load tags from route manifest."""
    manifest_path = dataset_root / route_id / "route_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("tags", {})


def _get_all_route_ids(dataset_root: Path) -> list[str]:
    """Get all route IDs that have a manifest."""
    return sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and (d / "route_manifest.json").exists()
    ])


def split_routes(
    routes: list[str],
    fractions: tuple[float, float, float],
    seed: int,
    stratify_by: Optional[str] = None,
    dataset_root: Optional[Path] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split route IDs into train/val/test sets.

    Args:
        routes: List of route IDs to split.
        fractions: Tuple of (train_frac, val_frac, test_frac). Must sum to 1.0.
        seed: Random seed for reproducibility.
        stratify_by: Optional tag key to stratify by ("env", "light", "hazard", "crowd").
                     Requires dataset_root to read manifests.
        dataset_root: Dataset root directory (required if stratify_by is set).

    Returns:
        Tuple of (train_routes, val_routes, test_routes) lists.

    Raises:
        ValueError: If fractions don't sum to 1.0, or stratify_by requires dataset_root.
    """
    train_frac, val_frac, test_frac = fractions
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError(f"Fractions must sum to 1.0, got {fractions}")

    rng = np.random.default_rng(seed)
    routes_arr = np.array(routes)

    if stratify_by is None:
        # Simple random shuffle
        perm = rng.permutation(len(routes_arr))
        routes_shuffled = routes_arr[perm]
    else:
        if dataset_root is None:
            raise ValueError("dataset_root required when stratify_by is set")

        # Group routes by tag value
        tag_groups: dict[str, list[str]] = {}
        for rid in routes:
            tags = _load_route_tags(dataset_root, rid)
            tag_val = tags.get(stratify_by, "unknown")
            tag_groups.setdefault(tag_val, []).append(rid)

        # Split each tag group on its own so every split sees every tag value
        # that has enough routes. Groups of fewer than 3 routes cannot be
        # spread over three splits and go to train (with a warning).
        train_out: list[str] = []
        val_out: list[str] = []
        test_out: list[str] = []
        for tag_val in sorted(tag_groups.keys()):
            group = np.array(tag_groups[tag_val])
            group = group[rng.permutation(len(group))]
            g = len(group)
            if g < 3:
                warnings.warn(
                    f"Stratification by '{stratify_by}': tag value '{tag_val}' has only "
                    f"{g} route(s); all placed in train.",
                    UserWarning,
                )
                train_out.extend(group.tolist())
                continue
            g_test = max(1, int(round(g * test_frac))) if test_frac > 0 else 0
            g_val = max(1, int(round(g * val_frac))) if val_frac > 0 else 0
            while g - g_test - g_val < 1:  # always keep at least one train route
                if g_val >= g_test and g_val > 0:
                    g_val -= 1
                elif g_test > 0:
                    g_test -= 1
                else:
                    break
            g_train = g - g_val - g_test
            train_out.extend(group[:g_train].tolist())
            val_out.extend(group[g_train:g_train + g_val].tolist())
            test_out.extend(group[g_train + g_val:].tolist())
        return train_out, val_out, test_out

    n = len(routes_shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_routes = routes_shuffled[:n_train].tolist()
    val_routes = routes_shuffled[n_train:n_train + n_val].tolist()
    test_routes = routes_shuffled[n_train + n_val:].tolist()

    return train_routes, val_routes, test_routes


def leave_one_route_out(routes: list[str]) -> list[tuple[list[str], list[str]]]:
    """Generate leave-one-route-out cross-validation folds.

    Args:
        routes: List of route IDs.

    Returns:
        List of (train_routes, test_routes) tuples, one per route.
        test_routes is a list with a single route ID.
    """
    folds = []
    for i, test_route in enumerate(routes):
        train_routes = routes[:i] + routes[i+1:]
        folds.append((train_routes, [test_route]))
    return folds


def assert_no_leakage(
    splits: dict[str, list[str]],
    dataset_root: Path,
) -> None:
    """Assert that no route appears in more than one split.

    Args:
        splits: Dict mapping split name to list of route IDs.
        dataset_root: Dataset root (used to verify route existence).

    Raises:
        AssertionError: If any route appears in multiple splits or doesn't exist.
    """
    seen = set()
    all_routes = _get_all_route_ids(dataset_root)
    all_routes_set = set(all_routes)

    for split_name, route_list in splits.items():
        for rid in route_list:
            if rid in seen:
                raise AssertionError(f"Route '{rid}' appears in multiple splits (seen in '{split_name}')")
            if rid not in all_routes_set:
                raise AssertionError(f"Route '{rid}' in split '{split_name}' does not exist in dataset")
            seen.add(rid)

    # Optionally warn about unused routes
    unused = all_routes_set - seen
    if unused:
        warnings.warn(f"Routes not assigned to any split: {sorted(unused)}", UserWarning)