"""Visiona3D Dataset Package.

Provides data structures, loading utilities, splitting strategies, and synthetic
dataset generation for navigation aid evaluation.
"""

from dataset.schema import (
    FrameRecord,
    DistancePoint,
    MountInfo,
    Tags,
    validate,
)
from dataset.loader import load_frame, load_route, iterate_dataset
from dataset.splits import split_routes, leave_one_route_out, assert_no_leakage
from dataset.synth_dataset import build_synthetic_dataset

__all__ = [
    # Schema
    "FrameRecord",
    "DistancePoint",
    "MountInfo",
    "Tags",
    "validate",
    # Loader
    "load_frame",
    "load_route",
    "iterate_dataset",
    # Splits
    "split_routes",
    "leave_one_route_out",
    "assert_no_leakage",
    # Synthetic
    "build_synthetic_dataset",
]