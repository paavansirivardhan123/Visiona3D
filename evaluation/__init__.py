"""Visiona3D Evaluation Package.

Provides metrics, bootstrap confidence intervals, logging, and reporting for
navigation aid evaluation.
"""

from evaluation.metrics_seg import compute_segmentation_metrics
from evaluation.metrics_depth import compute_depth_metrics
from evaluation.metrics_interval import compute_interval_metrics
from evaluation.metrics_abstain import compute_abstention_metrics
from evaluation.bootstrap import bootstrap_routes
from evaluation.logging_jsonl import JSONLLogger
from evaluation.report import generate_report

__all__ = [
    "compute_segmentation_metrics",
    "compute_depth_metrics",
    "compute_interval_metrics",
    "compute_abstention_metrics",
    "bootstrap_routes",
    "JSONLLogger",
    "generate_report",
]