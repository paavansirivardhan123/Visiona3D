"""Report generation for Visiona3D evaluation.

Produces text and JSON results tables with bootstrap confidence intervals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import json
import datetime
import numpy as np


def generate_report(
    results: dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    include_bootstrap: bool = True,
) -> str:
    """Generate a text report from evaluation results.

    Args:
        results: Dict containing metrics and bootstrap results.
                 Expected structure:
                 {
                     "metrics": {metric_name: value or {per_bin, per_tag}},
                     "bootstrap": {metric_name: {mean, ci_lower, ci_upper}},
                     "metadata": {dataset info}
                 }
        output_path: Optional path to write report file.
        include_bootstrap: Whether to include bootstrap CI in output.

    Returns:
        Formatted text report string.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("VISIONA3D EVALUATION REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
    lines.append("")

    # Metadata
    if "metadata" in results:
        lines.append("METADATA:")
        for k, v in results["metadata"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    # Overall metrics
    if "metrics" in results:
        lines.append("OVERALL METRICS:")
        lines.append("-" * 100)
        metrics = results["metrics"]
        bootstrap = results.get("bootstrap", {}) if include_bootstrap else {}

        # Print depth metrics
        if "depth" in metrics:
            lines.append("  Depth Estimation:")
            dm = metrics["depth"]
            for k in ["AbsRel", "RMSE", "MAE", "Bias", "Delta125"]:
                if k in dm:
                    val = dm[k]
                    bs = bootstrap.get(f"depth.{k}", {})
                    ci_str = ""
                    if bs and "ci_lower" in bs and "ci_upper" in bs:
                        ci_str = f"  [95% CI: {bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]"
                    lines.append(f"    {k}: {val:.4f}{ci_str}")

            if "per_bin" in dm:
                lines.append("    Per Range Bin:")
                for bin_name, bin_metrics in dm["per_bin"].items():
                    line = f"      {bin_name}:"
                    for k in ["AbsRel", "RMSE", "MAE", "Bias", "Delta125"]:
                        if k in bin_metrics:
                            val = bin_metrics[k]
                            line += f" {k}={val:.4f}"
                    lines.append(line)

        # Print segmentation metrics
        if "segmentation" in metrics:
            lines.append("  Segmentation:")
            sm = metrics["segmentation"]
            if "mean_iou" in sm:
                val = sm["mean_iou"]
                bs = bootstrap.get("segmentation.mean_iou", {})
                ci_str = f"  [95% CI: {bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]" if bs else ""
                lines.append(f"    Mean IoU: {val:.4f}{ci_str}")

            if "per_class_iou" in sm:
                lines.append("    Per-Class IoU:")
                for cls, iou in sm["per_class_iou"].items():
                    lines.append(f"      {cls}: {iou:.4f}")

            if "false_walkable_rate" in sm:
                val = sm["false_walkable_rate"]
                bs = bootstrap.get("segmentation.false_walkable_rate", {})
                ci_str = f"  [95% CI: {bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]" if bs else ""
                lines.append(f"    False-Walkable Rate: {val:.4f}{ci_str}")

            if "false_walkable_rate_by_bin" in sm:
                lines.append("    False-Walkable Rate by Bin:")
                for bin_name, rate in sm["false_walkable_rate_by_bin"].items():
                    lines.append(f"      {bin_name}: {rate:.4f}")

        # Print interval metrics
        if "interval" in metrics:
            lines.append("  Uncertainty/Interval:")
            im = metrics["interval"]
            for k in ["coverage", "mean_width", "mean_rel_width", "coverage_gap", "interval_score"]:
                if k in im:
                    val = im[k]
                    bs = bootstrap.get(f"interval.{k}", {})
                    ci_str = f"  [95% CI: {bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]" if bs else ""
                    lines.append(f"    {k}: {val:.4f}{ci_str}")

            if "per_bin" in im:
                lines.append("    Per Range Bin:")
                for bin_name, bin_metrics in im["per_bin"].items():
                    line = f"      {bin_name}:"
                    for k in ["coverage", "mean_width", "interval_score"]:
                        if k in bin_metrics:
                            val = bin_metrics[k]
                            line += f" {k}={val:.4f}"
                    lines.append(line)

        # Print abstention metrics
        if "abstention" in metrics:
            lines.append("  Abstention:")
            am = metrics["abstention"]
            if "aurc" in am:
                val = am["aurc"]
                bs = bootstrap.get("abstention.aurc", {})
                ci_str = f"  [95% CI: {bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]" if bs else ""
                lines.append(f"    AURC: {val:.4f}{ci_str}")
                if "random_aurc_mean" in am:
                    lines.append(f"    Random AURC (mean): {am['random_aurc_mean']:.4f}")

    lines.append("")
    lines.append("=" * 100)

    report_text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report_text, encoding="utf-8")

    return report_text


def generate_json_report(
    results: dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Generate a JSON-serializable report.

    Args:
        results: Same as generate_report.
        output_path: Optional path to write JSON file.

    Returns:
        JSON-serializable dict with results and metadata.
    """
    # Ensure all numpy types are converted to Python types
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(v) for v in obj)
        else:
            return obj

    json_results = convert(results)

    # Add generation timestamp
    json_results["_generated_at"] = datetime.datetime.now().isoformat()

    if output_path:
        Path(output_path).write_text(
            json.dumps(json_results, indent=2), encoding="utf-8"
        )

    return json_results