"""Model fetch helper for Visiona-Walk.

Default mode is ``--dry-run`` and makes NO network calls. Downloads happen
only with ``--yes``. ``--check`` asks the Hugging Face API whether each model
id resolves (network). Agents must not run ``--yes`` or ``--check``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS: list[dict[str, Any]] = [
    {
        "id": "nvidia/segformer-b0-finetuned-ade-512-512",
        "kind": "hf",
        "size_mb": 15.0,
        "target": "weights/segformer-b0-ade",
        "purpose": "SegFormer-B0 ADE20K segmentation",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "kind": "hf",
        "size_mb": 99.0,
        "target": "weights/depth-anything-v2-metric-indoor-small",
        "purpose": "Depth Anything V2 metric indoor depth",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "kind": "hf",
        "size_mb": 99.0,
        "target": "weights/depth-anything-v2-metric-outdoor-small",
        "purpose": "Depth Anything V2 metric outdoor depth",
    },
    {
        "id": "yolo11n.pt",
        "kind": "ultralytics",
        "size_mb": 5.6,
        "target": "yolo11n.pt",
        "purpose": "Ultralytics YOLO11-nano detector",
    },
    {
        "id": "mobile_sam.pt (MobileSAM ViT-T)",
        "kind": "manual",
        "size_mb": 40.0,
        "target": "weights/mobile-sam/mobile_sam.pt",
        "purpose": "Promptable floor refinement (manual download)",
        "source": "https://github.com/ChaoningZhang/MobileSAM",
    },
]


def _print_table() -> None:
    print(f"{'Model ID':<62} {'Size':>9}  {'Target':<48} Purpose")
    print("-" * 150)
    for model in MODELS:
        print(
            f"{model['id']:<62} {model['size_mb']:>7.1f} MB  "
            f"{model['target']:<48} {model['purpose']}"
        )


def _download_hf(model: dict[str, Any]) -> None:
    from huggingface_hub import snapshot_download  # lazy

    target = REPO_ROOT / model["target"]
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model["id"], local_dir=str(target))


def _download_ultralytics(model: dict[str, Any]) -> None:
    from ultralytics import YOLO  # lazy

    target = REPO_ROOT / model["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics downloads the checkpoint into the current working directory.
    YOLO(model["id"])
    downloaded = Path(model["id"])
    if downloaded.exists() and downloaded.resolve() != target.resolve():
        downloaded.replace(target)


def _check_hf_ids() -> int:
    import json
    from urllib.request import urlopen  # lazy network import

    failures = 0
    for model in MODELS:
        if model["kind"] != "hf":
            print(f"{model['id']}: non-HF asset, not a HF repo id (skipped)")
            continue
        url = f"https://huggingface.co/api/models/{model['id']}"
        try:
            with urlopen(url, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("id"):
                print(f"{model['id']}: resolves (id={payload['id']})")
            else:
                print(f"{model['id']}: unexpected response")
                failures += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{model['id']}: NOT FOUND ({exc})")
            failures += 1
    return 0 if failures == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List or download Visiona-Walk model weights")
    parser.add_argument("--dry-run", action="store_true", help="list only (default; no network)")
    parser.add_argument("--yes", action="store_true", help="actually download the weights")
    parser.add_argument("--check", action="store_true", help="verify model ids resolve (network)")
    args = parser.parse_args(argv)

    if args.yes and args.check:
        print("ERROR: --yes and --check are mutually exclusive")
        return 2

    print("Visiona-Walk model manifest")
    print(f"Repository root: {REPO_ROOT}")
    _print_table()

    if args.check:
        print("\nChecking model ids (network)...")
        return _check_hf_ids()

    if not args.yes:
        print("\nDRY RUN: no network calls and no downloads were made.")
        print("To download: python scripts/fetch_models.py --yes")
        return 0

    print("\nDownloading weights...")
    manual: list[dict[str, Any]] = []
    for model in MODELS:
        if model["kind"] == "manual":
            manual.append(model)
            continue
        print(f"- {model['id']} -> {model['target']}")
        if model["kind"] == "hf":
            _download_hf(model)
        else:
            _download_ultralytics(model)
    if manual:
        print("\nManual downloads still required (no network access performed):")
        for model in manual:
            print(
                f"- {model['id']}: get it from {model.get('source')} "
                f"and save it to {model['target']}"
            )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
