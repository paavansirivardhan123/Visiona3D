"""High-level steps used by the notebook, so the notebook itself stays short.

Every function takes the config dict from `make_config()`. Nothing here runs on import. Training only happens in `train()`.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch

from . import convert as C
from . import data as D
from . import pipeline as P
from .model import UnifiedYOLO26, export_onnx, load_unified

SEMANTIC_VAL = ("coco_stuff_val", "ade20k_val", "gdd_test", "stairs_kaist_valid", "hf_sidewalk_val")
SOURCE_WEIGHTS = {"coco_stuff_train": 0.30, "ade20k_train": 0.20, "gdd_train": 0.10, "stairs_kaist_train": 0.12, "hf_sidewalk": 0.28}
STUFF_URL = "https://calvin-vision.net/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"
ADE_URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


def make_config(**overrides) -> dict:
    """Paths and training settings. Override any key by keyword."""
    lib = Path(__file__).resolve().parents[1]                     # .../notebooks
    cfg = dict(
        lib=lib, data=lib / "Datasets", base_weights=lib.parent / "weights" / "yolo26n-seg.pt",
        ckpt=lib / "work" / "checkpoints", final=lib / "work" / "final",
        name="E1b_frozen_backbone_sidewalk", freeze=11,
        imgsz=320, batch=4, steps_per_epoch=500, epochs=20, time_budget_hours=9.0, patience=8,
        workers=0 if sys.platform == "win32" else 2, n_coco_images=20000, val_per_set=50, eval_limit=300,
    )
    cfg.update(overrides)
    for k in ("lib", "data", "base_weights", "ckpt", "final"):
        cfg[k] = Path(cfg[k])
    cfg["ckpt"].mkdir(parents=True, exist_ok=True)
    return cfg


# ----------------------------------------------------------------------------- data
def download_verified(url: str, dst: Path, size: int):
    """Resumable download that only appends when the server confirms the same offset, then checks exact size and zip CRC."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(200):
        have = dst.stat().st_size if dst.exists() else 0
        if have == size:
            break
        if have > size:
            dst.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.4.0", "Accept": "*/*", "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and (r.status != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {have}-")):
                    dst.unlink()
                    continue
                with open(dst, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
        except Exception:
            time.sleep(3)
    if dst.stat().st_size != size or zipfile.ZipFile(dst).testzip() is not None:
        raise RuntimeError(f"could not get a verified copy of {url}")
    return dst


def fetch_coco_and_ade(cfg: dict):
    """COCO labels, COCO-Stuff maps, ADE20K and the COCO training images (skips whatever is already on disk)."""
    d, coco = cfg["data"], cfg["data"] / "coco"
    dl = d / "_dl"
    if not (coco / "labels" / "train2017").exists():
        zipfile.ZipFile(download_verified(LABELS_URL, dl / "coco2017labels-segments.zip", 176694178)).extractall(d)
    stuff = d / "COCO-Stuff" / "stuffthingmaps"
    if len(list((stuff / "train2017").glob("*.png"))) < 117000 if (stuff / "train2017").exists() else True:
        z = download_verified(STUFF_URL, dl / "stuffthingmaps_trainval2017.zip", 659008564)
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith(".png") and m.startswith(("train2017/", "val2017/")) and not (stuff / m).exists():
                    zf.extract(m, stuff)
    if not (d / "ADEChallengeData2016" / "images" / "training").exists():
        zipfile.ZipFile(download_verified(ADE_URL, dl / "ADEChallengeData2016.zip", 967382037)).extractall(d)
    labels = cfg["lib"] / "analysis" / "cocostuff_labels.txt"
    if not (coco / "train_subset.txt").exists():
        keep, scanned = P.select_coco_subset(stuff / "train2017", C.cocostuff_lut(labels), cfg["n_coco_images"], 0.25, seed=0,
                                             label_dir=coco / "labels" / "train2017")
        print(f"selected {len(keep)} of {scanned} COCO train images")
        print("downloaded / failed:", P.download_images(keep, "http://images.cocodataset.org/train2017/{stem}.jpg",
                                                         coco / "images" / "train2017", workers=8))
        have = sorted(p.stem for p in (coco / "images" / "train2017").glob("*.jpg"))
        (coco / "train_subset.txt").write_text("\n".join(f"./images/train2017/{s}.jpg" for s in have) + "\n")
    val_list = "val_subset1000.txt" if (coco / "val_subset1000.txt").exists() else "val2017.txt"
    P.write_instance_yaml(coco / "coco_instance.yaml", coco, "train_subset.txt", val_list)


def fetch_sidewalk(cfg: dict):
    """Hugging Face sidewalk-semantic: gated + CC BY-NC. Needs YOUR login (`huggingface-cli login`) and accepted terms."""
    root = cfg["data"] / "hf_sidewalk_semantic"
    if (root / "images").exists():
        return
    try:
        from huggingface_hub import hf_hub_download
        for f in ("id2label.json", "data/train-00000-of-00001.parquet"):
            hf_hub_download("segments/sidewalk-semantic", f, repo_type="dataset", local_dir=str(root))
        print("sidewalk images exported:", P.export_hf_sidewalk(root / "data" / "train-00000-of-00001.parquet", root))
        shutil.rmtree(root / "data", ignore_errors=True)
    except Exception as e:
        print("sidewalk dataset NOT available (needs your Hugging Face login and accepted terms):", type(e).__name__)


def load_records(cfg: dict) -> dict:
    """Convert every available dataset to the 8-class labels (skips converted ones) and return {name: records}."""
    d = cfg["data"]
    todo = [n for n in ("gdd_train", "gdd_test", "stairs_kaist_train", "stairs_kaist_valid", "ade20k_train", "ade20k_val",
                        "coco_stuff_train", "coco_stuff_val", "hf_sidewalk") if not (d / "_converted" / n / "coverage.jsonl").exists()]
    if todo:
        P.convert_all(d, cfg["lib"] / "analysis" / "cocostuff_labels.txt", only=todo)
    rec = {p.parent.name: C.read_coverage(p) for p in sorted((d / "_converted").glob("*/coverage.jsonl"))}
    if "hf_sidewalk" in rec and "hf_sidewalk_val" not in rec:           # one split only: hold out the last 100 for validation
        rec["hf_sidewalk_val"] = rec["hf_sidewalk"][900:]
        rec["hf_sidewalk"] = rec["hf_sidewalk"][:900]
    print({k: len(v) for k, v in rec.items()})
    return rec


# ----------------------------------------------------------------------------- model and baseline
def save_baseline(cfg: dict) -> dict:
    """Score the pretrained yolo26n-seg on the COCO validation list and save it before any training."""
    from ultralytics import YOLO
    f = cfg["ckpt"] / "baseline_pretrained_yolo26n_seg.json"
    if not f.exists():
        r = YOLO(str(cfg["base_weights"])).val(data=str(cfg["data"] / "coco" / "coco_instance.yaml"), imgsz=640, batch=8,
                                               device="cpu", plots=False, verbose=False)
        f.write_text(json.dumps({"box_map50_95": float(r.box.map), "mask_map50_95": float(r.seg.map), "imgsz": 640,
                                 "note": "pretrained yolo26n-seg on the local COCO validation list, before any training"}, indent=1))
    return json.loads(f.read_text())


def show_model(cfg: dict):
    model = UnifiedYOLO26(str(cfg["base_weights"]))
    for name, freeze in (("E1 (backbone frozen)", 11), ("E2 (first 6 layers frozen)", 6), ("E3 (nothing frozen)", None)):
        model.set_freeze(freeze)
        rep = model.param_report()
        print(f"{name:28s} trainable {rep['params_trainable']:>9,} of {rep['params_total']:,}")


# ----------------------------------------------------------------------------- training
def train(cfg: dict, records: dict, name: str | None = None, freeze: int | None = None):
    """THE TRAINING RUN. Resumes from last_resume.pt when present. Checkpoints are written every epoch."""
    from .train import fit
    name = name or cfg["name"]
    freeze = cfg["freeze"] if freeze is None else freeze
    out = cfg["ckpt"] / name
    out.mkdir(parents=True, exist_ok=True)
    resume = out / "last_resume.pt"
    resume = resume if resume.exists() else None
    inst_loader, _ = D.make_instance_loader(str(cfg["data"] / "coco" / "coco_instance.yaml"), cfg["imgsz"], cfg["batch"],
                                            mode="train", workers=cfg["workers"])
    sources = {k: {"records": records[k], "weight": w} for k, w in SOURCE_WEIGHTS.items() if k in records}
    sem_loader = D.make_semantic_loader(sources, cfg["data"], cfg["batch"], imgsz=cfg["imgsz"], workers=cfg["workers"])
    device = torch.device("cpu")
    val_fn = P.make_semantic_val_fn({k: records[k] for k in SEMANTIC_VAL if k in records}, cfg["data"], device,
                                    per_set=cfg["val_per_set"], imgsz=416)
    fit(UnifiedYOLO26(str(cfg["base_weights"])), inst_loader, sem_loader, device, epochs=cfg["epochs"],
        steps_per_epoch=cfg["steps_per_epoch"], out_dir=out, freeze=freeze, patience=cfg["patience"], val_fn=val_fn,
        save_every=5, resume=resume, time_budget_s=cfg["time_budget_hours"] * 3600)
    print(name, "files:", sorted(x.name for x in out.iterdir()))


# ----------------------------------------------------------------------------- evaluation, speed, final files
def evaluate(cfg: dict, records: dict, name: str | None = None) -> dict:
    """Instance branch vs the pretrained baseline (Ultralytics), plus per-class semantic metrics. Saves evaluation.json."""
    from ultralytics import YOLO
    name = name or cfg["name"]
    out = cfg["ckpt"] / name
    yaml = str(cfg["data"] / "coco" / "coco_instance.yaml")
    res = {"name": name}
    for tag, w in (("baseline_pretrained", str(cfg["base_weights"])), (name + "_instance_branch", str(out / "best_instance.pt"))):
        r = YOLO(w).val(data=yaml, imgsz=640, batch=8, device="cpu", plots=False, verbose=False)
        res[tag] = {"box_map50_95": float(r.box.map), "mask_map50_95": float(r.seg.map)}
    res["instance_mask_map_change_vs_baseline"] = res[name + "_instance_branch"]["mask_map50_95"] - res["baseline_pretrained"]["mask_map50_95"]
    device = torch.device("cpu")
    fn = P.unified_predict_fn(load_unified(out / "best_resume.pt", cfg["base_weights"], device), device, 416)
    res["semantic"] = {}
    for k in SEMANTIC_VAL:
        if k in records:
            s = P.evaluate_semantic_records(fn, records[k], cfg["data"], limit=cfg["eval_limit"])
            res["semantic"][k] = s
            print(f"--- {k}: mIoU {s['miou']} | false-walkable (of predicted walkable): {s['false_walkable_of_predicted_walkable']}")
            for c, v in s["per_class"].items():
                if v["images_with_class"]:
                    print(f"   {c:14s} IoU {v['iou']} | pixel miss {v['pixel_miss_rate']} | image miss {v['image_miss_rate']}")
    (out / "evaluation.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("instance:", {k: v for k, v in res.items() if k.startswith(("baseline", name))})
    print("instance mask mAP50-95 change vs pretrained baseline:", round(res["instance_mask_map_change_vs_baseline"], 4))
    return res


def speed(cfg: dict, name: str | None = None) -> dict:
    """Model-forward timing only (no pre/post-processing, no depth, no lifting): NOT the application's FPS."""
    from ultralytics import YOLO
    name = name or cfg["name"]
    model = load_unified(cfg["ckpt"] / name / "best_resume.pt", cfg["base_weights"], "cpu")
    base = YOLO(str(cfg["base_weights"])).model.cpu().eval()
    res = {}
    for s in (320, 416, 480):
        x = torch.zeros(1, 3, s, s)
        with torch.no_grad():
            res[f"baseline_yolo26n_seg_{s}"] = P.time_callable(lambda: base(x), repeat=20)
            res[f"unified_{s}"] = P.time_callable(lambda: model(x), repeat=20)
    res["WARNING"] = "model forward only on this computer; not the full pipeline"
    (cfg["ckpt"] / name / "speed_cpu_forward_only.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:30s} median {v['median_ms']:.0f} ms")
    return res


def write_final(cfg: dict, name: str | None = None):
    name = name or cfg["name"]
    src, dst = cfg["ckpt"] / name, cfg["final"]
    dst.mkdir(parents=True, exist_ok=True)
    for a, b in (("best_resume.pt", "visiona_unified_best_resume.pt"), ("best_instance.pt", "visiona_unified_best_instance.pt"),
                 ("best_semantic_head.pt", "visiona_semantic_head.pt")):
        shutil.copy(src / a, dst / b)
    try:
        export_onnx(load_unified(src / "best_resume.pt", cfg["base_weights"], "cpu"), 416, dst / "visiona_unified_416.onnx")
    except Exception as e:
        print("ONNX export failed (the .pt files are still saved):", type(e).__name__, e)
    print("final files:", sorted(f.name for f in dst.iterdir()))
