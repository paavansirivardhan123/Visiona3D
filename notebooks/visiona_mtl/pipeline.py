"""Reusable steps shared by the notebook: subset selection, dataset layouts, evaluation loops, speed benchmark."""
from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from . import convert as C
from . import taxonomy as T
from .evaluate import SemanticMeter


# ------------------------------------------------------------------------------------------------
# convert everything that is on disk (missing datasets are skipped, never faked)
# ------------------------------------------------------------------------------------------------
def convert_all(data_root: Path, cocostuff_labels: Path, only=None, limit: int | None = None, verbose: bool = True):
    """Convert each available dataset under `data_root` to semantic PNGs + coverage.jsonl in data_root/_converted/<name>.

    Returns {name: {"records": [...], "pixel_counts": ndarray, "note": str}}. Datasets whose folders are missing are
    reported as skipped. Nothing is downloaded here.
    """
    data_root = Path(data_root)
    out = data_root / "_converted"
    results, skipped = {}, {}

    def run(name, available, fn, note=""):
        if only and name not in only:
            return
        if not available:
            skipped[name] = "folder not found"
            return
        t0 = time.time()
        recs, counts = fn()
        C.write_coverage(out / name / "coverage.jsonl", recs)
        results[name] = {"records": recs, "pixel_counts": counts, "note": note}
        if verbose:
            print(f"converted {name:22s} {len(recs):6d} images in {time.time() - t0:6.1f}s")

    for split, folder in (("gdd_train", "GDD (glass)_train/train"), ("gdd_test", "GDD (glass)_test/test")):
        root = data_root / folder
        run(split, root.exists(), lambda r=root, s=split: C.convert_gdd(s, r, out / s, data_root, limit))
    stairs = data_root / "stairs segmentation.v1i.yolo26"
    for split in ("train", "valid"):
        name = f"stairs_kaist_{split}"
        run(name, (stairs / split / "labels").exists(),
            lambda s=split: C.convert_yolo_polygons("stairs_kaist", stairs / s / "images", stairs / s / "labels",
                                                    out / f"stairs_kaist_{s}", {0: "staircase"}, data_root, limit)[:2])
    ade = data_root / "ADEChallengeData2016"
    for split, folder in (("ade20k_train", "training"), ("ade20k_val", "validation")):
        run(split, (ade / "images" / folder).exists(),
            lambda f=folder, s=split: C.convert_png_masks("ade20k", ade / "images" / f, ade / "annotations" / f,
                                                          out / s, C.ade20k_lut(), data_root, limit=limit))
    stuff = data_root / "COCO-Stuff" / "stuffthingmaps"
    for split, folder in (("coco_stuff_train", "train2017"), ("coco_stuff_val", "val2017")):
        imgs = data_root / "coco" / "images" / folder
        run(split, (stuff / folder).exists() and imgs.exists(),
            lambda f=folder, s=split, i=imgs: C.convert_png_masks(
                "coco_stuff", i, stuff / f, out / s, C.cocostuff_lut(cocostuff_labels), data_root, limit=limit,
                objects_labeled=True))
    hf = data_root / "hf_sidewalk_semantic"
    if (hf / "images").exists():
        run("hf_sidewalk", True, lambda: C.convert_png_masks("hf_sidewalk", hf / "images", hf / "labels", out / "hf_sidewalk",
                                                           C.hf_sidewalk_lut(), data_root, limit=limit, objects_labeled=False))
    elif not only or "hf_sidewalk" in only:
        skipped["hf_sidewalk"] = "gated dataset: download it after accepting its terms and logging in"
    if verbose and skipped:
        print("skipped:", skipped)
    return results, skipped


# ------------------------------------------------------------------------------------------------
# COCO subset (Colab): choose images that are rich in navigation-relevant stuff, then fetch only those
# ------------------------------------------------------------------------------------------------
def env_share(stuff_png: Path, lut: np.ndarray) -> float:
    """Share of pixels that map to one of the 8 environment classes."""
    label = cv2.imread(str(stuff_png), cv2.IMREAD_UNCHANGED)
    if label is None:
        return 0.0
    sem = lut[label]
    return float((sem < T.N_SEM).mean())


def select_coco_subset(stuff_dir: Path, lut: np.ndarray, n: int, min_env_share: float = 0.25, seed: int = 0,
                       label_dir: Path | None = None, limit_scan: int | None = None):
    """Pick up to `n` image stems whose COCO-Stuff map has >= min_env_share environment pixels (and, if label_dir is
    given, that also have a COCO instance label file, so objects are labeled). Reproducible via `seed`."""
    stems = sorted(p.stem for p in Path(stuff_dir).glob("*.png"))
    if limit_scan:
        stems = stems[:limit_scan]
    keep = []
    for s in stems:
        if label_dir is not None and not (Path(label_dir) / f"{s}.txt").exists():
            continue
        if env_share(Path(stuff_dir) / f"{s}.png", lut) >= min_env_share:
            keep.append(s)
    rng = np.random.default_rng(seed)
    if len(keep) > n:
        keep = sorted(rng.choice(keep, size=n, replace=False).tolist())
    return keep, len(stems)


def download_images(stems, url_template: str, out_dir: Path, workers: int = 16, retries: int = 3):
    """Fetch only the listed COCO images (`url_template` contains {stem}). Skips files that already exist."""
    import urllib.request
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def valid(path: Path) -> bool:
        """Full decode with PIL: OpenCV silently accepts truncated JPEGs, Ultralytics (rightly) does not."""
        try:
            from PIL import Image
            with Image.open(path) as im:
                im.load()
            return True
        except Exception:
            return False

    def one(stem):
        dst = out_dir / f"{stem}.jpg"
        if dst.exists() and dst.stat().st_size > 0 and valid(dst):
            return True
        for _ in range(retries):
            try:
                req = urllib.request.Request(url_template.format(stem=stem), headers={"User-Agent": "curl/8.4.0", "Accept": "*/*"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    dst.write_bytes(r.read())
                if valid(dst):
                    return True
                dst.unlink(missing_ok=True)
            except Exception:
                time.sleep(1.0)
        return False

    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(one, stems))
    return sum(ok), len(ok) - sum(ok)


def write_instance_yaml(path: Path, coco_root: Path, train_list: str, val_list: str):
    """Ultralytics data YAML for the instance stream. Deliberately has no `download:` block."""
    lines = [f"path: {Path(coco_root).as_posix()}", f"train: {train_list}", f"val: {val_list}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(T.COCO80)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------
# native Ultralytics semantic task: dataset layout with our label_mapping
# ------------------------------------------------------------------------------------------------
def native_lut(full_label: bool, other_as_class: bool) -> np.ndarray:
    """Value map for the SOFTMAX semantic task. 254 ('labeled other') becomes class 8 'other' only for fully labeled
    sources and only when other_as_class is on; for single-class sources (glass-only, stairs-only) 254 is unknown
    (a softmax cannot say 'not glass'), so it becomes 255."""
    lut = np.arange(256, dtype=np.uint8)
    lut[T.IGNORE] = T.IGNORE
    lut[T.OTHER_LABELED] = T.N_SEM if (full_label and other_as_class) else T.IGNORE
    return lut


def export_native_semantic(records_by_split: dict, out_root: Path, other_as_class: bool = False, link: bool = True):
    """Create <out_root>/images/<split>/ and <out_root>/annotations/<split>/ for Ultralytics' semantic task.

    records_by_split: {"train": [coverage records], "val": [...]}. Images are symlinked (copied if symlinks fail);
    masks are rewritten with native_lut so 254 never reaches a softmax unless 'other' is a real class.
    """
    out_root = Path(out_root)
    counts = {}
    for split, recs in records_by_split.items():
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "annotations" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for r in recs:
            full = len(r["supervised"]) == T.N_SEM
            stem = f"{r['source']}__{Path(r['image']).stem}"
            src_img = Path(r["_base"]) / r["image"]
            dst_img = out_root / "images" / split / f"{stem}{src_img.suffix.lower()}"
            if not dst_img.exists():
                try:
                    if link:
                        os.symlink(src_img, dst_img)
                    else:
                        shutil.copy2(src_img, dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)
            mask_file = Path(r["_base"]) / r["sem_mask"]
            sem = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            if sem is None:
                raise FileNotFoundError(f"semantic mask not readable: {mask_file}")
            cv2.imwrite(str(out_root / "annotations" / split / f"{stem}.png"), native_lut(full, other_as_class)[sem])
            n += 1
        counts[split] = n
    return counts


def native_semantic_yaml(path: Path, root: Path, other_as_class: bool = False):
    names = list(T.SEM_CLASSES) + (["other"] if other_as_class else [])
    lines = [f"path: {Path(root).as_posix()}", "train: images/train", "val: images/val", "masks_dir: annotations", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------
# evaluation loops
# ------------------------------------------------------------------------------------------------
def evaluate_semantic_records(predict_fn, records, base: Path, limit: int | None = None, imgsz: int = 640):
    """predict_fn(image_rgb_uint8) -> class map (H,W) uint8 at the image's own size (0..7 or 255).

    Loops over coverage records, resizes the ground truth to the prediction size when needed (nearest), accumulates a
    SemanticMeter and returns its summary dict plus the per-image runtime.
    """
    meter = SemanticMeter()
    times = []
    for n, r in enumerate(records):
        if limit is not None and n >= limit:
            break
        img = cv2.imread(str(Path(base) / r["image"]))
        gt = cv2.imread(str(Path(base) / r["sem_mask"]), cv2.IMREAD_UNCHANGED)
        if img is None or gt is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        pred = predict_fn(rgb)
        times.append((time.perf_counter() - t0) * 1000.0)
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        meter.update(pred, gt, r["supervised"])
    out = meter.summary()
    out["predict_ms_median"] = float(np.median(times)) if times else None
    return out


def segformer_predict_fn(weights_dir: Path, input_size: int = 256):
    """The EXISTING system's segmentation (SegFormer-B0) mapped to our ids, for the three-way comparison.

    Existing labels: 1 floor, 2 wall, 3 obstacle, 4 other (+ a sky mask). Mapped: floor->0, wall->3, sky->7, everything
    else -> 255 (the old model has no walkable_path/road/glass/stairs/door classes, so only floor, wall and sky are
    comparable; false-walkable is comparable only through the floor class).
    """
    from perception.seg_backend import SegFormerBackend
    backend = SegFormerBackend(weights_dir=weights_dir, input_size=input_size)

    def fn(rgb):
        out = backend.predict(rgb)
        pred = np.full(out.labels.shape, T.IGNORE, np.uint8)
        pred[out.labels == 1] = T.SEM_ID["floor"]
        pred[out.labels == 2] = T.SEM_ID["wall"]
        pred[out.sky_mask] = T.SEM_ID["sky"]
        return pred
    return fn


# ------------------------------------------------------------------------------------------------
# speed benchmark helper (CPU latency of any callable on one image)
# ------------------------------------------------------------------------------------------------
def time_callable(fn, repeat: int = 15, warmup: int = 3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "fps_median": float(1000.0 / np.median(a)), "n": int(a.size)}


# ------------------------------------------------------------------------------------------------
# remote zip: read a zip's directory over HTTP and fetch ONLY the members you need (byte ranges)
# ------------------------------------------------------------------------------------------------
class HTTPRangeFile:
    """Minimal read-only, seekable file over HTTP Range requests (enough for zipfile to read a central directory)."""

    def __init__(self, url: str, user_agent: str = "curl/8.4.0"):
        import urllib.request
        self.url, self.ua = url, user_agent
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": user_agent, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def _range(self, start: int, end_inclusive: int) -> bytes:
        import urllib.request
        req = urllib.request.Request(self.url, headers={"User-Agent": self.ua, "Accept": "*/*",
                                                        "Range": f"bytes={start}-{end_inclusive}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) != end_inclusive - start + 1:
            raise IOError(f"short range read: wanted {end_inclusive - start + 1}, got {len(data)}")
        return data

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = self._range(self.pos, self.pos + n - 1)
        self.pos += n
        return data


def remote_zip_infos(url: str):
    """Open a remote zip's central directory. Returns (list of ZipInfo, HTTPRangeFile)."""
    import zipfile
    f = HTTPRangeFile(url)
    return zipfile.ZipFile(f).infolist(), f


def fetch_zip_members(url: str, infos, out_root: Path, strip_prefix: str = "", workers: int = 8, retries: int = 3):
    """Download and decompress only `infos` (ZipInfo objects) from the remote zip into out_root (parallel range GETs).
    Verifies each member's CRC32 and size. Returns (ok_count, failed_names)."""
    import struct
    import zlib
    f = HTTPRangeFile(url)
    out_root = Path(out_root)

    def one(info):
        name = info.filename[len(strip_prefix):] if strip_prefix and info.filename.startswith(strip_prefix) else info.filename
        dst = out_root / name
        if dst.exists() and dst.stat().st_size == info.file_size:
            return None
        for _ in range(retries):
            try:
                head = f._range(info.header_offset, info.header_offset + 29)          # local file header is 30 bytes
                n_len, e_len = struct.unpack("<HH", head[26:30])
                start = info.header_offset + 30 + n_len + e_len
                raw = f._range(start, start + info.compress_size - 1) if info.compress_size else b""
                data = raw if info.compress_type == 0 else zlib.decompressobj(-15).decompress(raw)
                if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
                    raise IOError("size or CRC mismatch")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                return None
            except Exception:
                time.sleep(1.0)
        return info.filename

    with ThreadPoolExecutor(workers) as ex:
        failed = [x for x in ex.map(one, infos) if x]
    return len(list(infos)) - len(failed), failed


# ------------------------------------------------------------------------------------------------
# validation used DURING training (semantic side) and the unified prediction function
# ------------------------------------------------------------------------------------------------
def unified_predict_fn(model, device, imgsz: int = 416, threshold: float = 0.5):
    """Semantic prediction of the unified model on one RGB uint8 image, returned at the image's own size (0..7 or 255)."""
    import cv2
    from . import data as D, loss as L

    def fn(rgb):
        h, w = rgb.shape[:2]
        canvas, _ = D.letterbox_pair(rgb, None, imgsz, False, np.random.default_rng(0))
        x = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1)[None].float().div(255).to(device)
        with torch.no_grad():
            logits = model.forward_semantic(x)
        cm, _ = L.compose(torch.nn.functional.interpolate(logits, size=(imgsz, imgsz), mode="bilinear", align_corners=False), threshold)
        cm = cm[0].cpu().numpy()
        scale = imgsz / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        y0, x0 = (imgsz - nh) // 2, (imgsz - nw) // 2
        return cv2.resize(cm[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_NEAREST)
    return fn


def make_semantic_val_fn(val_sets: dict, base: Path, device, per_set: int = 100, imgsz: int = 416):
    """val_sets: {name: records}. Returns val_fn(model) -> {"score": mean of each set's mIoU, "per_set": {...}} using a fixed
    subset of `per_set` images per set so the score is comparable between epochs. Never touches training data."""
    subsets = {k: list(v)[:per_set] for k, v in val_sets.items() if len(v)}

    def val_fn(model):
        was_training = model.training
        model.eval()
        per = {}
        for name, recs in subsets.items():
            s = evaluate_semantic_records(unified_predict_fn(model, device, imgsz), recs, base, imgsz=imgsz)
            per[name] = s["miou"]
        model.train(was_training)
        vals = [v for v in per.values() if v is not None]
        return {"score": float(sum(vals) / len(vals)) if vals else 0.0, "per_set": per}
    return val_fn


# ------------------------------------------------------------------------------------------------
# Hugging Face sidewalk-semantic (gated; CC BY-NC): parquet -> images + raw label PNGs
# ------------------------------------------------------------------------------------------------
def export_hf_sidewalk(parquet: Path, out_root: Path):
    """Write the parquet's images to out_root/images/*.jpg and its label maps to out_root/labels/*.png (raw ids 0..34)."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    out_root = Path(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet)
    n = 0
    for batch in pf.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            stem = f"sidewalk_{n:05d}"
            img = Image.open(io.BytesIO(row["pixel_values"]["bytes"])).convert("RGB")
            lab = Image.open(io.BytesIO(row["label"]["bytes"]))
            img.save(out_root / "images" / f"{stem}.jpg", quality=95)
            lab.save(out_root / "labels" / f"{stem}.png")
            n += 1
    return n
