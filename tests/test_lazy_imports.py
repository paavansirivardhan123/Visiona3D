"""P9: importing perception modules must not import torch/transformers/ultralytics/cv2/mobile_sam."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "perception.interfaces",
    "perception.class_map",
    "perception.postprocess",
    "perception.fakes",
    "perception.seg_backend",
    "perception.depth_backend",
    "perception.detector_backend",
    "perception.pipeline",
    "perception.seg_fusion",
    "perception.fusion_config",
    "calibration.depth_interval_config",
]

CODE = """
import sys
for name in {modules!r}:
    __import__(name)
heavy = [name for name in ("torch", "transformers", "ultralytics", "cv2", "mobile_sam") if name in sys.modules]
print(",".join(heavy))
raise SystemExit(1 if heavy else 0)
""".format(modules=MODULES)


def test_p9_no_heavy_imports_in_fresh_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", CODE],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == ""
