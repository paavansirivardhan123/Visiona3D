"""Unified multi-task model: shared YOLO26 backbone + neck, instance head (80 COCO classes) + semantic head (8 classes).

    image -> backbone (layers 0..10) -> neck (layers 11..22) -> P3/8, P4/16, P5/32 features
                                                              |-> Segment26 head: boxes, class scores, mask coefficients, protos
                                                              |-> SemanticHead(P3, P4): per-pixel logits for the 8 classes

The instance branch is an UNMODIFIED Ultralytics SegmentationModel (so its weights stay a standard YOLO26-seg
checkpoint). Only the semantic head is new. Nothing in this file trains anything.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import Conv

from . import taxonomy as T


class SemanticHead(nn.Module):
    """Fuse P3 (stride 8) and P4 (stride 16), then classify each pixel with an independent sigmoid per class."""

    def __init__(self, ch=(64, 128), n_classes: int = T.N_SEM, mid: int | None = None, aux: bool = True):
        super().__init__()
        c = mid or ch[0]
        self.p3 = Conv(ch[0], c, 3)
        self.p4 = Conv(ch[1], c, 1)
        self.fuse = Conv(c, c, 3)
        self.classifier = nn.Conv2d(c, n_classes, 1)
        self.aux = nn.Sequential(Conv(ch[1], c, 3), nn.Conv2d(c, n_classes, 1)) if aux else None
        self.n_classes = n_classes

    def forward(self, feats):
        p3 = self.p3(feats[0])
        p4 = F.interpolate(self.p4(feats[1]), size=p3.shape[-2:], mode="nearest")
        logits = self.classifier(self.fuse(p3 + p4))               # (B, C, H/8, W/8)
        if self.training and self.aux is not None:
            return logits, self.aux(feats[1])
        return logits


class UnifiedYOLO26(nn.Module):
    """Shared backbone/neck with an instance head and a semantic head."""

    BACKBONE_LAYERS = 11                                            # layers 0..10 (YOLO26 backbone)

    def __init__(self, weights: str = "yolo26n-seg.pt", n_sem: int = T.N_SEM, sem_aux: bool = True):
        super().__init__()
        yolo = YOLO(weights)                                         # .pt = pretrained, .yaml = random init
        self.det = yolo.model                                        # SegmentationModel (instance head, 80 classes)
        self.names = dict(yolo.model.names) if hasattr(yolo.model, "names") else {}
        head = self.det.model[-1]
        self.feature_layers = [int(i) for i in head.f]               # [16, 19, 22]
        # channel widths of P3/P4/P5, found with one tiny forward pass
        was_training = self.det.training
        self.det.eval()
        with torch.no_grad():
            feats = self.features(torch.zeros(1, 3, 64, 64))
        self.det.train(was_training)
        self.feature_channels = [int(f.shape[1]) for f in feats]
        self.sem = SemanticHead(ch=tuple(self.feature_channels[:2]), n_classes=n_sem, aux=sem_aux)

    # -- shared trunk ---------------------------------------------------------------------------------------------
    def features(self, x: torch.Tensor):
        """Run layers 0..22 (backbone + neck) and return [P3, P4, P5]. Mirrors BaseModel._predict_once."""
        y = []
        for m in self.det.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.det.save else None)
        return [y[j] for j in self.feature_layers]

    # -- heads ----------------------------------------------------------------------------------------------------
    def forward_semantic(self, x: torch.Tensor):
        """Semantic-only pass (skips the instance head): used for batches that only carry environment labels."""
        return self.sem(self.features(x))

    def forward(self, x: torch.Tensor):
        """Both heads on the same trunk. Returns {'det': <Ultralytics output>, 'sem': logits}."""
        det_out = self.det(x)                                        # full pass through the unmodified instance model
        preds = det_out if isinstance(det_out, dict) else det_out[1]
        feats = preds["one2many"]["feats"] if "one2many" in preds else preds["feats"]
        return {"det": det_out, "sem": self.sem(feats)}

    # -- freezing -------------------------------------------------------------------------------------------------
    def set_freeze(self, n_layers: int | list[int] | None):
        """Freeze the first `n_layers` layers of the shared trunk (11 = whole backbone), or an explicit index list.

        The semantic head and the instance head always stay trainable.
        """
        for p in self.det.parameters():
            p.requires_grad = True
        if n_layers is None:
            return []
        idx = list(range(n_layers)) if isinstance(n_layers, int) else list(n_layers)
        for i in idx:
            for p in self.det.model[i].parameters():
                p.requires_grad = False
        return idx

    def frozen_to_eval(self):
        """Put every module whose parameters are ALL frozen into eval mode. Without this, `model.train()` lets BatchNorm
        running statistics of a "frozen" layer keep drifting, so the layer is not really frozen. Call after `.train()`."""
        n = 0
        for m in self.det.modules():
            ps = list(m.parameters(recurse=False))
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and ps and not any(p.requires_grad for p in ps):
                m.eval()
                n += 1
        return n

    def param_report(self):
        def count(params):
            return sum(p.numel() for p in params)
        det_layers = list(self.det.model)
        return {
            "layers_total": len(det_layers),
            "params_backbone_0_10": count(p for m in det_layers[:11] for p in m.parameters()),
            "params_neck_11_22": count(p for m in det_layers[11:-1] for p in m.parameters()),
            "params_instance_head_23": count(det_layers[-1].parameters()),
            "params_semantic_head_new": count(self.sem.parameters()),
            "params_total": count(self.parameters()),
            "params_trainable": count(p for p in self.parameters() if p.requires_grad),
            "feature_channels_P3_P4_P5": self.feature_channels,
        }

    def state_for_instance_checkpoint(self):
        """The instance branch only, as a plain Ultralytics-loadable model (copy)."""
        return copy.deepcopy(self.det)



def load_unified(resume_path, base_weights, device="cpu", use_ema=True):
    """Rebuild the unified model from a training checkpoint (`best_resume.pt` / `last_resume.pt`).

    `base_weights` is the original yolo26n-seg.pt (it only supplies the architecture; the trained values come from the checkpoint).
    """
    ck = torch.load(resume_path, map_location="cpu", weights_only=False)
    model = UnifiedYOLO26(base_weights)
    state = ck["ema"] if use_ema and ck.get("ema") is not None else ck["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


class UnifiedExportWrapper(nn.Module):
    """Traceable wrapper: (instance predictions, prototypes, semantic probabilities) from one trunk pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        out = self.model(x)
        preds, proto = out["det"][0]
        return preds, proto, out["sem"].sigmoid()


def export_onnx(model, imgsz, path):
    """Write the unified model as one ONNX file with three outputs: instances, prototypes, semantic_probs. Needs `onnx`."""
    wrapper = UnifiedExportWrapper(copy.deepcopy(model).cpu())
    torch.onnx.export(wrapper, torch.zeros(1, 3, imgsz, imgsz), str(path), opset_version=17,
                      input_names=["images"], output_names=["instances", "prototypes", "semantic_probs"], dynamo=False)
    return path
