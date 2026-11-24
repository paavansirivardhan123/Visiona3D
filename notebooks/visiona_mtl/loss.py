"""Semantic loss for the unified model: per-class sigmoid BCE + Dice with per-image supervision.

Why not softmax cross-entropy with ignore_index=255 (what Ultralytics' native semantic task uses)?
In a GDD image only glass is labeled. Under a softmax every non-glass pixel is either ignored (so the model only ever
sees glass positives) or forced into some other class. Here, each image carries `supervised`: the classes it can say
both "yes" and "no" about. For a class c in that set:
    target == c            -> positive
    target in {other classes, 254 (labeled other)} -> negative
    target == 255 (ignore) -> not counted
Classes not in the image's supervised set are not counted at all for that image.
A bonus: sigmoid scores give a per-pixel probability for each class, which the safety layer can calibrate.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import taxonomy as T


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, supervised: torch.Tensor,
                    pos_weight: torch.Tensor | float | None = None, dice_weight: float = 0.5, eps: float = 1.0):
    """logits (B,C,h,w) float; target (B,h,w) uint8/int (0..C-1, 254, 255); supervised (B,C) bool.

    Returns (loss, per_class) where per_class is {class_index: float} for classes that had at least one valid pixel.
    """
    b, c, h, w = logits.shape
    t = target.long()
    ignore = t == T.IGNORE
    supervised = supervised.to(device=logits.device, dtype=torch.bool)
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.full((c,), float(pos_weight), device=logits.device)
    losses, per_class = [], {}
    for k in range(c):
        valid = supervised[:, k].view(b, 1, 1) & (~ignore)
        n_valid = valid.sum()
        if n_valid == 0:
            continue
        y = (t == k).float()
        x = logits[:, k].float()
        vf = valid.float()
        pw = None if pos_weight is None else pos_weight[k]
        bce = F.binary_cross_entropy_with_logits(x, y, weight=vf, pos_weight=pw, reduction="sum") / n_valid
        p = x.sigmoid() * vf
        inter = (p * y * vf).sum()
        dice = 1.0 - (2.0 * inter + eps) / (p.sum() + (y * vf).sum() + eps)
        lk = (1.0 - dice_weight) * bce + dice_weight * dice
        losses.append(lk)
        per_class[k] = float(lk.detach())
    if not losses:                                   # nothing supervised in this batch: zero that still has a graph
        return logits.sum() * 0.0, per_class
    return torch.stack(losses).mean(), per_class


def downsample_target(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour resize of an integer label map (B,H,W) to (B,h,w); never blends class ids."""
    if tuple(target.shape[1:]) == tuple(size):
        return target
    return F.interpolate(target.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1).to(target.dtype)


def semantic_loss(sem_out, target: torch.Tensor, supervised: torch.Tensor, aux_weight: float = 0.4, **kw):
    """sem_out: logits (B,C,h,w) or (main, aux) in training. Supervision is applied at 1/4 input resolution."""
    aux = None
    if isinstance(sem_out, tuple):
        sem_out, aux = sem_out
    size = (target.shape[1] // 4, target.shape[2] // 4)
    main = F.interpolate(sem_out, size=size, mode="bilinear", align_corners=False)
    tgt = downsample_target(target, size)
    loss, per_class = masked_bce_dice(main, tgt, supervised, **kw)
    if aux is not None:
        aux_up = F.interpolate(aux, size=size, mode="bilinear", align_corners=False)
        a_loss, _ = masked_bce_dice(aux_up, tgt, supervised, **kw)
        loss = loss + aux_weight * a_loss
    return loss, per_class


@torch.no_grad()
def compose(logits: torch.Tensor, threshold: float = 0.5):
    """Per-pixel class map from sigmoid scores: argmax class if its probability >= threshold, else 255 (none).

    Returns (class_map (B,H,W) uint8, probs (B,C,H,W)). Unlike a softmax, 'none' is a legal answer, so uncertain
    pixels are never forced to be floor.
    """
    probs = logits.sigmoid()
    score, cls = probs.max(dim=1)
    out = torch.where(score >= threshold, cls, torch.full_like(cls, T.IGNORE)).to(torch.uint8)
    return out, probs
