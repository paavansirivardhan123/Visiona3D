"""Training step and loop for the unified model.

`compute_losses` is safe to call anywhere (it only computes losses). `fit` is the real training loop: DO NOT CALL IT
until the user has given permission to train. Nothing in this repository calls it automatically.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch

from . import taxonomy as T
from .loss import semantic_loss


def instance_hyp():
    """Default Ultralytics hyper-parameters as the instance loss expects them (box, cls, dfl gains, overlap_mask, ...)."""
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    return get_cfg(DEFAULT_CFG)


def to_device_instance(batch: dict, device) -> dict:
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out["img"] = out["img"].float() / 255.0
    return out


def compute_losses(model, inst_batch: dict | None, sem_batch: dict | None, device, w_inst: float = 1.0,
                   w_sem: float = 1.0, pos_weight: float | None = None):
    """Combined loss L = w_inst * L_instance + w_sem * L_semantic.

    Either batch may be None (that term is then skipped). Both flow through the same backbone and neck.
    Returns (total, parts) where parts holds detached floats for monitoring each task separately.
    """
    parts, total = {}, torch.zeros((), device=device)
    if inst_batch is not None:
        b = to_device_instance(inst_batch, device)
        preds = model.det(b["img"])
        loss_sum, items = model.det.loss(b, preds)                   # Ultralytics returns (sum * batch, items)
        l_inst = loss_sum.sum() / b["img"].shape[0]                   # per-image mean
        total = total + w_inst * l_inst
        parts["instance"] = float(l_inst.detach())
        if isinstance(items, dict):                                   # Ultralytics >= 8.4.5x returns a dict of loss terms
            parts["instance_items"] = [float(v.detach().sum()) if hasattr(v, "detach") else float(v) for v in items.values()]
            parts["instance_item_names"] = list(items.keys())
        else:                                                         # older versions return a tensor
            parts["instance_items"] = [float(x) for x in items.detach().cpu().tolist()]
    if sem_batch is not None:
        x = sem_batch["img"].to(device).float() / 255.0
        out = model.forward_semantic(x)
        l_sem, per_class = semantic_loss(out, sem_batch["sem"].to(device), sem_batch["supervised"].to(device),
                                         pos_weight=pos_weight)
        total = total + w_sem * l_sem
        parts["semantic"] = float(l_sem.detach())
        parts["semantic_per_class"] = {T.SEM_CLASSES[k]: v for k, v in per_class.items()}
    return total, parts


def make_optimizer(model, lr: float = 2e-4, weight_decay: float = 5e-4, trunk_lr_mult: float = 0.1):
    """AdamW with no weight decay on BatchNorm/bias, and a smaller learning rate for the pretrained trunk."""
    trunk, head_decay, head_no_decay, trunk_no_decay = [], [], [], []
    trunk_ids = {id(p) for l in list(model.det.model)[: model.BACKBONE_LAYERS + 12] for p in l.parameters()}
    inst_head_ids = {id(p) for p in model.det.model[-1].parameters()}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1
        is_trunk = id(p) in trunk_ids and id(p) not in inst_head_ids
        (trunk_no_decay if (is_trunk and no_decay) else trunk if is_trunk else head_no_decay if no_decay else head_decay).append(p)
    groups = [
        {"params": trunk, "lr": lr * trunk_lr_mult, "weight_decay": weight_decay},
        {"params": trunk_no_decay, "lr": lr * trunk_lr_mult, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": lr, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["initial_lr"] = g["lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def lr_factor(step: int, total_steps: int, warmup_steps: int, final: float = 0.05) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def save_checkpoint(model, ema, out_dir: Path, tag: str, epoch: int, extra: dict | None = None):
    """Writes (1) a standard Ultralytics-loadable instance checkpoint, (2) the semantic head, (3) a full resume file."""
    import ultralytics
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ema.ema if ema is not None else model
    det = copy.deepcopy(src.det).half()
    torch.save({"epoch": epoch, "model": det, "train_args": {}, "version": ultralytics.__version__},
               out_dir / f"{tag}_instance.pt")
    torch.save({"state_dict": src.sem.state_dict(), "classes": T.SEM_CLASSES, "channels": model.feature_channels},
               out_dir / f"{tag}_semantic_head.pt")
    torch.save({"epoch": epoch, "model": model.state_dict(), "ema": None if ema is None else ema.ema.state_dict(),
                **(extra or {})}, out_dir / f"{tag}_resume.pt")


def fit(model, inst_loader, sem_loader, device, epochs: int, steps_per_epoch: int, out_dir: Path,
        lr: float = 2e-4, w_inst: float = 1.0, w_sem: float = 1.0, freeze=None, amp: bool = True,
        warmup_steps: int = 300, resume: Path | None = None, log_every: int = 50, patience: int = 20,
        val_fn=None, pos_weight: float | None = None, clip: float = 10.0, save_every: int = 5,
        time_budget_s: float | None = None):
    """THE TRAINING LOOP. Not called anywhere automatically. Requires explicit permission.

    val_fn(model) -> dict with a scalar 'score' (higher is better); called once per epoch, best checkpoint kept.
    """
    from ultralytics.utils.torch_utils import ModelEMA

    model.to(device)
    model.det.args = instance_hyp()
    model.set_freeze(freeze)
    opt = make_optimizer(model, lr=lr)
    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler(enabled=amp and device.type == "cuda")
    total_steps = epochs * steps_per_epoch
    start_epoch, best, bad = 0, -1e9, 0
    if resume is not None:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", best)
    inst_it, sem_it = iter(inst_loader), iter(sem_loader)

    def nxt(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    step = start_epoch * steps_per_epoch
    t_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        model.frozen_to_eval()                                        # frozen BatchNorm must not drift
        t0 = time.time()
        for _ in range(steps_per_epoch):
            ib, inst_it = nxt(inst_it, inst_loader)
            sb, sem_it = nxt(sem_it, sem_loader)
            f = lr_factor(step, total_steps, warmup_steps)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * f
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                loss, parts = compute_losses(model, ib, sb, device, w_inst, w_sem, pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {float(loss):.3f} inst {parts.get('instance')} sem {parts.get('semantic')}")
            step += 1
        score = None
        if val_fn is not None:
            res = val_fn(ema.ema)
            score = res["score"]
            print(f"epoch {epoch} val {res} ({time.time() - t0:.0f}s)")
        save_checkpoint(model, ema, out_dir, "last", epoch, {"best": best})
        if time_budget_s is not None and time.time() - t_start > time_budget_s:
            print(f"time budget of {time_budget_s / 3600:.1f} h used: stopping after epoch {epoch}; resume later from last_resume.pt")
            break
        if save_every and (epoch + 1) % save_every == 0:                 # a kept snapshot every few epochs (stage checkpoints)
            save_checkpoint(model, ema, out_dir, f"epoch{epoch + 1:03d}", epoch, {"best": best})
        if score is not None and score > best:
            best, bad = score, 0
            save_checkpoint(model, ema, out_dir, "best", epoch, {"best": best})
        elif score is not None:
            bad += 1
            if bad >= patience:
                print("early stop: no validation improvement for", patience, "epochs")
                break
    return model
