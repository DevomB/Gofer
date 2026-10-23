"""Optimization: losses, training steps and checkpoints."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from training.data import Rows, Sampler
from training.model import GoferNet, ModelConfig


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    ownership_weight: float = 0.15
    q_weight: float = 0.0  # blend of root search value into the value target
    grad_clip: float = 5.0
    warmup_steps: int = 200


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def losses(model: GoferNet, batch) -> dict[str, torch.Tensor]:
    spatial, glob, policy, value, own, own_w = batch
    logits, v, o = model(spatial, glob)
    p_loss = -(policy * F.log_softmax(logits, dim=1)).sum(1).mean()
    # entropy of the target: the floor of the policy loss (reported, not optimized)
    p_entropy = -(policy * torch.log(policy.clamp_min(1e-12))).sum(1).mean()
    v_loss = F.mse_loss(v, value)
    o_loss = ((o - own) ** 2).mean(1)
    o_loss = (o_loss * own_w).sum() / own_w.sum().clamp_min(1.0)
    return {"policy": p_loss, "policy_entropy": p_entropy, "value": v_loss, "ownership": o_loss}


def make_optimizer(model: GoferNet, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 or "bn" in name else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )


def train_steps(
    model: GoferNet,
    opt: torch.optim.Optimizer,
    rows: Rows,
    steps: int,
    cfg: TrainConfig,
    lr: float,
    global_step: int,
    seed: int,
    log=print,
) -> tuple[dict[str, float], int]:
    """Run `steps` minibatch updates; returns mean metrics and the new global step."""
    dev = next(model.parameters()).device
    sampler = Sampler(rows, cfg.batch_size, cfg.q_weight, seed)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    sums: dict[str, float] = {}
    t0 = time.time()
    for i in range(steps):
        warm = min(1.0, (global_step + 1) / max(cfg.warmup_steps, 1))
        for g in opt.param_groups:
            g["lr"] = lr * warm
        batch = sampler.batch(dev)
        with torch.autocast(dev.type, enabled=use_amp):
            parts = losses(model, batch)
            loss = parts["policy"] + cfg.value_weight * parts["value"] + cfg.ownership_weight * parts["ownership"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        global_step += 1
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v.item()
        if (i + 1) % 500 == 0:
            log(f"  step {i + 1}/{steps}  policy {sums['policy'] / (i + 1):.3f}  value {sums['value'] / (i + 1):.3f}")
    metrics = {k: v / max(steps, 1) for k, v in sums.items()}
    metrics["steps"] = steps
    metrics["seconds"] = time.time() - t0
    metrics["samples_per_second"] = steps * cfg.batch_size / max(metrics["seconds"], 1e-9)
    return metrics, global_step


@torch.no_grad()
def evaluate(model: GoferNet, rows: Rows, batch_size: int = 1024) -> dict[str, float]:
    """Loss on held-out rows (no augmentation)."""
    dev = next(model.parameters()).device
    model.eval()
    sums: dict[str, float] = {}
    n = len(rows)
    for s in range(0, n, batch_size):
        sl = slice(s, min(n, s + batch_size))
        batch = (
            torch.from_numpy(rows.spatial[sl].astype("float32")).to(dev),
            torch.from_numpy(rows.glob[sl]).to(dev),
            torch.from_numpy(rows.policy[sl]).to(dev),
            torch.from_numpy(rows.value[sl]).to(dev),
            torch.from_numpy(rows.ownership[sl].astype("float32")).to(dev),
            torch.from_numpy(rows.ownership_weight[sl]).to(dev),
        )
        parts = losses(model, batch)
        w = sl.stop - sl.start
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v.item() * w
    return {k: v / max(n, 1) for k, v in sums.items()}


def save_checkpoint(path: Path, model: GoferNet, opt: torch.optim.Optimizer, step: int, extra: dict | None = None) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"model_config": model.cfg.to_dict(), "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, **(extra or {})},
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, cfg: TrainConfig, dev: torch.device) -> tuple[GoferNet, torch.optim.Optimizer, int]:
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = GoferNet(ModelConfig(**ck["model_config"])).to(dev)
    model.load_state_dict(ck["model"])
    opt = make_optimizer(model, cfg)
    opt.load_state_dict(ck["optimizer"])
    return model, opt, int(ck["step"])


def cosine_lr(base: float, gen: int, total_gens: int, floor: float = 0.1) -> float:
    """Per-generation learning rate: cosine from base down to floor*base."""
    if total_gens <= 1:
        return base
    t = min(gen / (total_gens - 1), 1.0)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))
