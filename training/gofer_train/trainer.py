"""Step-based trainer: AMP, EMA, warmup+cosine, crash-safe checkpoints, distillation.

Outputs in ``out_dir``:

  best.pt        plain state_dict of the best (EMA) weights — back-compatible with
                 every existing consumer (export_onnx, cycle scripts, --resume)
  best.ckpt      {"arch", "state_dict", "step", "val", ...} — self-describing
  last.pt        full resume state (model, EMA, optimizer, scheduler, counters);
                 also carries the legacy keys epoch / train_loss / val_loss
  metrics.jsonl  one JSON object per eval
  summary.json   final result for the orchestrator (keys: see SUMMARY_KEYS)
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import DeviceData, Sampler, split_by_game
from .losses import LossWeights, compute_loss
from .nets import ArchConfig, build_net, count_params, infer_arch, load_weights, resolve_arch
from .shards import Rows, load_rows

SUMMARY_KEYS = (
    "status",  # "ok" | "early_stop"
    "best_val_loss",
    "best_step",
    "best_epoch",
    "final_val_loss",
    "val",  # dict of best-eval metrics: policy, value, ownership, score, policy_acc, ...
    "steps",
    "epochs",
    "samples_seen",
    "samples_per_sec",
    "train_rows",
    "val_rows",
    "val_games",
    "arch",
    "params",
    "device",
    "best_pt",
    "wall_sec",
)


@dataclass
class TrainConfig:
    data: list[str] = field(default_factory=lambda: ["training/data/samples.jsonl"])
    out_dir: str = "training/checkpoints"
    arch: str = ""  # preset / kind-BxC; empty = infer from init weights, else legacy-6x64
    board_size: int = 9
    # schedule
    epochs: float = 25.0
    steps: int = 0  # >0 overrides epochs
    batch_size: int = 256
    lr: float = 0.01
    warmup_steps: int = -1  # -1 = 5% of total (max 500)
    min_lr_frac: float = 0.05
    optimizer: str = "sgd"  # sgd (nesterov) | adamw
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    # regularization / averaging
    ema_decay: float = 0.999  # 0 disables EMA
    augment: bool = True
    # data
    val_split: float = 0.1
    window_rows: int = 0
    window_decay: float = 0.0
    # eval / stopping
    eval_every: int = 0  # steps; 0 = once per epoch
    patience: int = 5  # evals without improvement; 0 disables
    # init
    init_from: str = ""  # weights to start from (strict shapes)
    teacher: str = ""  # checkpoint to distill from (any arch)
    resume_run: bool = False  # continue an interrupted run from out_dir/last.pt
    ckpt_every: int = 0  # steps between crash-safety saves; 0 = every eval
    # performance
    device: str = "auto"
    amp: str = "auto"  # auto | bf16 | fp16 | off
    compile: bool = False
    channels_last: bool = False
    seed: int = 1234
    split_seed: int = 42
    loss: LossWeights = field(default_factory=LossWeights)
    tensorboard: bool = False
    quiet: bool = False

    @classmethod
    def from_toml(cls, path: Path, **overrides: Any) -> "TrainConfig":
        import tomllib

        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        section = raw.get("train", raw)
        cfg = cls()
        for k, v in section.items():
            if k == "loss":
                cfg.loss = LossWeights(**v)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        if isinstance(cfg.data, str):
            cfg.data = [cfg.data]
        return cfg


# ------------------------------------------------------------ utilities ----


def pick_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():  # includes ROCm builds
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(spec: str, device: torch.device) -> torch.dtype | None:
    if spec == "off" or device.type == "cpu" and spec == "auto":
        return None
    if spec == "fp16":
        return torch.float16
    if spec == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def load_state(path: Path) -> tuple[dict[str, torch.Tensor], ArchConfig | None]:
    """Load weights from best.pt / best.ckpt / last.pt; returns (state, arch or None)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    arch = None
    if isinstance(obj, dict) and "state_dict" in obj:
        if obj.get("arch"):
            arch = ArchConfig.from_dict(obj["arch"])
        state = obj.get("ema_state_dict") or obj["state_dict"]
    else:
        state = obj
    # Companion best.ckpt next to a bare best.pt carries the arch.
    if arch is None and path.suffix == ".pt":
        side = path.with_suffix(".ckpt")
        if side.exists():
            meta = torch.load(side, map_location="cpu", weights_only=False)
            if isinstance(meta, dict) and meta.get("arch"):
                arch = ArchConfig.from_dict(meta["arch"])
    return state, arch


def load_net(path: Path, board_size: int = 9) -> tuple[nn.Module, ArchConfig]:
    state, arch = load_state(path)
    arch = arch or infer_arch(state, board_size)
    net = build_net(arch)
    load_weights(net, state)
    return net, arch


def _decay_groups(net: nn.Module, wd: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(net: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = _decay_groups(net, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.99))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def lr_lambda(total: int, warmup: int, min_frac: float):
    def f(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return f


class EMA:
    """Exponential moving average of parameters and buffers (BN stats included)."""

    def __init__(self, net: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in net.state_dict().items()}
        self._float_keys = [k for k, v in self.shadow.items() if v.dtype.is_floating_point]
        self._other_keys = [k for k, v in self.shadow.items() if not v.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, net: nn.Module, step: int) -> None:
        # Warm up the decay so early (random) weights are forgotten quickly.
        d = min(self.decay, (1 + step) / (10 + step))
        live = net.state_dict()
        dst = [self.shadow[k] for k in self._float_keys]
        src = [live[k].detach() for k in self._float_keys]
        torch._foreach_lerp_(dst, src, 1 - d)  # one fused kernel per dtype/device group
        for k in self._other_keys:
            self.shadow[k].copy_(live[k])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            self.shadow[k].copy_(v)


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------- training ----


@torch.no_grad()
def evaluate(
    net: nn.Module,
    data: DeviceData,
    idx: np.ndarray,
    cfg: TrainConfig,
    dtype: torch.dtype | None,
) -> dict[str, float]:
    was_training = net.training
    net.eval()
    sums: dict[str, float] = {}
    count = 0
    bs = max(256, cfg.batch_size)
    for i in range(0, len(idx), bs):
        chunk = torch.from_numpy(idx[i : i + bs])
        batch = data.gather(chunk, augment=False)
        with _autocast(data.device, dtype):
            out = net.forward_all(batch.spatial, batch.globals)
        _, parts = compute_loss(out, batch, cfg.loss)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v) * len(chunk)
        count += len(chunk)
    net.train(was_training)
    return {k: v / max(count, 1) for k, v in sums.items()}


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


class Trainer:
    def __init__(self, cfg: TrainConfig, rows: Rows | None = None) -> None:
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "metrics.jsonl"
        if not cfg.resume_run and self.log_path.exists():
            self.log_path.unlink()

        t0 = time.time()
        self.rows = rows if rows is not None else load_rows(
            [Path(d) for d in cfg.data], board_size=cfg.board_size, window_rows=cfg.window_rows
        )
        self.train_idx, self.val_idx = split_by_game(self.rows, cfg.val_split, cfg.split_seed)
        self.data = DeviceData(self.rows, self.device)
        self._say(
            f"data: {len(self.rows)} rows ({len(self.train_idx)} train / {len(self.val_idx)} val, "
            f"{len(np.unique(self.rows.game_id))} games, full_search={self.rows.full_search.mean():.2f}) "
            f"loaded in {time.time() - t0:.1f}s"
        )

        init_state, init_arch = (None, None)
        if cfg.init_from:
            init_state, init_arch = load_state(Path(cfg.init_from))
            init_arch = init_arch or infer_arch(init_state, cfg.board_size)
        if cfg.arch:
            self.arch = resolve_arch(cfg.arch, board_size=cfg.board_size)
        else:
            self.arch = init_arch or resolve_arch("legacy-6x64", board_size=cfg.board_size)
        self.net = build_net(self.arch).to(self.device)
        if init_state is not None:
            try:
                missing = load_weights(self.net, init_state)
                if missing:
                    self._say(f"init: {cfg.init_from} has no {missing}; those heads start fresh")
            except RuntimeError as e:
                raise SystemExit(
                    f"--init-from {cfg.init_from} does not fit arch {self.arch.name}: {e}\n"
                    "To change architecture, start fresh and distill from it with --teacher."
                ) from e
        if cfg.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)

        self.teacher: nn.Module | None = None
        if cfg.teacher:
            self.teacher, t_arch = load_net(Path(cfg.teacher), cfg.board_size)
            self.teacher.to(self.device).eval()
            if cfg.loss.distill <= 0:
                cfg.loss.distill = 1.0
            self._say(f"teacher: {t_arch.name} from {cfg.teacher} (distill weight {cfg.loss.distill})")

        self.sampler = Sampler(self.train_idx, cfg.batch_size, window_decay=cfg.window_decay, seed=cfg.seed)
        spe = self.sampler.steps_per_epoch
        self.total_steps = cfg.steps if cfg.steps > 0 else max(1, int(round(cfg.epochs * spe)))
        self.eval_every = cfg.eval_every or spe
        warm = cfg.warmup_steps if cfg.warmup_steps >= 0 else min(500, self.total_steps // 20)
        self.opt = make_optimizer(self.net, cfg)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda(self.total_steps, warm, cfg.min_lr_frac)
        )
        self.ema = EMA(self.net, cfg.ema_decay) if cfg.ema_decay > 0 else None
        self.dtype = amp_dtype(cfg.amp, self.device)
        self.scaler = torch.amp.GradScaler(enabled=self.dtype == torch.float16)
        self.fwd = self.net.forward_all
        if cfg.compile:
            # Inductor on CPU needs a C++ toolchain and compiles for minutes; only worth it on GPU.
            if self.device.type == "cuda":
                self.fwd = torch.compile(self.net.forward_all)
            else:
                self._say(f"--compile ignored on {self.device.type} (CUDA only)")

        self.step = 0
        self.best_val = float("inf")
        self.best_step = 0
        self.best_metrics: dict[str, float] = {}
        self.stale = 0
        self.samples_seen = 0
        self.last_val = float("nan")
        self.last_train = float("nan")
        if cfg.resume_run and (self.out / "last.pt").exists():
            self._restore(self.out / "last.pt")

        self.tb = None
        if cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.out / "tb"))
            except ImportError:
                self._say("tensorboard not installed; skipping")
        self._say(
            f"device: {self.device}"
            + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else "")
            + f" amp={self.dtype} arch={self.arch.name} params={count_params(self.net):,} "
            f"steps={self.total_steps} batch={self.sampler.batch_size} opt={cfg.optimizer} lr={cfg.lr}"
        )

    # ---------------------------------------------------------------- io ----

    def _say(self, msg: str) -> None:
        if not self.cfg.quiet:
            print(msg, flush=True)

    def _eval_net(self) -> nn.Module:
        if self.ema is None:
            return self.net
        shadow = build_net(self.arch).to(self.device)
        shadow.load_state_dict(self.ema.state_dict())
        return shadow

    def _save_last(self) -> None:
        epoch = self.step / self.sampler.steps_per_epoch
        _atomic_save(
            {
                "state_dict": self.net.state_dict(),
                "ema_state_dict": self.ema.state_dict() if self.ema else None,
                "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(),
                "scaler": self.scaler.state_dict(),
                "arch": self.arch.to_dict(),
                "step": self.step,
                "total_steps": self.total_steps,
                "epoch": int(math.ceil(epoch)),
                "train_loss": self.last_train,
                "val_loss": self.last_val,
                "best_val": self.best_val,
                "best_step": self.best_step,
                "best_metrics": self.best_metrics,
                "stale": self.stale,
                "samples_seen": self.samples_seen,
                "config": _jsonable(asdict(self.cfg)),
            },
            self.out / "last.pt",
        )

    def _restore(self, path: Path) -> None:
        st = torch.load(path, map_location=self.device, weights_only=False)
        if st.get("arch") and ArchConfig.from_dict(st["arch"]) != self.arch:
            raise SystemExit(f"{path} arch {st['arch']} != requested {self.arch.to_dict()}")
        self.net.load_state_dict(st["state_dict"])
        if self.ema and st.get("ema_state_dict"):
            self.ema.load_state_dict(st["ema_state_dict"])
        if "optimizer" in st:
            self.opt.load_state_dict(st["optimizer"])
            self.sched.load_state_dict(st["scheduler"])
            self.scaler.load_state_dict(st.get("scaler", {}))
        self.step = int(st.get("step", 0))
        self.best_val = float(st.get("best_val", float("inf")))
        self.best_step = int(st.get("best_step", 0))
        self.best_metrics = st.get("best_metrics", {})
        self.stale = int(st.get("stale", 0))
        self.samples_seen = int(st.get("samples_seen", 0))
        self._say(f"resumed run from {path} at step {self.step}/{self.total_steps}")

    def _save_best(self, metrics: dict[str, float]) -> None:
        state = {k: v.detach().cpu() for k, v in (self.ema.state_dict() if self.ema else self.net.state_dict()).items()}
        _atomic_save(state, self.out / "best.pt")
        _atomic_save(
            {
                "arch": self.arch.to_dict(),
                "state_dict": state,
                "step": self.step,
                "val": metrics,
                "format": "gofer-ckpt",
                "version": 1,
            },
            self.out / "best.ckpt",
        )

    # -------------------------------------------------------------- loop ----

    def _train_step(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = self.data.gather(idx, augment=self.cfg.augment, generator=self.sampler.gen)
        spatial = batch.spatial
        if self.cfg.channels_last:
            spatial = spatial.contiguous(memory_format=torch.channels_last)
        teacher_out = None
        with _autocast(self.device, self.dtype):
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_out = self.teacher.forward_all(spatial, batch.globals)
            out = self.fwd(spatial, batch.globals)
        loss, parts = compute_loss(out, batch, self.cfg.loss, teacher_out)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        if self.ema:
            self.ema.update(self.net, self.step)
        self.step += 1
        self.samples_seen += len(idx)
        return parts

    def _do_eval(self, train_parts: dict[str, float], t_start: float) -> bool:
        """Evaluate, log, checkpoint. Returns True to stop early."""
        net = self._eval_net()
        if len(self.val_idx):
            val = evaluate(net, self.data, self.val_idx, self.cfg, self.dtype)
        else:
            val = dict(train_parts)
        self.last_val = val["total"]
        self.last_train = train_parts.get("total", float("nan"))
        improved = val["total"] < self.best_val
        if improved:
            self.best_val, self.best_step, self.best_metrics, self.stale = val["total"], self.step, val, 0
            self._save_best(val)
        else:
            self.stale += 1
        elapsed = time.time() - t_start
        rec = {
            "step": self.step,
            "epoch": round(self.step / self.sampler.steps_per_epoch, 3),
            "lr": self.sched.get_last_lr()[0],
            "train": train_parts,
            "val": val,
            "best": improved,
            "samples_per_sec": self._rate(elapsed),
            "wall_sec": round(elapsed, 2),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if self.tb:
            for split, d in (("train", train_parts), ("val", val)):
                for k, v in d.items():
                    self.tb.add_scalar(f"{split}/{k}", v, self.step)
            self.tb.add_scalar("lr", rec["lr"], self.step)
        self._say(
            f"step {self.step}/{self.total_steps} ep {rec['epoch']:.2f} lr={rec['lr']:.2e} "
            f"train={self.last_train:.4f} val={val['total']:.4f} "
            f"(pol={val.get('policy', 0):.3f} acc={val.get('policy_acc', 0):.3f} "
            f"val={val.get('value', 0):.3f} own={val.get('ownership', 0):.3f}) "
            f"{rec['samples_per_sec']:.0f} samp/s{' *' if improved else ''}"
        )
        self._save_last()
        return self.cfg.patience > 0 and self.stale >= self.cfg.patience

    def _rate(self, elapsed: float) -> float:
        return (self.samples_seen - self._seen0) / max(elapsed, 1e-9)

    def run(self) -> dict[str, Any]:
        self.net.train()
        t_start = time.time()
        self._seen0 = self.samples_seen
        status = "ok"
        acc: dict[str, float] = {}
        n_acc = 0
        stop = False
        while self.step < self.total_steps and not stop:
            for idx in self.sampler.epoch():
                parts = self._train_step(idx)
                for k, v in parts.items():
                    acc[k] = acc.get(k, 0.0) + float(v)
                n_acc += 1
                if self.cfg.ckpt_every and self.step % self.cfg.ckpt_every == 0:
                    self._save_last()
                if self.step % self.eval_every == 0 or self.step >= self.total_steps:
                    train_parts = {k: v / max(n_acc, 1) for k, v in acc.items()}
                    acc, n_acc = {}, 0
                    if self._do_eval(train_parts, t_start):
                        self._say(f"early stop at step {self.step} (patience={self.cfg.patience})")
                        status, stop = "early_stop", True
                        break
                if self.step >= self.total_steps:
                    break
        if not (self.out / "best.pt").exists():
            self._save_best(self.best_metrics)
        wall = time.time() - t_start
        summary = {
            "status": status,
            "best_val_loss": self.best_val,
            "best_step": self.best_step,
            "best_epoch": int(math.ceil(self.best_step / self.sampler.steps_per_epoch)),
            "final_val_loss": self.last_val,
            "val": self.best_metrics,
            "steps": self.step,
            "epochs": round(self.step / self.sampler.steps_per_epoch, 3),
            "samples_seen": self.samples_seen,
            "samples_per_sec": round(self._rate(wall), 1),
            "train_rows": int(len(self.train_idx)),
            "val_rows": int(len(self.val_idx)),
            "val_games": int(len(np.unique(self.rows.game_id[self.val_idx]))) if len(self.val_idx) else 0,
            "arch": self.arch.to_dict(),
            "params": count_params(self.net),
            "device": str(self.device),
            "best_pt": str(self.out / "best.pt"),
            "wall_sec": round(wall, 2),
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self._say(f"best step={self.best_step} val_loss={self.best_val:.4f} -> {self.out / 'best.pt'}")
        if self.tb:
            self.tb.close()
        return summary


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def train(cfg: TrainConfig) -> dict[str, Any]:
    return Trainer(cfg).run()
