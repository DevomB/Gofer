"""Train the Gofer net from self-play data (JSONL or GOFER shards).

Backward compatible with the v3 loop's invocation::

    python training/train_bootstrap.py --data training/data/replay.jsonl \
        --epochs 15 --lr 0.001 --out-dir training/state/run --resume training/state/best.pt

New capabilities (all optional; see training/README.md):

    --data training/data/shards/ --window-rows 250000 --window-decay 2   # shard dir + recency window
    --arch gpool-6x64 --teacher training/state/best.pt                   # new arch, distilled from champion
    --config configs/train.toml --amp bf16 --compile --continue          # config file, perf, crash-resume
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from gofer_train.data import split_by_game  # noqa: E402,F401  (re-export)
from gofer_train.trainer import TrainConfig, Trainer, pick_device  # noqa: E402

SPLIT_SEED = 42
OWNERSHIP_LOSS_WEIGHT = 0.15  # legacy constant (net_size_ablation.py); see LossWeights


def training_device() -> torch.device:
    return pick_device("auto")


# ---- legacy dataclass API (tests, net_size_ablation) ------------------------


@dataclass
class TrainOptions:
    resume: Path | None = None
    init_from: Path | None = None
    val_split: float = 0.1
    patience: int = 5


@dataclass
class TrainJob:
    data: Path
    epochs: int
    lr: float
    out_dir: Path
    options: TrainOptions


def split_indices(n: int, val_split: float, seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
    """Row-level split kept for net_size_ablation.py reproducibility."""
    import random

    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    if n < 2:
        return idx, []
    n_val = min(max(1, int(round(n * val_split))), n - 1)
    return idx[n_val:], idx[:n_val]


def train(job: TrainJob, **overrides: object) -> Path:
    init = job.options.resume if job.options.resume and job.options.resume.exists() else job.options.init_from
    cfg = TrainConfig(
        data=[str(job.data)],
        out_dir=str(job.out_dir),
        epochs=float(job.epochs),
        lr=job.lr,
        val_split=job.options.val_split,
        patience=job.options.patience,
        init_from=str(init) if init and Path(init).exists() else "",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    Trainer(cfg).run()
    return job.out_dir / "best.pt"


# ---- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="", help="TOML file ([train] table); CLI flags override it")
    p.add_argument("--data", nargs="+", default=None, help="JSONL file(s), .npz shard(s) or shard dir(s)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--arch", default=None, help="preset (legacy-6x64, gpool-6x64, gpool-10x96, ...) or kind-BxC")
    p.add_argument("--board-size", type=int, default=None)
    g = p.add_argument_group("schedule")
    g.add_argument("--epochs", type=float, default=None)
    g.add_argument("--steps", type=int, default=None, help="total optimizer steps (overrides --epochs)")
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--warmup-steps", type=int, default=None)
    g.add_argument("--optimizer", choices=("sgd", "adamw"), default=None)
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--ema-decay", type=float, default=None, help="0 disables EMA")
    g.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    d = p.add_argument_group("data")
    d.add_argument("--val-split", type=float, default=None)
    d.add_argument("--window-rows", type=int, default=None, help="train on the newest N rows (0 = all)")
    d.add_argument("--window-decay", type=float, default=None, help="recency sampling: newest e^F x oldest")
    d.add_argument("--patience", type=int, default=None, help="evals without improvement before stopping")
    d.add_argument("--eval-every", type=int, default=None, help="steps between evals (0 = per epoch)")
    i = p.add_argument_group("init")
    i.add_argument("--resume", default="", help="warm-start weights (champion best.pt) if the file exists")
    i.add_argument("--init-from", default="", help="one-time seed weights")
    i.add_argument("--teacher", default=None, help="distill from this checkpoint (any arch)")
    i.add_argument("--continue", dest="resume_run", action="store_true", default=None,
                   help="continue an interrupted run from <out-dir>/last.pt (optimizer + step)")
    i.add_argument("--ckpt-every", type=int, default=None, help="crash-safety save interval in steps")
    f = p.add_argument_group("performance")
    f.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:1 | mps")
    f.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default=None)
    f.add_argument("--compile", action="store_true", default=None)
    f.add_argument("--channels-last", action="store_true", default=None)
    f.add_argument("--seed", type=int, default=None)
    f.add_argument("--tensorboard", action="store_true", default=None)
    f.add_argument("--quiet", action="store_true", default=None)
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    over = {
        k: getattr(args, k)
        for k in (
            "data", "out_dir", "arch", "board_size", "epochs", "steps", "batch_size", "lr",
            "warmup_steps", "optimizer", "weight_decay", "ema_decay", "augment", "val_split",
            "window_rows", "window_decay", "patience", "eval_every", "teacher", "resume_run",
            "ckpt_every", "device", "amp", "compile", "channels_last", "seed", "tensorboard", "quiet",
        )
    }
    cfg = TrainConfig.from_toml(Path(args.config), **over) if args.config else TrainConfig()
    if not args.config:
        for k, v in over.items():
            if v is not None:
                setattr(cfg, k, v)
    # --resume keeps its v3 meaning: warm-start weights when the file exists.
    if args.resume and Path(args.resume).exists():
        cfg.init_from = args.resume
    elif args.init_from and Path(args.init_from).exists():
        cfg.init_from = args.init_from
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    summary = Trainer(cfg).run()
    print(f"checkpoint: {summary['best_pt']}")


if __name__ == "__main__":
    main()
