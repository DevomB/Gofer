"""Offline net-size ablation on a frozen replay snapshot.

Does not change GoferBootstrapNet defaults or the live training pipeline.
See docs/decisions/0005-net-size-ablation.md for findings and decision.

Prerequisites
-------------
- Frozen JSONL at ``training/data/ablation/net-size-replay-snapshot.jsonl``
  (copy of production replay; do not append mid-run). See
  ``training/data/ablation/README.md``.
- Run from repo root with training on PYTHONPATH and the project venv active::

    cd /path/to/Gofer
    source .venv/bin/activate   # or .venv311 on Lightsail
    export PYTHONPATH=training

Full ablation (all candidates, ONNX export, latency)
----------------------------------------------------
Trains 6x64-baseline, 4x32, 4x48, 6x32 from scratch; writes
``.tectonix/reports/net-size-ablation/results.json``::

    python -u training/net_size_ablation.py \\
      --data training/data/ablation/net-size-replay-snapshot.jsonl \\
      --epochs 25 --out-dir .tectonix/reports/net-size-ablation

Seed variance check (subset of configs, no ONNX export)
-------------------------------------------------------
Re-trains named configs with multiple init seeds; writes
``seed-variance.json`` under ``--out-dir``::

    python -u training/net_size_ablation.py --variance \\
      --configs 4x48,6x32 --seeds 11,42,99 \\
      --epochs 25 --out-dir .tectonix/reports/net-size-ablation

Config names for ``--configs`` must match keys in ``CONFIG_LOOKUP`` below
(e.g. ``6x64-baseline``, ``4x48``, ``6x32``, ``4x32``).

Outputs
-------
- ``manifest.json`` — snapshot path, line count, hyperparameters
- ``results.json`` — full ablation metrics per candidate
- ``seed-variance.json`` — per-seed runs + mean/std/min/max summary
- ``<config>/best.pt`` — checkpoints (offline only; not deployed)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import SampleDataset
from export_onnx import PolicyValueNet
from model import BOARD_SIZE, GoferBootstrapNet
from train_bootstrap import OWNERSHIP_LOSS_WEIGHT, SPLIT_SEED, split_indices

NET_CONFIGS: list[tuple[str, int, int]] = [
    ("6x64-baseline", 6, 64),
    ("4x32", 4, 32),
    ("4x48", 4, 48),
    ("6x32", 6, 32),
]

CONFIG_LOOKUP: dict[str, tuple[int, int]] = {name: (b, c) for name, b, c in NET_CONFIGS}


@dataclass
class AblationResult:
    name: str
    blocks: int
    channels: int
    params: int
    best_val_loss: float
    final_val_loss: float
    epochs_run: int
    onnx_bytes: int
    latency_ms_median: float
    latency_ms_p95: float


def count_params(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def make_net(blocks: int, channels: int) -> GoferBootstrapNet:
    return GoferBootstrapNet(blocks=blocks, channels=channels)


def run_epoch(
    net: GoferBootstrapNet,
    loader: DataLoader,
    opt: torch.optim.Optimizer | None,
    *,
    train: bool,
) -> float:
    if train:
        net.train()
    else:
        net.eval()
    total = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for spatial, globals_, policy, value, ownership in loader:
            if train and opt is not None:
                opt.zero_grad()
            logits, pred_v, pred_own = net(spatial, globals_)
            target = policy / policy.sum(dim=1, keepdim=True).clamp(min=1e-8)
            loss_p = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_v = F.mse_loss(pred_v, value)
            loss_own = F.mse_loss(pred_own, ownership)
            loss = loss_p + loss_v + OWNERSHIP_LOSS_WEIGHT * loss_own
            if train and opt is not None:
                loss.backward()
                opt.step()
            total += loss.item()
            n_batches += 1
    return total / max(n_batches, 1)


def train_candidate(
    name: str,
    blocks: int,
    channels: int,
    data: Path,
    out_dir: Path,
    *,
    epochs: int,
    lr: float,
    val_split: float,
    init_seed: int | None = None,
) -> tuple[Path, float, float, int]:
    if init_seed is not None:
        torch.manual_seed(init_seed)

    ds = SampleDataset(data)
    train_idx, val_idx = split_indices(len(ds), val_split, SPLIT_SEED)
    gen = torch.Generator().manual_seed(init_seed if init_seed is not None else SPLIT_SEED)
    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=64, shuffle=True, generator=gen
    )
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=64, shuffle=False) if val_idx else None

    net = make_net(blocks, channels)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_val = float("inf")
    final_val = float("inf")

    for epoch in range(epochs):
        run_epoch(net, train_loader, opt, train=True)
        val_loss = run_epoch(net, val_loader, None, train=False) if val_loader else float("nan")
        final_val = val_loss
        print(f"  [{name}] epoch {epoch + 1}/{epochs} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(net.state_dict(), best_path)

    if not best_path.exists():
        torch.save(net.state_dict(), best_path)
    return best_path, best_val, final_val, epochs


def export_candidate(checkpoint: Path, out_path: Path, blocks: int, channels: int) -> None:
    net = make_net(blocks, channels)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    wrapper = PolicyValueNet(net)
    spatial = torch.zeros(1, 8, BOARD_SIZE, BOARD_SIZE)
    global_in = torch.zeros(1, 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (spatial, global_in),
        str(out_path),
        input_names=["spatial_input", "global_input"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "spatial_input": {0: "batch"},
            "global_input": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )


def benchmark_onnx_latency(onnx_path: Path, *, runs: int = 300) -> tuple[float, float]:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    spatial = np.zeros((1, 8, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    global_in = np.zeros((1, 4), dtype=np.float32)
    feeds = {"spatial_input": spatial, "global_input": global_in}

    for _ in range(20):
        session.run(None, feeds)

    times_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feeds)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    median = statistics.median(times_ms)
    p95 = times_ms[int(0.95 * (len(times_ms) - 1))]
    return median, p95


def run_ablation(
    data: Path,
    out_root: Path,
    *,
    epochs: int,
    lr: float,
    val_split: float,
) -> list[AblationResult]:
    results: list[AblationResult] = []
    for name, blocks, channels in NET_CONFIGS:
        print(f"\n=== training {name} ({blocks}x{channels}) ===")
        cand_dir = out_root / name
        ckpt, best_val, final_val, epochs_run = train_candidate(
            name, blocks, channels, data, cand_dir, epochs=epochs, lr=lr, val_split=val_split
        )
        net = make_net(blocks, channels)
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        params = count_params(net)

        onnx_path = cand_dir / f"gofer-9x9-{name}.onnx"
        export_candidate(ckpt, onnx_path, blocks, channels)
        median_ms, p95_ms = benchmark_onnx_latency(onnx_path)

        results.append(
            AblationResult(
                name=name,
                blocks=blocks,
                channels=channels,
                params=params,
                best_val_loss=best_val,
                final_val_loss=final_val,
                epochs_run=epochs_run,
                onnx_bytes=onnx_path.stat().st_size,
                latency_ms_median=median_ms,
                latency_ms_p95=p95_ms,
            )
        )
    return results


def run_seed_variance(
    data: Path,
    out_root: Path,
    config_names: list[str],
    seeds: list[int],
    *,
    epochs: int,
    lr: float,
    val_split: float,
) -> list[dict]:
    rows: list[dict] = []
    for name in config_names:
        if name not in CONFIG_LOOKUP:
            raise ValueError(f"unknown config {name!r}")
        blocks, channels = CONFIG_LOOKUP[name]
        for seed in seeds:
            run_name = f"{name}/seed-{seed}"
            print(f"\n=== training {run_name} ({blocks}x{channels}) init_seed={seed} ===")
            cand_dir = out_root / name / f"seed-{seed}"
            _, best_val, final_val, epochs_run = train_candidate(
                run_name,
                blocks,
                channels,
                data,
                cand_dir,
                epochs=epochs,
                lr=lr,
                val_split=val_split,
                init_seed=seed,
            )
            rows.append(
                {
                    "name": name,
                    "blocks": blocks,
                    "channels": channels,
                    "init_seed": seed,
                    "best_val_loss": best_val,
                    "final_val_loss": final_val,
                    "epochs_run": epochs_run,
                }
            )
    return rows


def summarize_variance(rows: list[dict]) -> list[dict]:
    by_name: dict[str, list[float]] = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row["best_val_loss"])
    summary: list[dict] = []
    for name, vals in sorted(by_name.items()):
        summary.append(
            {
                "name": name,
                "n": len(vals),
                "mean": statistics.mean(vals),
                "stddev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "vals": vals,
            }
        )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Net-size ablation on frozen replay snapshot")
    p.add_argument(
        "--data",
        default="training/data/ablation/net-size-replay-snapshot.jsonl",
        help="frozen replay JSONL (must not change mid-run)",
    )
    p.add_argument("--out-dir", default=".tectonix/reports/net-size-ablation")
    p.add_argument("--epochs", type=int, default=25, help="fixed epochs per candidate (no early stop)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument(
        "--variance",
        action="store_true",
        help="train only --configs with --seeds (no ONNX export/latency)",
    )
    p.add_argument(
        "--configs",
        default="4x48,6x32",
        help="comma-separated config names for --variance mode",
    )
    p.add_argument(
        "--seeds",
        default="11,42,99",
        help="comma-separated init seeds for --variance mode",
    )
    args = p.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"snapshot missing: {data}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "snapshot": str(data.resolve()),
        "snapshot_lines": sum(1 for _ in data.open(encoding="utf-8")),
        "epochs": args.epochs,
        "lr": args.lr,
        "val_split": args.val_split,
        "split_seed": SPLIT_SEED,
        "note": "Frozen pre-Piece-1-cycle replay from Lightsail; do not regenerate mid-ablation.",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.variance:
        config_names = [s.strip() for s in args.configs.split(",") if s.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        rows = run_seed_variance(
            data, out_root, config_names, seeds,
            epochs=args.epochs, lr=args.lr, val_split=args.val_split,
        )
        summary = summarize_variance(rows)
        payload = {"runs": rows, "summary": summary}
        out_path = out_root / "seed-variance.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("\n=== seed variance summary ===")
        for s in summary:
            spread = s["max"] - s["min"]
            print(
                f"{s['name']}: mean={s['mean']:.4f} std={s['stddev']:.4f} "
                f"min={s['min']:.4f} max={s['max']:.4f} spread={spread:.4f} vals={s['vals']}"
            )
        print(f"\nresults: {out_path}")
        return

    results = run_ablation(data, out_root, epochs=args.epochs, lr=args.lr, val_split=args.val_split)
    payload = [asdict(r) for r in results]
    (out_root / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== ablation summary ===")
    print(f"{'config':<16} {'params':>10} {'best_val':>10} {'onnx_MB':>8} {'lat_ms':>8}")
    for r in results:
        print(
            f"{r.name:<16} {r.params:>10,} {r.best_val_loss:>10.4f} "
            f"{r.onnx_bytes / 1_048_576:>8.2f} {r.latency_ms_median:>8.2f}"
        )
    print(f"\nresults: {out_root / 'results.json'}")


if __name__ == "__main__":
    main()
