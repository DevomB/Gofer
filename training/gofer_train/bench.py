"""Throughput benchmark: legacy DataLoader path vs the device-resident trainer.

    python -m gofer_train.bench --data training/data/samples.jsonl --steps 60

Reports load time and training samples/sec for the same net and batch size.
Numbers are what a cycle's train step costs; run on the target box (CPU or GPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .nets import resolve_arch
from .shards import load_rows, write_shard
from .trainer import TrainConfig, Trainer, pick_device


def legacy_throughput(data: Path, steps: int, batch: int, device: torch.device) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from torch.utils.data import DataLoader

    from dataset import SampleDataset
    from model import GoferBootstrapNet

    t0 = time.time()
    ds = SampleDataset(data)
    load = time.time() - t0
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    net = GoferBootstrapNet().to(device).train()
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    done, seen = 0, 0
    t0 = time.time()
    while done < steps:
        for spatial, glob, policy, value, own in loader:
            spatial, glob, policy, value, own = (x.to(device) for x in (spatial, glob, policy, value, own))
            opt.zero_grad()
            logits, v, o = net(spatial, glob)
            loss = -(policy * torch.log_softmax(logits, 1)).sum(1).mean() + F.mse_loss(v, value) + 0.15 * F.mse_loss(o, own)
            loss.backward()
            opt.step()
            done += 1
            seen += len(value)
            if done >= steps:
                break
    return {"load_sec": round(load, 2), "samples_per_sec": round(seen / (time.time() - t0), 1)}


def new_throughput(data: list[Path], steps: int, batch: int, arch: str, extra: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        rows = load_rows(data)
        load = time.time() - t0
        cfg = TrainConfig(out_dir=tmp, arch=arch, steps=steps, batch_size=batch, eval_every=10**9,
                          patience=0, val_split=0.0, quiet=True, **extra)
        summary = Trainer(cfg, rows=rows).run()
    return {"load_sec": round(load, 2), "samples_per_sec": summary["samples_per_sec"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="training/data/samples.jsonl")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--archs", default="legacy-6x64,gpool-6x64")
    p.add_argument("--skip-legacy", action="store_true")
    args = p.parse_args()
    data = Path(args.data)
    device = pick_device()
    results: dict[str, dict] = {"device": str(device), "batch": args.batch_size}  # type: ignore[dict-item]

    if not args.skip_legacy and data.suffix == ".jsonl":
        results["legacy-dataloader (6x64)"] = legacy_throughput(data, args.steps, args.batch_size, device)

    if data.suffix == ".jsonl":
        with tempfile.TemporaryDirectory() as tmp:
            shard = write_shard(Path(tmp) / "s.npz", load_rows(data))
            t0 = time.time()
            load_rows(shard)
            results["shard_load_sec"] = round(time.time() - t0, 3)  # type: ignore[assignment]
            results["jsonl_bytes"] = data.stat().st_size  # type: ignore[assignment]
            results["shard_bytes"] = shard.stat().st_size  # type: ignore[assignment]

    for arch in args.archs.split(","):
        resolve_arch(arch)
        results[f"new ({arch}, aug+ema)"] = new_throughput([data], args.steps, args.batch_size, arch, {})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
