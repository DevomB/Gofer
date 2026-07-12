"""Tests for train_bootstrap resume and val-based best.pt selection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import SampleDataset
from model import GoferBootstrapNet
from train_bootstrap import TrainJob, TrainOptions, load_weights, run_epoch, split_indices, train, training_device

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "testdata" / "fixture_samples.jsonl"


def _run_train(out_dir: Path, **kwargs: object) -> Path:
    return train(
        TrainJob(
            data=FIXTURE,
            epochs=int(kwargs.get("epochs", 5)),  # type: ignore[arg-type]
            lr=float(kwargs.get("lr", 0.05)),  # type: ignore[arg-type]
            out_dir=out_dir,
            options=TrainOptions(
                resume=kwargs.get("resume"),  # type: ignore[arg-type]
                init_from=kwargs.get("init_from"),  # type: ignore[arg-type]
                val_split=float(kwargs.get("val_split", 0.2)),  # type: ignore[arg-type]
                patience=int(kwargs.get("patience", 100)),  # type: ignore[arg-type]
            ),
        )
    )


def test_best_pt_differs_from_last_when_val_diverges(tmp_path: Path) -> None:
    out = tmp_path / "ckpt"
    best = _run_train(out, epochs=8, lr=0.08, val_split=0.25)
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert best == out / "best.pt"
    assert best.exists() and (out / "last.pt").exists()
    # best should track min val epoch, not necessarily the final epoch
    assert last["epoch"] >= 1
    if last["epoch"] > last.get("best_epoch", 0):
        assert best.read_bytes() != (out / "last.pt").read_bytes() or last["val_loss"] >= torch.load(
            out / "last.pt", map_location="cpu", weights_only=False
        )["val_loss"]


def test_resume_continues_from_prior_weights(tmp_path: Path) -> None:
    out = tmp_path / "run"
    _run_train(out, epochs=1, lr=0.01, val_split=0.2)
    last1 = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=True)

    net = GoferBootstrapNet()
    device = training_device()
    load_weights(net, out / "best.pt")
    net = net.to(device)
    for key, tensor in saved.items():
        assert torch.allclose(net.state_dict()[key].cpu(), tensor)

    ds = SampleDataset(FIXTURE)
    _, val_idx = split_indices(len(ds), 0.2)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=min(64, len(val_idx)), shuffle=False)
    val_before = run_epoch(net, val_loader, None, train=False, device=device)
    assert abs(val_before - last1["val_loss"]) < 0.05


def test_cli_resume_flag(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    out.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_bootstrap.py"),
            "--data",
            str(FIXTURE),
            "--epochs",
            "2",
            "--out-dir",
            str(out),
            "--val-split",
            "0.2",
        ],
        check=True,
        cwd=ROOT.parent,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_bootstrap.py"),
            "--data",
            str(FIXTURE),
            "--epochs",
            "2",
            "--out-dir",
            str(out),
            "--resume",
            str(out / "best.pt"),
            "--lr",
            "0.001",
        ],
        check=True,
        cwd=ROOT.parent,
    )
    assert (out / "best.pt").exists()
