"""train_bootstrap CLI / legacy API: best.pt selection, warm-start, crash-resume."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from gofer_train.trainer import TrainConfig, Trainer
from model import GoferBootstrapNet
from train_bootstrap import TrainJob, TrainOptions, split_indices, train

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "testdata" / "fixture_samples.jsonl"
SHARD = ROOT / "testdata" / "sample-shard.npz"


def _run_train(out_dir: Path, **kw: object) -> Path:
    return train(
        TrainJob(
            data=FIXTURE,
            epochs=int(kw.get("epochs", 3)),  # type: ignore[arg-type]
            lr=float(kw.get("lr", 0.05)),  # type: ignore[arg-type]
            out_dir=out_dir,
            options=TrainOptions(
                resume=kw.get("resume"),  # type: ignore[arg-type]
                val_split=0.25,
                patience=100,
            ),
        ),
        batch_size=16,
        quiet=True,
    )


def test_outputs_and_legacy_keys(tmp_path: Path) -> None:
    out = tmp_path / "ckpt"
    best = _run_train(out)
    assert best == out / "best.pt"
    # best.pt stays a plain state_dict loadable by the legacy class.
    net = GoferBootstrapNet()
    net.load_state_dict(torch.load(best, map_location="cpu", weights_only=True))
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    for key in ("state_dict", "epoch", "train_loss", "val_loss", "optimizer", "arch", "step"):
        assert key in last
    summary = json.loads((out / "summary.json").read_text())
    assert summary["arch"]["kind"] == "legacy"
    assert summary["best_val_loss"] <= summary["final_val_loss"] + 1e-9
    lines = (out / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3  # one eval per epoch


def test_resume_warm_starts_from_weights(tmp_path: Path) -> None:
    out1, out2 = tmp_path / "a", tmp_path / "b"
    best1 = _run_train(out1, epochs=1)
    # One step at lr ~0 must land (almost) exactly on the warm-start weights.
    cfg = TrainConfig(data=[str(FIXTURE)], out_dir=str(out2), init_from=str(best1), steps=1,
                      lr=1e-12, ema_decay=0, batch_size=16, quiet=True)
    t = Trainer(cfg)
    before = {k: v.clone() for k, v in t.net.state_dict().items()}
    src = torch.load(best1, map_location="cpu", weights_only=True)
    for k in src:
        assert torch.allclose(before[k].cpu(), src[k]), k


def test_continue_interrupted_run(tmp_path: Path) -> None:
    out = tmp_path / "run"
    base = dict(data=[str(SHARD)], out_dir=str(out), arch="gpool-tiny", batch_size=32,
                steps=12, eval_every=4, patience=0, quiet=True)
    Trainer(TrainConfig(**base)).run()
    # A preempted/finished run continues from last.pt with a longer schedule.
    st = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert st["step"] == 12
    t = Trainer(TrainConfig(**{**base, "steps": 20}, resume_run=True))
    assert t.step == 12
    summary = t.run()
    assert summary["steps"] == 20


def test_cli_resume_flag(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    common = [sys.executable, str(ROOT / "train_bootstrap.py"), "--data", str(FIXTURE),
              "--epochs", "2", "--out-dir", str(out), "--val-split", "0.2", "--quiet"]
    subprocess.run(common, check=True, cwd=ROOT.parent)
    subprocess.run(common + ["--resume", str(out / "best.pt"), "--lr", "0.001"], check=True, cwd=ROOT.parent)
    assert (out / "best.pt").exists()


def test_cli_shard_dir_window(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "a.npz").write_bytes(SHARD.read_bytes())
    out = tmp_path / "o"
    subprocess.run(
        [sys.executable, str(ROOT / "train_bootstrap.py"), "--data", str(shard_dir), "--steps", "4",
         "--arch", "gpool-tiny", "--window-rows", "200", "--window-decay", "2", "--batch-size", "32",
         "--out-dir", str(out), "--quiet"],
        check=True, cwd=ROOT.parent,
    )
    s = json.loads((out / "summary.json").read_text())
    assert s["train_rows"] + s["val_rows"] == 200


def test_split_indices_legacy_is_stable() -> None:
    a = split_indices(100, 0.1)
    assert a == split_indices(100, 0.1) and len(a[1]) == 10
