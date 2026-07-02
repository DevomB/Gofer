"""Train bootstrap 9x9 net from self-play JSONL."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import SampleDataset
from model import GoferBootstrapNet

SPLIT_SEED = 42


def split_indices(n: int, val_split: float, seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    if n < 2:
        return idx, []
    n_val = max(1, int(round(n * val_split)))
    n_val = min(n_val, n - 1)
    return idx[n_val:], idx[:n_val]


def load_weights(net: GoferBootstrapNet, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)


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
    for spatial, globals_, policy, value in loader:
        if train and opt is not None:
            opt.zero_grad()
        logits, pred_v = net(spatial, globals_)
        target = policy / policy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        loss_p = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss_v = F.mse_loss(pred_v, value)
        loss = loss_p + loss_v
        if train and opt is not None:
            loss.backward()
            opt.step()
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


def train(
    data: Path,
    epochs: int,
    lr: float,
    out_dir: Path,
    *,
    resume: Path | None = None,
    init_from: Path | None = None,
    val_split: float = 0.1,
    patience: int = 5,
) -> Path:
    ds = SampleDataset(data)
    train_idx, val_idx = split_indices(len(ds), val_split)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=min(64, len(train_idx)), shuffle=True)
    val_loader = (
        DataLoader(Subset(ds, val_idx), batch_size=min(64, len(val_idx)), shuffle=False)
        if val_idx
        else None
    )

    net = GoferBootstrapNet()
    if resume and resume.exists():
        load_weights(net, resume)
    elif init_from and init_from.exists():
        load_weights(net, init_from)

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    best_val = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(epochs):
        train_loss = run_epoch(net, train_loader, opt, train=True)
        val_loss = run_epoch(net, val_loader, None, train=False) if val_loader else train_loss
        print(
            f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        torch.save(
            {
                "state_dict": net.state_dict(),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            last_path,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            stale = 0
            torch.save(net.state_dict(), best_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch + 1} (patience={patience})")
                break

    if not best_path.exists():
        torch.save(net.state_dict(), best_path)

    print(f"best.pt epoch={best_epoch} val_loss={best_val:.4f}")
    return best_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="training/data/samples.jsonl")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out-dir", default="training/checkpoints")
    p.add_argument("--resume", default="", help="load weights from checkpoint if exists")
    p.add_argument("--init-from", default="", help="one-time seed weights (e.g. cycle2)")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()

    resume = Path(args.resume) if args.resume else None
    init_from = Path(args.init_from) if args.init_from else None
    best = train(
        Path(args.data),
        args.epochs,
        args.lr,
        Path(args.out_dir),
        resume=resume,
        init_from=init_from,
        val_split=args.val_split,
        patience=args.patience,
    )
    print(f"checkpoint: {best}")


if __name__ == "__main__":
    main()
