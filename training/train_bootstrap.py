"""Train bootstrap 9x9 net from self-play JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SampleDataset
from model import GoferBootstrapNet


def train(data: Path, epochs: int, lr: float, out_dir: Path) -> Path:
    ds = SampleDataset(data)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    net = GoferBootstrapNet()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = out_dir / "best.pt"

    for epoch in range(epochs):
        total = 0.0
        for spatial, globals_, policy, value in loader:
            opt.zero_grad()
            logits, pred_v = net(spatial, globals_)
            target = policy / policy.sum(dim=1, keepdim=True).clamp(min=1e-8)
            loss_p = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_v = F.mse_loss(pred_v, value)
            loss = loss_p + loss_v
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch+1}/{epochs} loss={total/len(loader):.4f}")
        torch.save(net.state_dict(), best)
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="training/data/samples.jsonl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out-dir", default="training/checkpoints")
    args = p.parse_args()
    best = train(Path(args.data), args.epochs, args.lr, Path(args.out_dir))
    print(f"checkpoint: {best}")


if __name__ == "__main__":
    main()
