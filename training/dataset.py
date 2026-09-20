"""Legacy row-level JSONL dataset (net_size_ablation.py only; the trainer uses gofer_train.shards)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

BOARD_SIZE = 9
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1
PLANES = 8
SPATIAL_SIZE = PLANES * BOARD_SIZE * BOARD_SIZE
GLOBALS = 4


def iter_samples(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "header":
                if row.get("schema_version", 1) != 1:
                    raise ValueError(f"unsupported schema {row.get('schema_version')}")
                continue
            yield row


def check_ownership(rows: list[dict], path: Path) -> None:
    """Every row must carry full ownership labels; never substitute zeros.

    The ownership head is trained with an unmasked loss, so a zero-filled row is
    indistinguishable from a genuinely neutral board and quietly teaches
    "neutral everywhere". A wrong length is worse still: it is normally a
    board-size mismatch, and filling it would disguise that as a
    training-quality problem. Checked once at load so the failure names a row,
    instead of surfacing thousands of batches into an epoch.
    """
    points = BOARD_SIZE * BOARD_SIZE
    for i, row in enumerate(rows):
        own = row.get("ownership")
        if not own:
            raise ValueError(f"{path}: row {i} has no ownership labels; every row needs {points}")
        if len(own) != points:
            raise ValueError(
                f"{path}: row {i} has {len(own)} ownership labels, want {points} (board-size mismatch?)"
            )


class SampleDataset(Dataset):
    """Self-play samples with exported features and board-indexed policy."""

    def __init__(self, path: Path) -> None:
        self.rows = [
            r for r in iter_samples(path)
            if len(r.get("policy", [])) == POLICY_SIZE and r.get("features_spatial")
        ]
        if not self.rows:
            raise ValueError(f"no valid samples in {path}")
        check_ownership(self.rows, path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        spatial = torch.tensor(row["features_spatial"], dtype=torch.float32).reshape(PLANES, BOARD_SIZE, BOARD_SIZE)
        globals_ = torch.tensor(row["features_global"], dtype=torch.float32)
        policy = torch.tensor(row["policy"], dtype=torch.float32)
        value = torch.tensor(float(row.get("value", 0.0)), dtype=torch.float32)
        ownership = torch.tensor(row["ownership"], dtype=torch.float32)
        return spatial, globals_, policy, value, ownership
