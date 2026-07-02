"""Load self-play JSONL samples for bootstrap training."""

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


class SampleDataset(Dataset):
    """Self-play samples with exported features and board-indexed policy."""

    def __init__(self, path: Path) -> None:
        self.rows = [
            r for r in iter_samples(path)
            if len(r.get("policy", [])) == POLICY_SIZE and r.get("features_spatial")
        ]
        if not self.rows:
            raise ValueError(f"no valid samples in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        spatial = torch.tensor(row["features_spatial"], dtype=torch.float32).reshape(PLANES, BOARD_SIZE, BOARD_SIZE)
        globals_ = torch.tensor(row["features_global"], dtype=torch.float32)
        policy = torch.tensor(row["policy"], dtype=torch.float32)
        value = torch.tensor(float(row.get("value", 0.0)), dtype=torch.float32)
        own = row.get("ownership") or [0.0] * (BOARD_SIZE * BOARD_SIZE)
        if len(own) != BOARD_SIZE * BOARD_SIZE:
            own = [0.0] * (BOARD_SIZE * BOARD_SIZE)
        ownership = torch.tensor(own, dtype=torch.float32)
        return spatial, globals_, policy, value, ownership
