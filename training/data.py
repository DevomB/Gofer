"""Self-play shards (.npz written by `gofer selfplay`) and augmented batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from training.features import NUM_GLOBALS, NUM_PLANES

FIELDS = ("spatial", "global", "policy", "value", "q", "ownership", "ownership_weight")


@dataclass
class Rows:
    spatial: np.ndarray  # uint8 [N, C, S, S]
    glob: np.ndarray  # float32 [N, G]
    policy: np.ndarray  # float32 [N, S*S+1]
    value: np.ndarray  # float32 [N] game result for the side to move
    q: np.ndarray  # float32 [N] root search value
    ownership: np.ndarray  # int8 [N, S*S]
    ownership_weight: np.ndarray  # float32 [N]

    def __len__(self) -> int:
        return len(self.value)

    @property
    def size(self) -> int:
        return self.spatial.shape[-1]


def load_shard(path: Path) -> Rows:
    with np.load(path) as d:
        rows = Rows(
            spatial=d["spatial"],
            glob=d["global"],
            policy=d["policy"],
            value=d["value"],
            q=d["q"],
            ownership=d["ownership"],
            ownership_weight=d["ownership_weight"],
        )
    if rows.spatial.shape[1] != NUM_PLANES or rows.glob.shape[1] != NUM_GLOBALS:
        raise ValueError(f"{path}: feature schema mismatch {rows.spatial.shape} {rows.glob.shape}")
    return rows


def concat(parts: list[Rows]) -> Rows:
    return Rows(*(np.concatenate([getattr(p, f) for p in parts]) for f in Rows.__dataclass_fields__))


def load_window(shards: list[Path], max_rows: int) -> Rows:
    """Load the newest shards (by order given, oldest first) up to max_rows."""
    parts: list[Rows] = []
    total = 0
    for path in reversed(shards):
        r = load_shard(path)
        parts.append(r)
        total += len(r)
        if total >= max_rows:
            break
    rows = concat(list(reversed(parts)))
    if len(rows) > max_rows:
        rows = Rows(*(getattr(rows, f)[-max_rows:] for f in Rows.__dataclass_fields__))
    return rows


def symmetrize(spatial: torch.Tensor, policy: torch.Tensor, own: torch.Tensor, sym: int):
    """Apply dihedral symmetry `sym` (bit 4: transpose, 1: flip x, 2: flip y)
    to a batch, matching internal/nn/features.go."""
    s = spatial.shape[-1]
    board = policy[:, :-1].reshape(-1, s, s)
    own = own.reshape(-1, s, s)
    tensors = [spatial, board[:, None], own[:, None]]
    out = []
    for t in tensors:
        if sym & 4:
            t = t.transpose(-1, -2)
        if sym & 1:
            t = t.flip(-1)
        if sym & 2:
            t = t.flip(-2)
        out.append(t)
    spatial, board, own = out
    policy = torch.cat([board.reshape(-1, s * s), policy[:, -1:]], dim=1)
    return spatial, policy, own.reshape(-1, s * s)


class Sampler:
    """Uniform random minibatches from a window, each with a random symmetry."""

    def __init__(self, rows: Rows, batch_size: int, q_weight: float, seed: int) -> None:
        self.rows = rows
        self.batch_size = batch_size
        self.q_weight = q_weight
        self.rng = np.random.default_rng(seed)

    def batch(self, device: torch.device):
        r = self.rows
        idx = self.rng.integers(0, len(r), self.batch_size)
        spatial = torch.from_numpy(r.spatial[idx].astype(np.float32))
        glob = torch.from_numpy(r.glob[idx])
        policy = torch.from_numpy(r.policy[idx])
        value = torch.from_numpy((1 - self.q_weight) * r.value[idx] + self.q_weight * r.q[idx])
        own = torch.from_numpy(r.ownership[idx].astype(np.float32))
        own_w = torch.from_numpy(r.ownership_weight[idx])
        syms = self.rng.integers(0, 8, self.batch_size)
        sp_out, pol_out, own_out = torch.empty_like(spatial), torch.empty_like(policy), torch.empty_like(own)
        for s in range(8):
            m = torch.from_numpy(np.nonzero(syms == s)[0])
            if len(m) == 0:
                continue
            a, b, c = symmetrize(spatial[m], policy[m], own[m], s)
            sp_out[m], pol_out[m], own_out[m] = a, b, c
        return tuple(t.to(device, non_blocking=True) for t in (sp_out, glob, pol_out, value, own_out, own_w))
