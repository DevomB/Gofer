"""Device-resident training data: game-level split, recency sampling, D4 symmetry.

The whole replay window lives on the training device as compact tensors
(spatial planes as uint8: 50k rows of 9x9 is ~32 MB). Batches are gathered by
index on-device and augmented with a random dihedral symmetry per sample, so
there is no DataLoader, no per-row Python tensor construction and no
host->device copy in the hot loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .shards import Rows


@dataclass
class Batch:
    spatial: torch.Tensor  # float [B, 8, S, S]
    globals: torch.Tensor  # float [B, 4]
    policy: torch.Tensor  # float [B, P]
    policy_opp: torch.Tensor  # float [B, P]
    value: torch.Tensor  # float [B]
    score: torch.Tensor  # float [B] (NaN = unknown)
    ownership: torch.Tensor  # float [B, S*S]
    full_search: torch.Tensor  # bool [B]

    def __len__(self) -> int:
        return int(self.value.shape[0])


def split_by_game(rows: Rows, val_frac: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Train/val row indices with whole games on one side.

    Row-level splits leak: consecutive positions of one game are near-duplicates,
    so val loss would track memorization rather than generalization.
    """
    n = len(rows)
    if n < 2 or val_frac <= 0:
        return np.arange(n), np.arange(0)
    games = np.unique(rows.game_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    if len(games) >= 2:
        n_val = min(max(1, int(round(len(games) * val_frac))), len(games) - 1)
        val_games = games[:n_val]
        is_val = np.isin(rows.game_id, val_games)
    else:  # a single game: fall back to a row split
        is_val = np.zeros(n, bool)
        perm = rng.permutation(n)
        is_val[perm[: min(max(1, int(round(n * val_frac))), n - 1)]] = True
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


class DeviceData:
    """Replay rows as tensors on ``device`` plus batched, augmented gathers."""

    def __init__(self, rows: Rows, device: torch.device) -> None:
        self.n = len(rows)
        self.size = rows.board_size
        self.device = device

        def t(a: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)

        pol = rows.policy.astype(np.float32)
        pol_sum = pol.sum(1, keepdims=True)
        pol = np.divide(pol, pol_sum, out=np.zeros_like(pol), where=pol_sum > 0)
        opp = rows.policy_opp.astype(np.float32)
        opp_sum = opp.sum(1, keepdims=True)
        opp = np.divide(opp, opp_sum, out=np.zeros_like(opp), where=opp_sum > 0)

        self.spatial = t(rows.spatial, torch.uint8)
        self.globals = t(rows.globals, torch.float32)
        self.policy = t(pol, torch.float32)
        self.policy_opp = t(opp, torch.float32)
        self.value = t(rows.value, torch.float32)
        self.score = t(rows.score, torch.float32)
        self.ownership = t(rows.ownership, torch.int8)
        self.full_search = t(rows.full_search, torch.bool)

    def gather(self, idx: torch.Tensor, *, augment: bool, generator: torch.Generator | None = None) -> Batch:
        idx = idx.to(self.device)
        s = self.size
        spatial = self.spatial[idx].float()
        policy = self.policy[idx]
        opp = self.policy_opp[idx]
        own = self.ownership[idx].float()
        if augment:
            sym = torch.randint(0, 8, (len(idx),), generator=generator, device="cpu").to(self.device)
            spatial = apply_symmetry(spatial, sym)
            policy = _sym_policy(policy, sym, s)
            opp = _sym_policy(opp, sym, s)
            own = apply_symmetry(own.reshape(-1, 1, s, s), sym).reshape(-1, s * s)
        return Batch(
            spatial=spatial,
            globals=self.globals[idx],
            policy=policy,
            policy_opp=opp,
            value=self.value[idx],
            score=self.score[idx],
            ownership=own,
            full_search=self.full_search[idx],
        )


def _d4(x: torch.Tensor, k: int) -> torch.Tensor:
    """One of the 8 dihedral transforms on the last two dims."""
    if k >= 4:
        x = x.transpose(-1, -2)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def apply_symmetry(x: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """Per-sample D4 transform of ``x`` [B, C, S, S] by ``sym`` [B] in 0..7."""
    out = torch.empty_like(x)
    for k in range(8):
        m = sym == k
        if bool(m.any()):
            out[m] = _d4(x[m], k)
    return out


def _sym_policy(p: torch.Tensor, sym: torch.Tensor, s: int) -> torch.Tensor:
    board = apply_symmetry(p[:, : s * s].reshape(-1, 1, s, s), sym).reshape(-1, s * s)
    return torch.cat([board, p[:, s * s :]], dim=1)


class Sampler:
    """Yields batches of train indices; uniform epochs or recency-weighted draws.

    ``window_decay`` > 0 weights row i (0 = oldest) by exp(decay * (i/n - 1)):
    the newest data is sampled e^decay times more often than the oldest, the
    usual trick for keeping a large replay window without drowning fresh games.
    """

    def __init__(
        self,
        train_idx: np.ndarray,
        batch_size: int,
        *,
        window_decay: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.idx = torch.from_numpy(np.asarray(train_idx, np.int64))
        self.batch_size = max(1, min(batch_size, len(self.idx)))
        self.gen = torch.Generator().manual_seed(seed)
        self.weights: torch.Tensor | None = None
        if window_decay > 0 and len(self.idx) > 1:
            pos = self.idx.double() / max(1, int(self.idx.max()))
            self.weights = torch.exp(window_decay * (pos - 1.0)).float()

    @property
    def steps_per_epoch(self) -> int:
        return max(1, (len(self.idx) + self.batch_size - 1) // self.batch_size)

    def epoch(self):
        if self.weights is not None:
            for _ in range(self.steps_per_epoch):
                pick = torch.multinomial(self.weights, self.batch_size, replacement=True, generator=self.gen)
                yield self.idx[pick]
            return
        perm = self.idx[torch.randperm(len(self.idx), generator=self.gen)]
        for i in range(0, len(perm), self.batch_size):
            chunk = perm[i : i + self.batch_size]
            if len(chunk) == self.batch_size or i == 0:
                yield chunk
