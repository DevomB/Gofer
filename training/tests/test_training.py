"""Tests for the training package. Run from the repo root: python -m pytest training"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from training import elo
from training.data import Sampler, load_shard, load_window, symmetrize
from training.export import export
from training.features import NUM_GLOBALS, NUM_PLANES
from training.model import GoferNet, ModelConfig
from training.train import TrainConfig, evaluate, make_optimizer, train_steps

FIXTURE = Path(__file__).parent / "fixtures" / "shard.npz"


def test_shard_written_by_go_engine_loads():
    rows = load_shard(FIXTURE)
    assert rows.spatial.dtype == np.uint8 and rows.spatial.shape[1:] == (NUM_PLANES, 9, 9)
    assert rows.glob.shape[1] == NUM_GLOBALS
    np.testing.assert_allclose(rows.policy.sum(1), 1, atol=1e-5)
    assert set(np.unique(rows.value)) <= {-1.0, 0.0, 1.0}
    # on-board plane is all ones
    assert (rows.spatial[:, 2] == 1).all()


@pytest.mark.parametrize("sym", range(8))
def test_symmetry_moves_stones_and_policy_together(sym):
    s = 9
    spatial = torch.zeros(1, NUM_PLANES, s, s)
    policy = torch.zeros(1, s * s + 1)
    own = torch.zeros(1, s * s)
    y, x = 1, 6
    spatial[0, 0, y, x] = 1
    policy[0, y * s + x] = 1
    own[0, y * s + x] = 1
    sp, pol, ow = symmetrize(spatial, policy, own, sym)
    stone = int(sp[0, 0].flatten().argmax())
    assert int(pol[0, :-1].argmax()) == stone == int(ow[0].argmax())
    assert pol.sum() == 1


def test_symmetries_are_distinct():
    s = 9
    spatial = torch.arange(s * s, dtype=torch.float32).reshape(1, 1, s, s).repeat(1, NUM_PLANES, 1, 1)
    seen = set()
    for sym in range(8):
        sp, _, _ = symmetrize(spatial, torch.zeros(1, s * s + 1), torch.zeros(1, s * s), sym)
        seen.add(tuple(sp[0, 0].flatten().tolist()))
    assert len(seen) == 8


def test_model_shapes_and_training_reduces_loss():
    torch.manual_seed(0)
    rows = load_window([FIXTURE], 10_000)
    model = GoferNet(ModelConfig(blocks=2, channels=16, head_channels=8))
    cfg = TrainConfig(batch_size=32, lr=3e-3, warmup_steps=1)
    opt = make_optimizer(model, cfg)
    before = evaluate(model, rows)
    train_steps(model, opt, rows, 150, cfg, cfg.lr, 0, seed=0, log=lambda *_: None)
    after = evaluate(model, rows)
    assert after["policy"] < before["policy"] and after["value"] < before["value"]
    p, v, o = model(torch.zeros(3, NUM_PLANES, 9, 9), torch.zeros(3, NUM_GLOBALS))
    assert p.shape == (3, 82) and v.shape == (3,) and o.shape == (3, 81)


def test_export_matches_pytorch(tmp_path):
    model = GoferNet(ModelConfig(blocks=2, channels=16, head_channels=8))
    # Non-trivial BatchNorm statistics catch exports traced in training mode.
    model.train()
    with torch.no_grad():
        model(torch.rand(64, NUM_PLANES, 9, 9) * 3, torch.rand(64, NUM_GLOBALS))
    export(model, tmp_path / "m.onnx")  # raises if ONNX and PyTorch disagree
    assert not model.training


def test_sampler_batches():
    rows = load_shard(FIXTURE)
    batch = Sampler(rows, 16, q_weight=0.5, seed=1).batch(torch.device("cpu"))
    spatial, glob, policy, value, own, own_w = batch
    assert spatial.shape == (16, NUM_PLANES, 9, 9) and policy.shape == (16, 82)
    torch.testing.assert_close(policy.sum(1), torch.ones(16))


def test_elo_fit_recovers_ordering():
    results = [
        {"a": "b", "b": "a", "wins": 30, "losses": 10, "draws": 0},
        {"a": "c", "b": "b", "wins": 30, "losses": 10, "draws": 0},
    ]
    r = elo.fit(results, anchor="a")
    assert r["a"]["elo"] == 0
    assert 0 < r["b"]["elo"] < r["c"]["elo"]
    # 75% score is about +190 Elo
    assert 140 < r["b"]["elo"] < 230
