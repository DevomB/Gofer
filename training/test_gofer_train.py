"""Unit tests for gofer_train: shards, symmetry, splits, losses, nets, distillation, export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from export_onnx import export_onnx
from gofer_train.data import DeviceData, Sampler, apply_symmetry, split_by_game
from gofer_train.losses import LossWeights, compute_loss
from gofer_train.nets import PRESETS, build_net, infer_arch, load_weights, resolve_arch
from gofer_train.shards import load_rows, read_shard, rows_from_jsonl, write_shard
from gofer_train.trainer import TrainConfig, Trainer

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "testdata" / "fixture_samples.jsonl"
SHARD = ROOT / "testdata" / "sample-shard.npz"
CPU = torch.device("cpu")


# ---------------------------------------------------------------- shards ----


def test_go_shard_score_matches_ownership() -> None:
    rows = read_shard(SHARD)
    black = rows.globals[:, 2] == 1
    own = rows.ownership.sum(1).astype(np.float64)
    komi = 6.5
    assert np.allclose(np.where(black, own - komi, own + komi), rows.score)
    assert set(np.unique(rows.full_search)) == {0, 1}


def test_shard_roundtrip(tmp_path: Path) -> None:
    rows = read_shard(SHARD)
    p = write_shard(tmp_path / "x.npz", rows, model="test")
    back = read_shard(p)
    for name in ("spatial", "policy", "value", "ownership", "full_search", "move_num"):
        assert np.array_equal(getattr(rows, name), getattr(back, name)), name


def test_jsonl_to_rows_side_to_move_ownership() -> None:
    rows = rows_from_jsonl(FIXTURE)
    raw = [json.loads(line) for line in FIXTURE.read_text().splitlines()[1:] if line.strip()]
    for i, r in enumerate(raw[: len(rows)]):
        sign = -1 if r["to_play"] == 2 else 1
        assert np.array_equal(rows.ownership[i], np.sign(np.asarray(r["ownership"]) * sign))
    # game ids increase whenever move_num resets
    assert rows.game_id[0] == 0 and np.all(np.diff(rows.game_id) >= 0)


def test_window_keeps_newest_rows(tmp_path: Path) -> None:
    rows = read_shard(SHARD)
    write_shard(tmp_path / "old.npz", rows.take(slice(0, 100)))
    write_shard(tmp_path / "new.npz", rows.take(slice(100, None)))
    import os
    os.utime(tmp_path / "old.npz", (1, 1))
    got = load_rows(tmp_path, window_rows=50)
    assert len(got) == 50
    assert np.array_equal(got.spatial, rows.spatial[-50:])


# -------------------------------------------------------------- symmetry ----


def test_symmetry_is_consistent_between_board_and_policy() -> None:
    """Moving a stone and the policy mass with it must stay aligned under all 8 syms."""
    s = 9
    rows = read_shard(SHARD).take(slice(0, 8))
    rows.spatial[:] = 0
    rows.policy[:] = 0
    rows.ownership[:] = 0
    for i in range(8):
        rows.spatial[i, 0, 1, 2] = 1  # own stone at (row 1, col 2)
        rows.policy[i, 1 * s + 2] = 1.0
        rows.ownership[i, 1 * s + 2] = 1
    data = DeviceData(rows, CPU)
    seen = set()
    for _ in range(30):
        b = data.gather(torch.arange(8), augment=True)
        for i in range(8):
            stone = int(b.spatial[i, 0].flatten().argmax())
            assert int(b.policy[i, : s * s].argmax()) == stone
            assert int(b.ownership[i].argmax()) == stone
            seen.add(stone)
    assert len(seen) == 8  # all 8 images of an asymmetric point are reached


def test_symmetry_group_identity() -> None:
    x = torch.randn(8, 3, 9, 9)
    assert torch.equal(apply_symmetry(x, torch.zeros(8, dtype=torch.long)), x)


# ------------------------------------------------------------ split/sample ----


def test_split_by_game_has_no_leak() -> None:
    rows = read_shard(SHARD)
    tr, va = split_by_game(rows, 0.34)
    assert len(tr) and len(va)
    assert not set(rows.game_id[tr]) & set(rows.game_id[va])


def test_recency_sampler_prefers_new_rows() -> None:
    s = Sampler(np.arange(1000), 100, window_decay=3.0, seed=0)
    picks = torch.cat(list(s.epoch()))
    assert (picks >= 500).float().mean() > 0.7


# ------------------------------------------------------------------ loss ----


def test_policy_loss_ignores_fast_rows() -> None:
    rows = read_shard(SHARD)
    data = DeviceData(rows, CPU)
    net = build_net(resolve_arch("gpool-tiny")).eval()
    fast = torch.from_numpy(np.flatnonzero(rows.full_search == 0)[:16])
    b = data.gather(fast, augment=False)
    out = net.forward_all(b.spatial, b.globals)
    _, parts = compute_loss(out, b, LossWeights())
    assert float(parts["policy"]) == 0.0
    assert float(parts["value"]) > 0.0


# ------------------------------------------------------------------ nets ----


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_arch_roundtrip_and_infer(name: str) -> None:
    arch = resolve_arch(name)
    net = build_net(arch)
    assert infer_arch(net.state_dict()) == arch
    out = net.eval().forward_all(torch.zeros(2, 8, 9, 9), torch.zeros(2, 4))
    assert out.policy_logits.shape == (2, 82) and out.value.shape == (2,)


def test_gpool_is_board_size_agnostic() -> None:
    net = build_net(resolve_arch("gpool-tiny")).eval()
    for s in (9, 13, 19):
        out = net.forward_all(torch.zeros(1, 8, s, s), torch.zeros(1, 4))
        assert out.policy_logits.shape == (1, s * s + 1)
        assert out.ownership.shape == (1, s * s)


def test_old_checkpoint_without_ownership_head_loads() -> None:
    net = build_net(resolve_arch("legacy-4x48"))
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("ownership_conv")}
    assert load_weights(build_net(resolve_arch("legacy-4x48")), state) == [
        "ownership_conv.weight", "ownership_conv.bias"
    ]
    with pytest.raises(RuntimeError):
        load_weights(build_net(resolve_arch("legacy-4x48")), {k: v for k, v in state.items() if "stem" not in k})


# --------------------------------------------------------- distill/export ----


def test_distill_new_arch_from_legacy_teacher(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.pt"
    torch.save(build_net(resolve_arch("legacy-4x48")).state_dict(), teacher)
    cfg = TrainConfig(data=[str(SHARD)], out_dir=str(tmp_path / "s"), arch="gpool-tiny",
                      teacher=str(teacher), steps=4, batch_size=32, quiet=True)
    summary = Trainer(cfg).run()
    assert "distill" in summary["val"] or summary["steps"] == 4
    assert (tmp_path / "s" / "best.ckpt").exists()


def test_export_gpool_from_trainer_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "run"
    Trainer(TrainConfig(data=[str(SHARD)], out_dir=str(out), arch="gpool-tiny",
                        steps=3, batch_size=32, quiet=True)).run()
    onnx_path = tmp_path / "m.onnx"
    info = export_onnx(onnx_path, out / "best.pt")  # arch comes from best.ckpt sidecar
    assert info["arch"] == "gpool-2x16"
    assert info["parity_max_abs_diff"] < 1e-3
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["policy_logits", "value"]
    meta = sess.get_modelmeta().custom_metadata_map
    assert json.loads(meta["gofer.arch"])["kind"] == "gpool"
    logits, value = sess.run(None, {"spatial_input": np.zeros((5, 8, 9, 9), np.float32),
                                    "global_input": np.zeros((5, 4), np.float32)})
    assert logits.shape == (5, 82) and value.shape == (5,)


def test_jsonl_rejects_rows_without_ownership(tmp_path: Path) -> None:
    """The ownership loss is unmasked, so an unlabelled row is not a neutral
    board - it teaches the head that every point is neutral. WriteSampleShard
    rejects this on the Go side; the JSONL path must agree."""
    lines = FIXTURE.read_text().splitlines()
    header, first = lines[0], json.loads(lines[1])
    del first["ownership"]
    bad = tmp_path / "no-ownership.jsonl"
    bad.write_text("\n".join([header, json.dumps(first), *lines[2:]]))

    with pytest.raises(ValueError, match="ownership labels"):
        rows_from_jsonl(bad)
