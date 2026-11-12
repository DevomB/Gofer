"""Tests for the training loop's bookkeeping (no gofer binary needed)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from training.features import NUM_GLOBALS, NUM_PLANES
from training import loop
from training.loop import LoopConfig, Run, anchor_problems, match_seed


@pytest.fixture
def run(tmp_path, monkeypatch):
    r = Run(tmp_path / "run", LoopConfig(blocks=1, channels=8, anchors=[]), Path("gofer"), threads=1)
    monkeypatch.setattr(r, "log", lambda msg: None)
    return r


def executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_every_match_gets_its_own_openings(run, monkeypatch):
    # These tags have the same length. The seed used to be len(tag), so every
    # evaluation of a run replayed the same openings.
    tags = ["gen-0001-vs-gen-0000", "gen-0002-vs-gen-0001", "gen-0002-vs-gnugo-l1", "gen-0002-vs-gnugo-l6"]
    assert len({len(t) for t in tags}) == 1
    assert len({match_seed(t) for t in tags}) == len(tags)
    assert match_seed(tags[0]) == match_seed(tags[0])  # a resumed run replays the same match

    calls = []

    def fake_gofer(args):
        calls.append(args)
        Path(args[args.index("-json") + 1]).write_text(json.dumps({"wins": 1, "losses": 1, "draws": 0}))

    monkeypatch.setattr(run, "gofer_run", fake_gofer)
    for tag in tags[:2]:
        run.match("nn:a.onnx", "nn:b.onnx", 2, 8, tag)
    seeds = [args[args.index("-seed") + 1] for args in calls]
    assert seeds == [str(match_seed(t)) for t in tags[:2]]
    run.match("nn:a.onnx", "nn:b.onnx", 2, 8, tags[0])
    assert len(calls) == 2  # a finished match is read back, not replayed


def test_games_the_arena_could_not_play_are_reported(run, monkeypatch):
    def fake_gofer(args):
        Path(args[args.index("-json") + 1]).write_text(json.dumps({"wins": 30, "losses": 9, "draws": 0, "skipped": [17]}))

    logged = []
    monkeypatch.setattr(run, "gofer_run", fake_gofer)
    monkeypatch.setattr(run, "log", logged.append)
    r = run.match("nn:a.onnx", "gnugo,level=1", 40, 100, "gen-0004-vs-gnugo-l1")
    assert (r["wins"], r["losses"]) == (30, 9)
    assert len(logged) == 1 and "1 of the 40 games of gen-0004-vs-gnugo-l1" in logged[0]


def test_init_exports_the_network_when_only_the_checkpoint_exists(run):
    import onnxruntime as ort

    run.init()
    ckpt, model = run.ckpt_path(0), run.model_path(0)
    assert ckpt.exists() and model.exists()
    # a run stopped between saving the checkpoint and exporting it
    model.unlink()
    before = ckpt.read_bytes()
    run.init()
    assert ckpt.read_bytes() == before
    # the engine's contract: these names, any batch size
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["spatial", "global"]
    assert [o.name for o in sess.get_outputs()] == ["policy", "value"]
    for batch in (1, 7):
        policy, value = sess.run(None, {"spatial": np.zeros((batch, NUM_PLANES, 9, 9), np.float32),
                                        "global": np.zeros((batch, NUM_GLOBALS), np.float32)})
        assert policy.shape == (batch, 82) and value.shape == (batch,)


def test_anchor_problems_names_what_is_missing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    path = str(empty)
    assert anchor_problems(["rollout,visits=800", "uniform"], path) == []
    (problem,) = anchor_problems(["gnugo,level=1"], path)
    assert "gnugo,level=1" in problem and "GNU Go" in problem
    assert len(anchor_problems(["gnugo,level=1", "gnugo,level=6", "rollout"], path)) == 2
    assert len(anchor_problems(["gtp:some-engine --mode gtp"], path)) == 1
    assert len(anchor_problems([f"nn:{tmp_path}/missing.onnx,visits=100"], path)) == 1

    executable(empty / "gnugo")
    executable(empty / "some-engine")
    (tmp_path / "net.onnx").write_bytes(b"")
    assert anchor_problems(["gnugo,level=1", "gtp:some-engine --mode gtp", f"nn:{tmp_path}/net.onnx,visits=100"], path) == []
    # a GNU Go binary given explicitly
    assert anchor_problems([f"gnugo,level=1,bin={empty}/gnugo"], str(tmp_path)) == []
    assert len(anchor_problems([f"gnugo,level=1,bin={empty}/nope"], path)) == 1


def loop_main(monkeypatch, run_dir: Path, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["training.loop", "--run", str(run_dir), "--gens", "0", *args])
    loop.main()


def test_loop_refuses_to_start_when_an_anchor_cannot_play(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit) as stop:
        loop_main(monkeypatch, run_dir, "--set", 'anchors=["gnugo,level=1,bin=/nonexistent/gnugo", "rollout"]')
    message = str(stop.value)
    assert "gnugo,level=1" in message and "install gnugo" in message and "rollout:" not in message
    # refused before anything was written: a config.json left behind would
    # make the next attempt's --set a change to an existing run
    assert not run_dir.exists()


def test_set_may_repeat_a_runs_settings_but_not_change_them(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    tiny = ["--set", "blocks=1", "--set", "channels=8", "--set", "anchors=[]"]
    loop_main(monkeypatch, run_dir, *tiny)
    saved = (run_dir / "config.json").read_text()
    assert json.loads(saved)["channels"] == 8
    loop_main(monkeypatch, run_dir, *tiny)  # the same command resumes
    loop_main(monkeypatch, run_dir)  # and so does the command without the settings
    with pytest.raises(SystemExit, match="channels"):
        loop_main(monkeypatch, run_dir, "--set", "channels=16")
    assert (run_dir / "config.json").read_text() == saved
    with pytest.raises(SystemExit, match="unknown config key"):
        loop_main(monkeypatch, tmp_path / "other", "--set", "chanels=16")
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        loop_main(monkeypatch, tmp_path / "other", "--set", "channels")


def trained_checkpoint(path: Path, size: int = 9) -> Path:
    """A checkpoint whose weights differ from the random initial network's."""
    import torch

    from training.model import GoferNet, ModelConfig
    from training.train import TrainConfig, make_optimizer, save_checkpoint

    torch.manual_seed(5)
    model = GoferNet(ModelConfig(board_size=size, blocks=1, channels=8))
    save_checkpoint(path, model, make_optimizer(model, TrainConfig()), 1234)
    return path


def test_a_run_can_start_on_a_larger_board_from_another_runs_network(tmp_path, monkeypatch):
    import onnxruntime as ort
    import torch

    from training.model import GoferNet, ModelConfig

    src = trained_checkpoint(tmp_path / "small-board.pt")
    cfg = LoopConfig(board_size=13, blocks=1, channels=8, anchors=[], init_from=str(src))
    r = Run(tmp_path / "run", cfg, Path("gofer"), threads=1)
    logged = []
    monkeypatch.setattr(r, "log", logged.append)
    r.init()
    assert "started from" in logged[0] and "small-board.pt" in logged[0]

    source, start = torch.load(src, weights_only=False), torch.load(r.ckpt_path(0), weights_only=False)
    assert start["model_config"]["board_size"] == 13
    assert start["step"] == 0  # the new run has its own schedule
    assert source["model"].keys() == start["model"].keys()
    assert all(torch.equal(source["model"][k], start["model"][k].cpu()) for k in source["model"])

    # the exported network plays the new size and is the source network there
    model = GoferNet(ModelConfig(board_size=13, blocks=1, channels=8)).eval()
    model.load_state_dict(source["model"])
    rng = np.random.default_rng(1)
    spatial = (rng.random((3, NUM_PLANES, 13, 13)) < 0.3).astype(np.float32)
    glob = rng.standard_normal((3, NUM_GLOBALS)).astype(np.float32)
    sess = ort.InferenceSession(str(r.model_path(0)), providers=["CPUExecutionProvider"])
    policy, value = sess.run(None, {"spatial": spatial, "global": glob})
    with torch.no_grad():
        want_policy, want_value, _ = model(torch.from_numpy(spatial), torch.from_numpy(glob))
    assert policy.shape == (3, 13 * 13 + 1)
    assert np.allclose(policy, want_policy.numpy(), atol=1e-3) and np.allclose(value, want_value.numpy(), atol=1e-3)

    # a resumed run does not need the source checkpoint any more
    src.unlink()
    r.init()


def test_a_checkpoint_can_be_exported_for_another_board_size(tmp_path, capsys):
    import onnxruntime as ort

    from training import export

    src = trained_checkpoint(tmp_path / "big-board.pt", size=13)
    export.main([str(src), str(tmp_path / "own.onnx")])
    export.main([str(src), str(tmp_path / "small.onnx"), "--size", "9"])
    assert "(13x13)" in capsys.readouterr().out.splitlines()[0]
    for name, size in (("own", 13), ("small", 9)):
        sess = ort.InferenceSession(str(tmp_path / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        policy, _ = sess.run(None, {"spatial": np.zeros((2, NUM_PLANES, size, size), np.float32),
                                    "global": np.zeros((2, NUM_GLOBALS), np.float32)})
        assert policy.shape == (2, size * size + 1)


def test_loop_refuses_a_starting_network_it_cannot_use(tmp_path, monkeypatch, capsys):
    src = trained_checkpoint(tmp_path / "small-board.pt")
    tiny = ["--set", "anchors=[]", "--set", "board_size=13", "--set", f"init_from={src}"]
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit) as stop:
        loop_main(monkeypatch, run_dir, *tiny)  # the run defaults to 6x64
    assert "1x8" in str(stop.value) and "6x64" in str(stop.value) and "--set blocks=1 --set channels=8" in str(stop.value)
    assert not run_dir.exists()
    with pytest.raises(SystemExit, match="not a file"):
        loop_main(monkeypatch, run_dir, "--set", "anchors=[]", "--set", f"init_from={tmp_path}/missing.pt")
    assert not run_dir.exists()

    loop_main(monkeypatch, run_dir, *tiny, "--set", "blocks=1", "--set", "channels=8")
    assert json.loads((run_dir / "config.json").read_text())["init_from"] == str(src)
    assert (run_dir / "ckpt" / "gen-0000.pt").exists()
    src.unlink()
    loop_main(monkeypatch, run_dir)  # resuming works without the source checkpoint
