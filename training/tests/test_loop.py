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
