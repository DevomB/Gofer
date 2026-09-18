"""Publishing: every best is kept, alias/latest move forward, false bests are held."""

from __future__ import annotations

import json

import pytest

from training.pipeline.publish import load_index, retry_releases, rollback
from training.pipeline.state import load_state
from training.pipeline.test_pipeline import REPO, FakeExecutor, make_pipe
from training.pipeline.runner import Pipeline

REGRESSION_BATCH = 900  # publish.py seeds the regression match at cycle*1000 + 900


class FakeGitHub:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.releases: dict[str, list[str]] = {}
        self.latest: str | None = None

    def create_release(self, tag, title, notes, files):
        if self.fail:
            raise RuntimeError("network down")
        self.releases[tag] = [f.name for f in files]
        self.latest = tag
        return f"https://github.com/o/r/releases/tag/{tag}"

    def mark_latest(self, tag):
        self.latest = tag


def arena(regression_score_wins: bool = True):
    """Every candidate sweeps its gate; the regression match is scripted."""
    def play(cycle, batch, games):
        if batch == REGRESSION_BATCH:
            return (games, 0, 0) if regression_score_wins else (0, games, 0)
        return games, 0, 0
    return play


def patch_github(monkeypatch, gh: FakeGitHub) -> None:
    monkeypatch.setattr("training.pipeline.publish.GitHub", lambda repo, cwd: gh)


def test_every_best_is_kept_and_alias_moves(tmp_path, monkeypatch):
    gh = FakeGitHub()
    patch_github(monkeypatch, gh)
    alias = tmp_path / "best.onnx"
    pipe = make_pipe(tmp_path, FakeExecutor(arena=arena()), publish__alias=str(alias), publish__github_release=True)
    pipe.run(max_cycles=3)

    reg = tmp_path / "registry"
    index = load_index(reg)
    assert [e["generation"] for e in index["entries"]] == [1, 2, 3]
    assert index["best"] == "t/gen0003" and index["previous_best"] == "t/gen0002"
    for g in (1, 2, 3):  # no older best was deleted or overwritten
        assert (reg / f"gofer-9x9-t-gen{g:04d}.onnx").read_bytes() == f"candidate-{g:04d}.onnx".encode()
        card = json.loads((reg / f"gofer-9x9-t-gen{g:04d}.json").read_text())
        assert card["sha256"] and card["release_url"].endswith(f"gofer-9x9-t-gen{g:04d}")
    assert alias.read_bytes() == b"candidate-0003.onnx"
    assert set(gh.releases) == {"gofer-9x9-t-gen0001", "gofer-9x9-t-gen0002", "gofer-9x9-t-gen0003"}
    assert gh.latest == "gofer-9x9-t-gen0003"


def test_regression_failure_holds_publication(tmp_path, monkeypatch):
    patch_github(monkeypatch, FakeGitHub())
    alias = tmp_path / "best.onnx"
    pipe = make_pipe(tmp_path, FakeExecutor(arena=arena(regression_score_wins=False)),
                     publish__alias=str(alias), publish__regression_games=10, publish__regression_lookback=1)
    pipe.run(max_cycles=2)

    index = load_index(tmp_path / "registry")
    statuses = {e["generation"]: e["status"] for e in index["entries"]}
    assert statuses == {1: "published", 2: "held"}  # gen 1 has no older baseline to check against
    assert index["best"] == "t/gen0001"
    assert alias.read_bytes() == b"candidate-0001.onnx"
    held = index["entries"][1]
    assert held["regression"]["vs_generation"] == 1 and not held["regression"]["passed"]
    # hold: gen 2 remains the training champion
    assert load_state(pipe.state_path).champion.generation == 2


def test_regression_demote_reverts_champion(tmp_path, monkeypatch):
    patch_github(monkeypatch, FakeGitHub())
    pipe = make_pipe(tmp_path, FakeExecutor(arena=arena(regression_score_wins=False)),
                     publish__regression_games=10, publish__regression_lookback=1, publish__regression_action="demote")
    pipe.run(max_cycles=3)
    st = load_state(pipe.state_path)
    # cycle 2 promoted gen 2 then demoted back to gen 1; cycle 3's candidate becomes gen 3 (no number reuse)
    assert [g.generation for g in st.generations] == [1, 2, 1, 3, 1]
    assert st.champion.generation == 1


def test_failed_release_does_not_stop_training_and_retries(tmp_path, monkeypatch):
    gh = FakeGitHub(fail=True)
    patch_github(monkeypatch, gh)
    pipe = make_pipe(tmp_path, FakeExecutor(arena=arena()), publish__github_release=True)
    assert pipe.run(max_cycles=2) == 2
    index = load_index(tmp_path / "registry")
    assert all("release_error" in e and "release_url" not in e for e in index["entries"])

    gh.fail = False
    done = retry_releases(pipe, gh)
    assert done == ["t/gen0001", "t/gen0002"]
    assert all(e.get("release_url") for e in load_index(tmp_path / "registry")["entries"])


def test_rollback_repoints_best_without_deleting(tmp_path, monkeypatch):
    gh = FakeGitHub()
    patch_github(monkeypatch, gh)
    alias = tmp_path / "best.onnx"
    pipe = make_pipe(tmp_path, FakeExecutor(arena=arena()), publish__alias=str(alias), publish__github_release=True)
    pipe.run(max_cycles=3)

    pipe = Pipeline(pipe.cfg, pipe.x, REPO, out=lambda _m: None)
    res = rollback(pipe, 2, champion=True, github=gh)
    assert res["best"] == "t/gen0002" and res["previous_best"] == "t/gen0003"
    assert alias.read_bytes() == b"candidate-0002.onnx"
    assert gh.latest == "gofer-9x9-t-gen0002"
    assert (tmp_path / "registry" / "gofer-9x9-t-gen0003.onnx").exists()
    st = load_state(pipe.state_path)
    assert st.champion.generation == 2 and [g.generation for g in st.generations][-1] == 2

    # next promotion gets a fresh number, never overwriting gen 3's files
    pipe.run(max_cycles=1)
    assert load_state(pipe.state_path).champion.generation == 4

    with pytest.raises(ValueError):
        rollback(pipe, 99, champion=False, github=gh)
