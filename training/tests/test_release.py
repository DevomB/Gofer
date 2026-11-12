"""Tests for champion tracking and GitHub releases (no gofer binary or network needed)."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from training import release
from training.loop import LoopConfig, Run


def test_won_needs_sprt_acceptance():
    assert release.won({"sprt": "H1", "elo_ci_low": -50})
    assert not release.won({"sprt": "H0", "elo_ci_low": 10})
    # an undecided match promotes nothing, however good its point estimate
    assert not release.won({"sprt": "", "elo_ci_low": 5})
    assert not release.won({"elo_ci_low": 5})


def fake_gh(tmp: Path, existing: bool = False, fail: bool = False) -> Path:
    """A gh stand-in that records its arguments; `release view` finds the
    release only when existing, `release create` fails when fail."""
    log = tmp / "gh.log"
    script = tmp / "gh"
    script.write_text(f"""#!/bin/sh
echo "$@" >> {log}
case "$2" in
  view) exit {0 if existing else 1} ;;
  create) {'echo "HTTP 403" >&2; exit 1' if fail else 'echo https://github.com/o/r/releases/tag/$3; exit 0'} ;;
esac
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "/dev/null")
    root = tmp_path / "9x9"
    r = Run(root, LoopConfig(champion_min_elo=1000), Path("gofer"), threads=1)
    for g in range(5):
        r.model_path(g).write_bytes(f"model {g}".encode())
        r.state["done"][str(g)] = {"selfplay": {"rows": 100, "seconds": 1.0}}
    r.state["results"] = [{"a": "gen-0002", "b": "gnugo-l1", "wins": 30, "losses": 10, "draws": 0}]
    monkeypatch.setattr(r, "log", lambda msg: None)
    return r


def set_ratings(r: Run, elos: dict[int, float]) -> None:
    ratings = {f"gen-{g:04d}": {"elo": e, "stderr": 30.0, "games": 40} for g, e in elos.items()}
    (r.root / "eval" / "ratings.json").write_text(json.dumps(ratings))


def test_first_champion_needs_min_elo_then_challengers_must_win(run, monkeypatch):
    # gen 2 (an anchor generation) is rated below champion_min_elo: no champion
    set_ratings(run, {0: 0, 1: 400, 2: 900})
    run.champion(1)
    assert "champion" not in run.state

    # gen 2 passes the bar later: it becomes the first champion without a match
    run.state["done"]["1"].pop("champion")
    set_ratings(run, {0: 0, 1: 400, 2: 1200})
    run.champion(1)
    assert run.state["champion"]["gen"] == 2
    d = run.root / "champion" / "gen-0002"
    info = json.loads((d / "champion.json").read_text())
    assert info["games"] == 600 * 2 and info["positions"] == 200
    assert info["anchors"] == [{"a": "gen-0002", "b": "gnugo-l1", "wins": 30, "losses": 10, "draws": 0}]
    assert (run.root / "champion" / "gofer-9x9.onnx").read_bytes() == b"model 2"
    assert run.state["releases"] == [{"gen": 2, "tag": "model-9x9-g0002", "published": False}]

    # gen 4 is rated higher and wins its SPRT match: new champion
    calls = []
    def match(a, b, games, visits, tag, extra=None):
        calls.append((a, b, extra))
        return {"wins": 70, "losses": 30, "draws": 0, "elo": 70, "elo_ci_low": 30, "elo_ci_high": 110, "sprt": "H1"}
    monkeypatch.setattr(run, "match", match)
    set_ratings(run, {0: 0, 1: 400, 2: 1200, 4: 1300})
    run.champion(3)
    assert calls[0][1].endswith("gen-0002.onnx") and calls[0][2] == ["-sprt", "0,50"]
    assert run.state["champion"]["gen"] == 4
    assert (run.root / "champion" / "gofer-9x9.onnx").read_bytes() == b"model 4"
    assert run.state["results"][-1] == {"a": "gen-0004", "b": "gen-0002", "wins": 70, "losses": 30, "draws": 0}


def test_challenger_that_does_not_win_is_not_promoted(run, monkeypatch):
    run.state["champion"] = {"gen": 2, "elo": 1200, "tag": "model-9x9-g0002"}
    monkeypatch.setattr(run, "match", lambda *a, **k: {"wins": 45, "losses": 55, "draws": 0, "elo": -35,
                                                       "elo_ci_low": -90, "elo_ci_high": 20, "sprt": "H0"})
    set_ratings(run, {0: 0, 2: 1200, 4: 1250})
    run.champion(3)
    assert run.state["champion"]["gen"] == 2


def test_lower_rated_generation_does_not_play_the_champion(run, monkeypatch):
    run.state["champion"] = {"gen": 2, "elo": 1200, "tag": "model-9x9-g0002"}
    monkeypatch.setattr(run, "match", lambda *a, **k: pytest.fail("no match expected"))
    set_ratings(run, {0: 0, 2: 1200, 4: 1150})
    run.champion(3)
    assert run.state["champion"]["gen"] == 2


def test_a_champion_is_challenged_on_a_schedule_whatever_the_ratings_say(run, monkeypatch):
    # Seen in a real run: the last network beat the champion 97-23 in a direct
    # match, but the ladder had rated every generation after the champion below
    # it for 20 generations, so none of them ever played it.
    run.state["champion"] = {"gen": 2, "elo": 1200, "tag": "model-9x9-g0002"}
    for g in range(5, 21):
        run.model_path(g).write_bytes(f"model {g}".encode())
        run.state["done"][str(g)] = {"selfplay": {"rows": 100, "seconds": 1.0}}
    played = []
    result = {"wins": 45, "losses": 55, "draws": 0, "elo": -35, "elo_ci_low": -90, "elo_ci_high": 20, "sprt": "H0"}

    def match(a, b, games, visits, tag, extra=None):
        played.append(tag)
        return result

    monkeypatch.setattr(run, "match", match)
    below = lambda new: set_ratings(run, {0: 0, 2: 1200, new: 1100})

    for new in (4, 6, 8):  # fewer than champion_every (8) generations after the champion
        below(new)
        run.champion(new - 1)
    assert played == []
    below(10)
    run.champion(9)  # eight generations after it: plays although rated below
    assert played == ["gen-0010-vs-champion-gen-0002"] and run.state["champion"]["gen"] == 2
    below(12)
    run.champion(11)  # the champion was challenged two generations ago
    assert len(played) == 1
    result = {"wins": 97, "losses": 23, "draws": 0, "elo": 250, "elo_ci_low": 172, "elo_ci_high": 328, "sprt": "H1"}
    below(18)
    run.champion(17)  # eight generations after the last challenge
    assert played[-1] == "gen-0018-vs-champion-gen-0002" and run.state["champion"]["gen"] == 18
    below(20)
    run.champion(19)  # the new champion's own count starts at its generation
    assert len(played) == 2


def test_champion_every_zero_leaves_challenges_to_the_ratings(tmp_path, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "/dev/null")
    r = Run(tmp_path / "9x9", LoopConfig(champion_min_elo=1000, champion_every=0), Path("gofer"), threads=1)
    monkeypatch.setattr(r, "log", lambda msg: None)
    monkeypatch.setattr(r, "match", lambda *a, **k: pytest.fail("no match expected"))
    r.state["champion"] = {"gen": 2, "elo": 1200, "tag": "model-9x9-g0002"}
    set_ratings(r, {0: 0, 2: 1200, 40: 1100})
    r.champion(39)
    assert r.state["champion"]["gen"] == 2


def test_publish_newest_pending_with_stable_asset_name(run, tmp_path):
    set_ratings(run, {0: 0, 2: 1200, 4: 1300})
    run.promote(2, json.loads((run.root / "eval" / "ratings.json").read_text()), None)
    run.promote(4, json.loads((run.root / "eval" / "ratings.json").read_text()),
                {"wins": 70, "losses": 30, "draws": 0, "elo": 70, "elo_ci_low": 30, "elo_ci_high": 110,
                 "sprt": "H1", "sprt_pairs": 50, "sprt_sweeps_a": 25, "sprt_sweeps_b": 5,
                 "opponent": "gen-0002", "visits": 200})
    gh = fake_gh(tmp_path)
    assert release.publish_pending(run.root, run.state, "o/r", log=lambda m: None, gh=str(gh))
    assert run.state["releases"] == [
        {"gen": 2, "tag": "model-9x9-g0002", "published": False, "superseded": True},
        {"gen": 4, "tag": "model-9x9-g0004", "published": True},
    ]
    create = [l for l in (tmp_path / "gh.log").read_text().splitlines() if l.startswith("release create")]
    assert len(create) == 1
    args = create[0].split()
    assert args[2] == "model-9x9-g0004" and "--latest" in args and args[args.index("--repo") + 1] == "o/r"
    assert args[3].endswith("gen-0004/gofer-9x9.onnx") and args[4].endswith("gen-0004/champion.json")
    text = (run.root / "champion" / "gen-0004" / "notes.md").read_text()
    assert "beat generation 2 70–30" in text and "releases/latest/download/gofer-9x9.onnx" in text
    assert "SPRT accepted after 50 opening pairs: 25 sweeps to 5" in text
    assert release.sha256(run.root / "champion" / "gen-0004" / "gofer-9x9.onnx") in text


def test_publish_failure_keeps_release_pending(run, tmp_path):
    set_ratings(run, {0: 0, 2: 1200})
    run.promote(2, json.loads((run.root / "eval" / "ratings.json").read_text()), None)
    assert release.publish_pending(run.root, run.state, "o/r", log=lambda m: None, gh=str(fake_gh(tmp_path, fail=True)))
    assert run.state["releases"][0]["published"] is False
    # an existing release (e.g. published before state was saved) counts as published
    assert release.publish_pending(run.root, run.state, "o/r", log=lambda m: None, gh=str(fake_gh(tmp_path, existing=True)))
    assert run.state["releases"][0]["published"] is True


def test_no_repo_means_no_publishing(run, tmp_path):
    set_ratings(run, {0: 0, 2: 1200})
    run.promote(2, json.loads((run.root / "eval" / "ratings.json").read_text()), None)
    assert not release.publish_pending(run.root, run.state, "", gh=str(fake_gh(tmp_path)))
    assert not (tmp_path / "gh.log").exists()
