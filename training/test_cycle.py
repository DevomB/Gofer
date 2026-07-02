"""Tests for train-loop v3 cycle helpers."""

from __future__ import annotations

import json
from pathlib import Path

from training.cycle import append_replay, init_from_cycle2, record_cycle, should_promote


def test_should_promote_head_to_head() -> None:
    # Clears the win bar and is significantly above 0.5 -> promote.
    assert should_promote(0.60, 0.53, promote_win=0.55)
    # Wins on average but not significantly (CI low <= 0.5) -> reject noise.
    assert not should_promote(0.56, 0.49, promote_win=0.55)
    # Below the win bar -> reject.
    assert not should_promote(0.52, 0.50, promote_win=0.55)


def test_seed_champion_forces_promote(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state"
    data = tmp_path / "data"
    state.mkdir(parents=True)
    data.mkdir(parents=True)
    manifest = state / "manifest.json"
    replay = data / "replay.jsonl"
    manifest.write_text(json.dumps({"version": 3, "cycle": 0, "replay_rows": 0, "seed": "fresh"}), encoding="utf-8")
    replay.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("training.cycle.STATE_DIR", state)
    monkeypatch.setattr("training.cycle.DATA_DIR", data)
    monkeypatch.setattr("training.cycle.MANIFEST_PATH", manifest)
    monkeypatch.setattr("training.cycle.REPLAY_PATH", replay)

    report = tmp_path / "arena.json"
    # A losing sanity result (0.42 vs heuristic) must still seed the first champion.
    report.write_text(json.dumps({"win_rate_challenger": 0.42, "wilson_ci_low": 0.35, "wilson_ci_high": 0.49}))
    promote, _ = record_cycle(1, report, promote_win=0.55, history_dir=tmp_path / "h", seed_champion=True)
    assert promote


def test_record_cycle_writes_history(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state"
    data = tmp_path / "data"
    history = tmp_path / "history"
    manifest = state / "manifest.json"
    replay = data / "replay.jsonl"
    state.mkdir(parents=True)
    data.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "version": 3,
                "cycle": 1,
                "best_win_rate": 0.58,
                "best_cycle": 2,
                "replay_rows": 1,
                "seed": "test",
            }
        ),
        encoding="utf-8",
    )
    replay.write_text('{"type":"header"}\n{}\n', encoding="utf-8")

    monkeypatch.setattr("training.cycle.STATE_DIR", state)
    monkeypatch.setattr("training.cycle.DATA_DIR", data)
    monkeypatch.setattr("training.cycle.MANIFEST_PATH", manifest)
    monkeypatch.setattr("training.cycle.REPLAY_PATH", replay)

    report = tmp_path / "arena.json"
    report.write_text(json.dumps({"win_rate_challenger": 0.62, "wilson_ci_low": 0.55, "wilson_ci_high": 0.7}))

    promote, entry = record_cycle(3, report, promote_win=0.55, history_dir=history)
    assert promote
    assert entry["win_rate"] == 0.62
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["best_win_rate"] == 0.62
    assert (history / "cycle-3.json").exists()
