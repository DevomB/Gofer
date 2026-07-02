"""Tests for FIFO replay buffer."""

from __future__ import annotations

from pathlib import Path

from replay import append_jsonl, count_lines, trim

FIXTURE = Path(__file__).resolve().parent / "testdata" / "fixture_samples.jsonl"


def test_append_and_count(tmp_path: Path) -> None:
    dst = tmp_path / "replay.jsonl"
    n = append_jsonl(FIXTURE, dst)
    assert n > 0
    assert count_lines(dst) == n


def test_append_skips_duplicate_header(tmp_path: Path) -> None:
    dst = tmp_path / "replay.jsonl"
    append_jsonl(FIXTURE, dst)
    before = count_lines(dst)
    append_jsonl(FIXTURE, dst)
    after = count_lines(dst)
    assert after > before
    lines = dst.read_text(encoding="utf-8").strip().splitlines()
    headers = [ln for ln in lines if '"type":"header"' in ln.replace(" ", "")]
    assert len(headers) == 1


def test_trim_fifo(tmp_path: Path) -> None:
    dst = tmp_path / "replay.jsonl"
    append_jsonl(FIXTURE, dst)
    append_jsonl(FIXTURE, dst)
    removed = trim(dst, 10)
    assert count_lines(dst) <= 10
    assert removed >= 0
