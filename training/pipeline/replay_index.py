"""Shard-based replay buffer: an index over selfplay/*.npz plus window policy.

Replaces the v3 replay.jsonl (append + full-file rewrite on every trim). Shards
are immutable; the "buffer" is just the newest shards whose rows fit in the
current window, and aging out is a file move into selfplay/archive/.
"""

from __future__ import annotations

import ast
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from training.pipeline.config import ReplayConfig


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    rows: int
    games: int
    model: str
    cycle: int


def read_shard_meta(path: Path) -> dict:
    """Read the shard's meta JSON without decompressing the data arrays."""
    with zipfile.ZipFile(path) as zf:
        blob = zf.read("meta.npy")
    if blob[:6] != b"\x93NUMPY":
        raise ValueError(f"{path}: meta.npy is not an npy array")
    major = blob[6]
    if major == 1:
        hlen, start = int.from_bytes(blob[8:10], "little"), 10
    else:
        hlen, start = int.from_bytes(blob[8:12], "little"), 12
    header = ast.literal_eval(blob[start : start + hlen].decode("latin1"))
    if header.get("descr") != "|u1":
        raise ValueError(f"{path}: meta must be uint8 bytes")
    meta = json.loads(blob[start + hlen :].decode("utf-8"))
    if meta.get("format") != "gofer-shard":
        raise ValueError(f"{path}: not a gofer shard (format={meta.get('format')!r})")
    return meta


def cycle_of(path: Path) -> int:
    """selfplay/cycle-0007.npz -> 7; unknown naming -> -1."""
    stem = path.stem
    if stem.startswith("cycle-"):
        try:
            return int(stem.split("-", 1)[1].split(".")[0])
        except ValueError:
            return -1
    return -1


def scan(directory: Path) -> list[ShardInfo]:
    """All shards in directory, oldest first (by cycle, then name)."""
    if not directory.exists():
        return []
    out = []
    for p in directory.glob("*.npz"):
        meta = read_shard_meta(p)
        out.append(ShardInfo(p, int(meta.get("rows", 0)), int(meta.get("games", 0)), str(meta.get("model", "")), cycle_of(p)))
    out.sort(key=lambda s: (s.cycle, s.path.name))
    return out


def window_rows(total_rows: int, cfg: ReplayConfig) -> int:
    """KataGo's power-law window: grows like total^alpha once past min_window_rows.

    window = c * (1 + beta * ((N/c)^alpha - 1) / alpha), c = min_window_rows,
    clamped to [min(c, N), max_window_rows].
    """
    c = cfg.min_window_rows
    if total_rows <= c:
        return total_rows
    grown = c * (1 + cfg.window_beta * ((total_rows / c) ** cfg.window_alpha - 1) / cfg.window_alpha)
    return int(min(grown, cfg.max_window_rows))


def total_rows(shards: list[ShardInfo]) -> int:
    return sum(s.rows for s in shards)


def select_window(shards: list[ShardInfo], rows: int) -> list[ShardInfo]:
    """Newest shards covering at least `rows` rows (whole shards; oldest first)."""
    picked: list[ShardInfo] = []
    acc = 0
    for s in reversed(shards):
        if acc >= rows:
            break
        picked.append(s)
        acc += s.rows
    return list(reversed(picked))


def archive_old(directory: Path, cfg: ReplayConfig, lifetime_rows: int) -> list[Path]:
    """Move shards beyond archive_factor * window into directory/archive/. Returns moved paths.

    lifetime_rows is every row ever generated (including archived shards), so the
    window keeps growing even after old shards leave the live directory.
    """
    if cfg.archive_factor <= 0:
        return []
    shards = scan(directory)
    keep_rows = int(window_rows(lifetime_rows, cfg) * cfg.archive_factor)
    keep = {s.path for s in select_window(shards, keep_rows)}
    archive = directory / "archive"
    moved = []
    for s in shards:
        if s.path in keep:
            continue
        archive.mkdir(parents=True, exist_ok=True)
        dest = archive / s.path.name
        shutil.move(str(s.path), dest)
        moved.append(dest)
    return moved
