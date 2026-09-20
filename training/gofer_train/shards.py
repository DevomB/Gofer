"""GOFER SHARD v1: packed self-play training rows (.npz).

Produced directly by the Go engine (``gofer -selfplay -o x.npz``) or by
``python -m gofer_train.shards pack`` from legacy JSONL. Layout (one board size
per shard, N rows, S = board size):

  spatial      uint8   [N, 8, S, S]   0/1 feature planes (BuildFeaturesV2 order)
  globals      float32 [N, 4]
  policy       float32 [N, S*S+1]     normalized visit distribution
  policy_opp   float32 [N, S*S+1]     opponent's reply policy; all-zero if unknown
  value        float32 [N]            outcome, side-to-move: -1/0/+1
  score        float32 [N]            final area margin incl. komi, side-to-move (NaN = unknown)
  ownership    int8    [N, S*S]       final owner, side-to-move: +1 own / -1 opp / 0
  full_search  uint8   [N]            1 = full-cap search (policy target trusted)
  game_id      int32   [N]
  move_num     int16   [N]
  meta         uint8   [K]            UTF-8 JSON (format, version, board_size, rows, ...)

CLI::

  python -m gofer_train.shards pack --in replay.jsonl --out training/data/shards/
  python -m gofer_train.shards inspect training/data/shards/
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

FORMAT = "gofer-shard"
VERSION = 1
PLANES = 8
GLOBALS = 4
BLACK, WHITE = 1, 2


@dataclass
class Rows:
    """Column-oriented block of training rows (numpy, CPU)."""

    spatial: np.ndarray
    globals: np.ndarray
    policy: np.ndarray
    policy_opp: np.ndarray
    value: np.ndarray
    score: np.ndarray
    ownership: np.ndarray
    full_search: np.ndarray
    game_id: np.ndarray
    move_num: np.ndarray

    def __len__(self) -> int:
        return int(self.value.shape[0])

    @property
    def board_size(self) -> int:
        return int(self.spatial.shape[-1])

    def take(self, idx: np.ndarray | slice) -> "Rows":
        return Rows(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    @staticmethod
    def concat(parts: list["Rows"]) -> "Rows":
        if not parts:
            raise ValueError("no rows")
        sizes = {p.board_size for p in parts}
        if len(sizes) != 1:
            raise ValueError(f"mixed board sizes {sorted(sizes)} in one Rows block")
        out = {}
        offset = 0
        gids = []
        for p in parts:
            # Keep game ids unique across shards so game-level splits stay correct.
            g = p.game_id.astype(np.int64)
            gids.append(g - (g.min() if len(g) else 0) + offset)
            offset = int(gids[-1].max()) + 1 if len(g) else offset
        for f in fields(Rows):
            if f.name == "game_id":
                out[f.name] = np.concatenate(gids)
            else:
                out[f.name] = np.concatenate([getattr(p, f.name) for p in parts])
        return Rows(**out)


def empty_rows(n: int, size: int) -> Rows:
    pol = size * size + 1
    return Rows(
        spatial=np.zeros((n, PLANES, size, size), np.uint8),
        globals=np.zeros((n, GLOBALS), np.float32),
        policy=np.zeros((n, pol), np.float32),
        policy_opp=np.zeros((n, pol), np.float32),
        value=np.zeros(n, np.float32),
        score=np.full(n, np.nan, np.float32),
        ownership=np.zeros((n, size * size), np.int8),
        full_search=np.ones(n, np.uint8),
        game_id=np.zeros(n, np.int64),
        move_num=np.zeros(n, np.int16),
    )


# ------------------------------------------------------------------ npz ----


def write_shard(path: Path, rows: Rows, **meta: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "format": FORMAT,
        "version": VERSION,
        "board_size": rows.board_size,
        "rows": len(rows),
        "games": int(len(np.unique(rows.game_id))),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **meta,
    }
    arrays = {
        "spatial": rows.spatial.astype(np.uint8),
        "globals": rows.globals.astype(np.float32),
        "policy": rows.policy.astype(np.float32),
        "policy_opp": rows.policy_opp.astype(np.float32),
        "value": rows.value.astype(np.float32),
        "score": rows.score.astype(np.float32),
        "ownership": rows.ownership.astype(np.int8),
        "full_search": rows.full_search.astype(np.uint8),
        "game_id": rows.game_id.astype(np.int32),
        "move_num": rows.move_num.astype(np.int16),
        "meta": np.frombuffer(json.dumps(info).encode(), dtype=np.uint8),
    }
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)  # deflate, like the Go writer (~40x smaller than JSONL)
    tmp.replace(path)
    return path


def read_meta(path: Path) -> dict:
    with np.load(path) as z:
        if "meta" not in z.files:
            return {}
        return json.loads(bytes(z["meta"]).decode())


def read_shard(path: Path) -> Rows:
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode()) if "meta" in z.files else {}
        if meta.get("format", FORMAT) != FORMAT or int(meta.get("version", VERSION)) > VERSION:
            raise ValueError(f"{path}: unsupported shard {meta.get('format')} v{meta.get('version')}")
        n = int(z["value"].shape[0])
        size = int(z["spatial"].shape[-1])
        pol = size * size + 1

        def opt(name: str, default: np.ndarray) -> np.ndarray:
            return z[name] if name in z.files else default

        return Rows(
            spatial=z["spatial"].astype(np.uint8, copy=False),
            globals=z["globals"].astype(np.float32, copy=False),
            policy=z["policy"].astype(np.float32, copy=False),
            policy_opp=opt("policy_opp", np.zeros((n, pol), np.float32)).astype(np.float32, copy=False),
            value=z["value"].astype(np.float32, copy=False),
            score=opt("score", np.full(n, np.nan, np.float32)).astype(np.float32, copy=False),
            ownership=opt("ownership", np.zeros((n, size * size), np.int8)).astype(np.int8, copy=False),
            full_search=opt("full_search", np.ones(n, np.uint8)).astype(np.uint8, copy=False),
            game_id=opt("game_id", np.zeros(n, np.int32)).astype(np.int64),
            move_num=opt("move_num", np.zeros(n, np.int16)).astype(np.int16, copy=False),
        )


def is_shard(path: Path) -> bool:
    return path.suffix == ".npz" and zipfile.is_zipfile(path)


# ---------------------------------------------------------------- jsonl ----


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "header":
                if row.get("schema_version", 1) != 1:
                    raise ValueError(f"unsupported JSONL schema {row.get('schema_version')}")
                continue
            yield row


def rows_from_jsonl(path: Path, board_size: int = 9) -> Rows:
    """Legacy JSONL -> Rows, normalizing conventions to the shard's.

    * ownership: JSONL is absolute (Black=+1); shards are side-to-move. It is
      required on every row, matching WriteSampleShard.
    * policy_opp: JSONL's ``policy_opp`` is the *previous* move's policy, not a
      reply target, so it is ignored; ``policy_next`` (if present) is used.
    * game ids: JSONL has none; a new game starts whenever move_num does not
      increase.
    """
    pol = board_size * board_size + 1
    spatial_len = PLANES * board_size * board_size
    kept: list[dict] = [
        r
        for r in _iter_jsonl(path)
        if len(r.get("policy") or ()) == pol and len(r.get("features_spatial") or ()) == spatial_len
    ]
    if not kept:
        raise ValueError(f"no valid {board_size}x{board_size} samples in {path}")
    n = len(kept)
    rows = empty_rows(n, board_size)
    rows.spatial[:] = (
        np.asarray([r["features_spatial"] for r in kept], np.float32)
        .reshape(n, PLANES, board_size, board_size)
        .round()
        .astype(np.uint8)
    )
    rows.globals[:] = np.asarray([r["features_global"] for r in kept], np.float32)
    rows.policy[:] = np.asarray([r["policy"] for r in kept], np.float32)
    rows.value[:] = np.asarray([float(r.get("value", 0.0)) for r in kept], np.float32)

    game, prev_move = -1, None
    for i, r in enumerate(kept):
        mv = int(r.get("move_num", 0))
        if "game_id" in r:
            game = int(r["game_id"])
        elif prev_move is None or mv <= prev_move:
            game += 1
        prev_move = mv
        rows.game_id[i] = game
        rows.move_num[i] = mv
        rows.full_search[i] = 1 if r.get("full_search", True) else 0
        sign = -1 if int(r.get("to_play", BLACK)) == WHITE else 1
        own = r.get("ownership")
        if not own or len(own) != board_size * board_size:
            raise ValueError(
                f"{path}: row {i} has {len(own or ())} ownership labels, want "
                f"{board_size * board_size}. The ownership loss is unmasked, so a "
                f"zero-filled row trains the head toward neutral everywhere. "
                f"WriteSampleShard rejects this too; regenerate the JSONL."
            )
        rows.ownership[i] = np.sign(np.asarray(own, np.float32) * sign).astype(np.int8)
        nxt = r.get("policy_next")
        if nxt and len(nxt) == pol:
            rows.policy_opp[i] = nxt
        if "score_margin" in r:
            rows.score[i] = float(r["score_margin"])
    return rows


# --------------------------------------------------------------- loading ----


def list_sources(data: Path) -> list[Path]:
    """Expand a file or directory into ordered sources (oldest first)."""
    if data.is_dir():
        found = sorted(
            [p for p in data.iterdir() if p.suffix in (".npz", ".jsonl") and not p.name.endswith(".tmp.npz")],
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        if not found:
            raise FileNotFoundError(f"no *.npz or *.jsonl in {data}")
        return found
    if not data.exists():
        raise FileNotFoundError(data)
    return [data]


def load_rows(data: Path | Iterable[Path], *, board_size: int = 9, window_rows: int = 0) -> Rows:
    """Load one or many sources. ``window_rows`` keeps only the newest N rows."""
    sources: list[Path] = []
    for d in [data] if isinstance(data, Path) else list(data):
        sources.extend(list_sources(Path(d)))
    parts: list[Rows] = []
    total = 0
    # Walk newest -> oldest so a window never reads shards it would discard.
    for src in reversed(sources):
        if window_rows and total >= window_rows:
            break
        part = read_shard(src) if src.suffix == ".npz" else rows_from_jsonl(src, board_size)
        if part.board_size != board_size:
            print(f"skip {src}: board {part.board_size} != {board_size}", file=sys.stderr)
            continue
        parts.append(part)
        total += len(part)
    parts.reverse()
    rows = Rows.concat(parts)
    if window_rows and len(rows) > window_rows:
        rows = rows.take(slice(len(rows) - window_rows, None))
    return rows


# -------------------------------------------------------------------- cli ----


def _pack(args: argparse.Namespace) -> None:
    rows = load_rows(Path(args.inp), board_size=args.board_size)
    out = Path(args.out)
    if out.suffix != ".npz":
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(args.inp).stem
        per = args.rows_per_shard or len(rows)
        # Split on game boundaries so no game straddles two shards.
        starts = np.flatnonzero(np.r_[True, rows.game_id[1:] != rows.game_id[:-1]])
        cut, idx = 0, 0
        while cut < len(rows):
            target = cut + per
            nxt = starts[starts >= target]
            end = int(nxt[0]) if len(nxt) and target < len(rows) else len(rows)
            p = write_shard(out / f"{stem}-{idx:04d}.npz", rows.take(slice(cut, end)), source=str(args.inp))
            print(f"wrote {p} rows={end - cut}")
            cut, idx = end, idx + 1
    else:
        write_shard(out, rows, source=str(args.inp))
        print(f"wrote {out} rows={len(rows)}")


def _inspect(args: argparse.Namespace) -> None:
    for src in list_sources(Path(args.path)):
        if src.suffix != ".npz":
            continue
        meta = read_meta(src)
        rows = read_shard(src)
        full = float(rows.full_search.mean()) if len(rows) else 0.0
        has_opp = float((rows.policy_opp.sum(1) > 0).mean()) if len(rows) else 0.0
        has_score = float(np.isfinite(rows.score).mean()) if len(rows) else 0.0
        print(
            f"{src.name}: rows={len(rows)} games={len(np.unique(rows.game_id))} "
            f"size={rows.board_size} full_search={full:.2f} policy_opp={has_opp:.2f} "
            f"score={has_score:.2f} value_mean={rows.value.mean():+.3f} meta={json.dumps(meta)}"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="GOFER SHARD v1 tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pack", help="convert JSONL (or shards) into .npz shard(s)")
    pk.add_argument("--in", dest="inp", required=True)
    pk.add_argument("--out", required=True, help="file.npz or a directory")
    pk.add_argument("--board-size", type=int, default=9)
    pk.add_argument("--rows-per-shard", type=int, default=25000)
    ins = sub.add_parser("inspect", help="summarize shard file(s)")
    ins.add_argument("path")
    args = p.parse_args(argv)
    {"pack": _pack, "inspect": _inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
