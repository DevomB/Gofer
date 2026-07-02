"""Train-loop v3 cycle helpers (manifest, replay, promote decision)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.manifest import default_manifest, load_manifest, save_manifest, utc_now
from training.replay import append_jsonl, count_lines, trim

STATE_DIR = Path("training/state")
DATA_DIR = Path("training/data")
MANIFEST_PATH = STATE_DIR / "manifest.json"
REPLAY_PATH = DATA_DIR / "replay.jsonl"
BEST_PT = STATE_DIR / "best.pt"
BEST_ONNX = Path("models/gofer-9x9-best.onnx")
BOOTSTRAP_ONNX = Path("models/gofer-9x9-bootstrap.onnx")


def is_header_line(line: str) -> bool:
    return '"type":"header"' in line.replace(" ", "")


def init_from_cycle2(*, best_win_rate: float = 0.58, best_cycle: int = 2) -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src_pt = Path("training/checkpoints/cycle2/best.pt")
    if not src_pt.exists():
        raise FileNotFoundError(f"missing seed checkpoint: {src_pt}")
    shutil.copy2(src_pt, BEST_PT)
    if REPLAY_PATH.exists():
        REPLAY_PATH.unlink()
    for name in ("samples-cycle1.jsonl", "samples-cycle2.jsonl"):
        path = DATA_DIR / name
        if path.exists():
            append_jsonl(path, REPLAY_PATH)
    manifest = default_manifest(seed="cycle2", best_win_rate=best_win_rate, best_cycle=best_cycle)
    manifest["cycle"] = best_cycle
    manifest["replay_rows"] = count_lines(REPLAY_PATH)
    save_manifest(MANIFEST_PATH, manifest)
    if BEST_ONNX.exists():
        shutil.copy2(BEST_ONNX, BOOTSTRAP_ONNX)
    return manifest


def append_replay(samples: Path, *, max_lines: int = 50000) -> int:
    append_jsonl(samples, REPLAY_PATH)
    trim(REPLAY_PATH, max_lines)
    return count_lines(REPLAY_PATH)


def should_promote(win_rate: float, wilson_ci_low: float, *, promote_win: float) -> bool:
    """Head-to-head gate: the candidate must beat the CURRENT champion.

    Promotion requires clearing the win-rate bar (default 0.55) AND the Wilson
    lower bound being above 0.5, i.e. significantly better than a coin flip over
    the arena sample. There is no fixed-heuristic anchor and no win-target stop,
    so strength keeps climbing until candidates can no longer beat the champion.
    """
    return win_rate >= promote_win and wilson_ci_low > 0.5


def record_cycle(
    cycle: int,
    report_path: Path,
    *,
    promote_win: float,
    history_dir: Path,
) -> tuple[bool, dict]:
    arena = json.loads(report_path.read_text(encoding="utf-8"))
    rate = float(arena.get("win_rate_challenger", 0))
    ci_low = float(arena.get("wilson_ci_low", 0))
    manifest = load_manifest(MANIFEST_PATH)
    promote = should_promote(rate, ci_low, promote_win=promote_win)

    manifest["cycle"] = cycle
    manifest["replay_rows"] = count_lines(REPLAY_PATH)
    entry = {
        "cycle": cycle,
        "win_rate": rate,
        "wilson_ci_low": ci_low,
        "wilson_ci_high": float(arena.get("wilson_ci_high", 0)),
        "promote_threshold": promote_win,
        "promoted": promote,
        "arena": arena,
        "manifest_before": dict(manifest),
    }
    if promote:
        # best_win_rate here is the candidate's rate vs the champion it beat,
        # recorded for history; each new champion resets the head-to-head baseline.
        manifest["best_win_rate"] = rate
        manifest["best_cycle"] = cycle
        manifest["last_promoted_at"] = utc_now()
    entry["manifest_after"] = dict(manifest)
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"cycle-{cycle}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    save_manifest(MANIFEST_PATH, manifest)
    return promote, entry


def main() -> None:
    p = argparse.ArgumentParser(description="train-loop v3 cycle helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init-cycle2", help="seed state from cycle-2 artifacts")
    init_p.add_argument("--best-win-rate", type=float, default=0.58)
    init_p.add_argument("--best-cycle", type=int, default=2)

    append_p = sub.add_parser("append-replay", help="append samples and trim replay buffer")
    append_p.add_argument("samples", type=Path)
    append_p.add_argument("--max-lines", type=int, default=50000)

    rec_p = sub.add_parser("record-cycle", help="write cycle history and update manifest")
    rec_p.add_argument("cycle", type=int)
    rec_p.add_argument("report", type=Path)
    rec_p.add_argument("--promote-win", type=float, default=0.55,
                       help="head-to-head win rate the candidate must clear to replace the champion")
    rec_p.add_argument("--history-dir", type=Path, default=Path(".tectonix/reports/training-history"))

    args = p.parse_args()
    if args.cmd == "init-cycle2":
        m = init_from_cycle2(best_win_rate=args.best_win_rate, best_cycle=args.best_cycle)
        print(json.dumps(m))
    elif args.cmd == "append-replay":
        print(append_replay(args.samples, max_lines=args.max_lines))
    elif args.cmd == "record-cycle":
        promote, _ = record_cycle(
            args.cycle,
            args.report,
            promote_win=args.promote_win,
            history_dir=args.history_dir,
        )
        print("promote" if promote else "reject")


if __name__ == "__main__":
    main()
