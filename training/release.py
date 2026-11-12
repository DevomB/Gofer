"""Champions and GitHub releases.

The training loop plays every anchor generation whose ladder rating passes the
current champion's against that champion, and promotes it only if the match's
SPRT accepts H1 (see internal/arena: with the default 0,50 bounds and
alpha = beta = 0.05, a challenger that is not actually stronger is promoted
with probability at most 5.3%; an undecided match promotes nothing). Each champion
is copied to runs/<run>/champion/ and, when a repository is given, published
as a GitHub Release whose model asset always has the same name, so the newest
champion is one link away:

    https://github.com/OWNER/REPO/releases/latest/download/gofer-9x9.onnx

Publishing uses the GitHub CLI on the training machine: `gh auth login`, or
GH_TOKEN set to a fine-grained token with "Contents: read and write" on the
repository. A release that fails to publish stays pending; the loop retries it,
and so does this command:

    python -m training.release --run runs/9x9 --repo DevomB/Gofer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def asset_name(size: int) -> str:
    return f"gofer-{size}x{size}.onnx"


def release_tag(run: Path, gen: int) -> str:
    return f"model-{run.name}-g{gen:04d}"


def won(match: dict) -> bool:
    """Whether the challenger (player A) proved stronger than the champion:
    only an SPRT acceptance counts, so the error bound holds."""
    return match.get("sprt") == "H1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _opponent(name: str) -> str:
    if name.startswith("gen-"):
        return f"generation {int(name[4:])}"
    if name.startswith("gnugo-l"):
        return f"GNU Go level {name[7:]}"
    return name


def notes(info: dict, repo: str) -> str:
    """Release notes for a champion described by info (see Run.promote)."""
    size = info["board_size"]
    asset = asset_name(size)
    lines = [
        f"Generation {info['gen']} of training run `{info['run']}` is the new champion.",
        "",
        "| | |",
        "|---|---|",
    ]
    m = info.get("match")
    if m:
        lines.append(f"| Champion match | beat {_opponent(m['opponent'])} {m['wins']}–{m['losses']}"
                     f"{'–' + str(m['draws']) if m['draws'] else ''} (Elo {m['elo']:+.0f} "
                     f"[{m['elo_ci_low']:+.0f}, {m['elo_ci_high']:+.0f}]) at {m['visits']} visits |")
        if m.get("sprt_pairs"):
            lines.append(f"| Evidence | SPRT accepted after {m['sprt_pairs']} opening pairs: "
                         f"{m['sprt_sweeps_a']} sweeps to {m['sprt_sweeps_b']} (split pairs carry no evidence) |")
    else:
        lines.append("| Champion match | first champion of the run |")
    lines.append(f"| Rating | Elo {info['elo']:+.0f} ± {info['stderr']:.0f} on the run's ladder (generation 0 = 0) |")
    if info.get("anchors"):
        anchors = ", ".join(f"{_opponent(a['b'])} {a['wins']}–{a['losses']}" for a in info["anchors"])
        lines.append(f"| Latest anchor results | {anchors} |")
    lines += [
        f"| Network | {info['blocks']} blocks × {info['channels']} channels, {size}×{size}, komi {info['komi']} |",
        f"| Training so far | {info['gen']} generations, {info['games']:,} self-play games, {info['positions']:,} positions"
        f"{', on top of the network this run started from (`' + Path(info['init_from']).name + '`)' if info.get('init_from') else ''} |",
        f"| SHA-256 | `{info['sha256']}` |",
        "",
        f"Play it: `bin/gofer play -model {asset}` or `bin/gofer gtp -model {asset}` (GTP, for Sabaki or Lizzie).",
    ]
    if repo:
        lines.append(f"Always the newest champion: https://github.com/{repo}/releases/latest/download/{asset}")
    return "\n".join(lines) + "\n"


def publish(repo: str, tag: str, title: str, notes_file: Path, assets: list[Path],
            log: Callable[[str], None] = print, gh: str = "gh") -> bool:
    """Create the GitHub Release (or find it already there). Returns success."""
    if shutil.which(gh) is None:
        log(f"release {tag}: GitHub CLI '{gh}' not found; install it and run gh auth login (release kept pending)")
        return False
    view = subprocess.run([gh, "release", "view", tag, "--repo", repo], capture_output=True, text=True)
    if view.returncode == 0:
        log(f"release {tag}: already on GitHub")
        return True
    cmd = [gh, "release", "create", tag, *map(str, assets), "--repo", repo, "--title", title,
           "--notes-file", str(notes_file), "--latest"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        log(f"release {tag}: gh failed (kept pending): {(out.stderr or out.stdout).strip()}")
        return False
    log(f"release {tag}: published {out.stdout.strip()}")
    return True


def publish_pending(run: Path, state: dict, repo: str, log: Callable[[str], None] = print, gh: str = "gh") -> bool:
    """Publish the newest unpublished champion; older unpublished ones are
    superseded rather than released out of order. Returns whether state changed."""
    pending = [r for r in state.get("releases", []) if not r.get("published") and not r.get("superseded")]
    if not repo or not pending:
        return False
    newest = max(pending, key=lambda r: r["gen"])
    for r in pending:
        if r is not newest:
            r["superseded"] = True
    d = run / "champion" / f"gen-{newest['gen']:04d}"
    info = json.loads((d / "champion.json").read_text())
    notes_file = d / "notes.md"
    notes_file.write_text(notes(info, repo))
    title = f"Gofer {info['board_size']}×{info['board_size']} · generation {info['gen']} · Elo {info['elo']:+.0f}"
    if publish(repo, newest["tag"], title, notes_file, [d / asset_name(info["board_size"]), d / "champion.json"], log, gh):
        newest["published"] = True
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="training run directory")
    ap.add_argument("--repo", default="", help="OWNER/REPO to publish the newest pending champion to")
    args = ap.parse_args()
    state_path = args.run / "state.json"
    state = json.loads(state_path.read_text())
    champ = state.get("champion")
    print(f"champion: generation {champ['gen']}, Elo {champ['elo']:+.0f}" if champ else "no champion yet")
    for r in state.get("releases", []):
        status = "published" if r.get("published") else "superseded" if r.get("superseded") else "pending"
        print(f"  {r['tag']}: {status}")
    if args.repo and publish_pending(args.run, state, args.repo):
        state_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
