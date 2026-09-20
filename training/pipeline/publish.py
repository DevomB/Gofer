"""Champion publishing: registry of every best, stable alias, GitHub Releases.

Nothing here ever deletes or overwrites an older best:
  * each promoted generation is copied to <registry>/<prefix><run>-gen<NNNN>.onnx
    next to a .json card (gate stats, Elo, sha256), and listed in index.json;
  * index.json tracks `best` and `previous_best`, and `rollback` re-points them;
  * the optional alias (e.g. models/gofer-9x9-best.onnx) is only a convenience
    copy of the current best; the registry keeps the real history;
  * the optional GitHub Release is one release per generation, so older bests
    stay downloadable and `rollback` just re-marks an older release as latest.

A "false best" (a lucky gate, or a non-transitive gain that loses to older
champions) is guarded by an optional regression match against the champion
from `regression_lookback` generations back before anything is published.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from training.pipeline import stats
from training.pipeline.state import Generation, save_state, utc_now, write_json_atomic

if TYPE_CHECKING:
    from training.pipeline.runner import Pipeline

INDEX_FORMAT = "gofer-champions"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_index(registry: Path) -> dict[str, Any]:
    path = registry / "index.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"format": INDEX_FORMAT, "version": 1, "best": None, "previous_best": None, "entries": [], "log": []}


def save_index(registry: Path, index: dict[str, Any]) -> None:
    write_json_atomic(registry / "index.json", index)


def entry_key(run: str, generation: int) -> str:
    return f"{run}/gen{generation:04d}"


class GitHub:
    """Thin gh CLI wrapper; replaced in tests."""

    def __init__(self, repo: str, cwd: Path) -> None:
        self.repo = repo
        self.cwd = cwd

    def _gh(self, *args: str) -> subprocess.CompletedProcess:
        cmd = ["gh", *args] + (["--repo", self.repo] if self.repo else [])
        # check=False: callers inspect returncode (release_exists treats non-zero
        # as "no such release"), so a failure here is not exceptional.
        return subprocess.run(cmd, cwd=self.cwd, capture_output=True, text=True, check=False)

    def release_exists(self, tag: str) -> bool:
        return self._gh("release", "view", tag).returncode == 0

    def create_release(self, tag: str, title: str, notes: str, files: list[Path]) -> str:
        if not self.release_exists(tag):
            r = self._gh("release", "create", tag, *map(str, files), "--title", title, "--notes", notes, "--latest")
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or r.stdout.strip())
        r = self._gh("release", "view", tag, "--json", "url", "--jq", ".url")
        return r.stdout.strip()

    def mark_latest(self, tag: str) -> None:
        r = self._gh("release", "edit", tag, "--latest")
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or r.stdout.strip())


class Publisher:
    def __init__(self, pipe: "Pipeline", github: GitHub | None = None) -> None:
        self.pipe = pipe
        self.cfg = pipe.cfg.publish
        self.run_name = pipe.cfg.run.name
        self.registry = pipe._abs(Path(self.cfg.registry))
        self.github = github or GitHub(self.cfg.github_repo, pipe.root)

    def _stem(self, generation: int) -> str:
        return f"{self.cfg.release_prefix}{self.run_name}-gen{generation:04d}"

    def _generation(self, generation: int) -> Generation | None:
        """The newest lineage entry for this generation number."""
        return next((g for g in reversed(self.pipe.state.generations) if g.generation == generation), None)

    # ------------------------------------------------------------ regression

    def _regression_baseline(self, champ: Generation) -> Generation | None:
        """The newest generation at least `regression_lookback` behind the champion.

        Chosen by generation number, not by position: rollbacks and demotions
        append a duplicate entry carrying an older number, so the last matching
        element of the list can be far older than intended.
        """
        older = [g for g in self.pipe.state.generations
                 if g.generation <= champ.generation - self.cfg.regression_lookback]
        return max(older, key=lambda g: g.generation, default=None)

    def regression_check(self, cycle: int, champ: Generation) -> dict[str, Any] | None:
        if self.cfg.regression_games <= 0:
            return None
        base = self._regression_baseline(champ)
        if base is None:
            return None
        report = self.pipe.gating_dir / f"cycle-{cycle:04d}" / f"regression-vs-gen{base.generation:04d}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        rep = self.pipe.run_match(report, self.cfg.regression_games, self.pipe.cfg.run.seed + cycle * 1000 + 900,
                                  self.pipe._abs(Path(base.onnx)), self.pipe._abs(Path(champ.onnx)),
                                  log_name=f"regression-{cycle:04d}")
        tally = stats.tally_from_arena(rep)
        passed = tally.score >= self.cfg.regression_min_score
        self.pipe.log(f"regression gen {champ.generation} vs gen {base.generation}: score {tally.score:.3f} over "
                      f"{tally.games} games -> {'pass' if passed else 'FAIL'} (min {self.cfg.regression_min_score})")
        return {"vs_generation": base.generation, "passed": passed, "min_score": self.cfg.regression_min_score, **tally.to_dict()}

    # --------------------------------------------------------------- publish

    def publish_champion(self, cycle: int, generation: int | None = None) -> dict[str, Any]:
        """Publish the generation this cycle promoted.

        `generation` comes from the promote stage rather than from the current
        champion, so replaying an interrupted publish still refers to the
        generation under test even if a demotion already moved the champion.
        """
        st = self.pipe.state
        champ = self._generation(generation) if generation is not None else st.champion
        assert champ is not None
        reg = self.regression_check(cycle, champ)
        self.registry.mkdir(parents=True, exist_ok=True)
        index = load_index(self.registry)
        stem = self._stem(champ.generation)
        entry: dict[str, Any] = {
            "key": entry_key(self.run_name, champ.generation),
            "run": self.run_name,
            "generation": champ.generation,
            "cycle": cycle,
            "ladder_elo": champ.elo,
            "gate": champ.gate,
            "regression": reg,
            "created_at": utc_now(),
        }
        if reg is not None and not reg["passed"]:
            entry["status"] = "held"
            demote = self.cfg.regression_action == "demote" and st.champion is not None
            # Only demote while the failed generation is still champion: replaying
            # an interrupted publish must not promote it back over its replacement.
            if demote and st.champion.generation == champ.generation:
                prev = next((g for g in reversed(st.generations[:-1]) if g.generation != champ.generation), None)
                if prev is not None:
                    st.generations.append(replace(prev, cycle=cycle, promoted_at=utc_now(),
                                                  gate={"demoted": champ.generation, "reason": "failed regression check"}))
                    save_state(self.pipe.state_path, st)
                    entry["demoted_to"] = prev.generation
            elif demote:
                entry["demoted_to"] = st.champion.generation
            self.pipe.log(f"gen {champ.generation} HELD: not published (regression check failed)")
        else:
            onnx = self.registry / f"{stem}.onnx"
            shutil.copy2(self.pipe._abs(Path(champ.onnx)), onnx)
            entry.update(status="published", file=onnx.name, sha256=sha256_file(onnx))
            card = self.registry / f"{stem}.json"
            write_json_atomic(card, entry)
            if index.get("best") != entry["key"]:
                index["previous_best"], index["best"] = index.get("best"), entry["key"]
            self._update_alias(onnx)
            if self.cfg.github_release:
                self._release(entry, onnx, card)
            write_json_atomic(card, entry)
            self.pipe.log(f"published gen {champ.generation} -> {onnx.name}"
                          + (f" ({entry['release_url']})" if entry.get("release_url") else ""))
        index["entries"] = [e for e in index["entries"] if e.get("key") != entry["key"]] + [entry]
        save_index(self.registry, index)
        return {"published": entry["status"], "publish_key": entry["key"], "regression": reg,
                "release_url": entry.get("release_url", "")}

    def _update_alias(self, onnx: Path) -> None:
        if not self.cfg.alias:
            return
        alias = self.pipe._abs(Path(self.cfg.alias))
        alias.parent.mkdir(parents=True, exist_ok=True)
        tmp = alias.with_suffix(alias.suffix + ".tmp")
        shutil.copy2(onnx, tmp)
        tmp.replace(alias)

    def _release(self, entry: dict[str, Any], onnx: Path, card: Path) -> None:
        tag = self._stem(entry["generation"])
        gate = entry.get("gate") or {}
        notes = (f"Gofer 9x9 champion, run `{entry['run']}` generation {entry['generation']} (cycle {entry['cycle']}).\n\n"
                 f"- Ladder Elo vs generation 1: {entry['ladder_elo']:+.0f}\n"
                 f"- Gate: {gate.get('reason', gate.get('kind', 'n/a'))}\n"
                 f"- sha256: `{entry['sha256']}`\n\n"
                 "Older champions remain available as earlier releases.")
        try:
            entry["release_url"] = self.github.create_release(tag, f"Gofer {entry['run']} gen {entry['generation']}", notes, [onnx, card])
            entry["release_tag"] = tag
        except Exception as e:  # a failed upload must not stop training; retry with `publish --retry`
            entry["release_error"] = str(e)[:500]
            self.pipe.log(f"GitHub release for {tag} failed: {e}")


def retry_releases(pipe: "Pipeline", github: GitHub | None = None) -> list[str]:
    """Re-attempt GitHub releases that failed earlier (network, auth)."""
    pub = Publisher(pipe, github)
    index = load_index(pub.registry)
    done = []
    for entry in index["entries"]:
        if entry.get("run") != pub.run_name or entry.get("status") != "published" or entry.get("release_url"):
            continue
        stem = pub._stem(entry["generation"])
        entry.pop("release_error", None)
        pub._release(entry, pub.registry / f"{stem}.onnx", pub.registry / f"{stem}.json")
        write_json_atomic(pub.registry / f"{stem}.json", entry)
        if entry.get("release_url"):
            done.append(entry["key"])
    save_index(pub.registry, index)
    return done


def rollback(pipe: "Pipeline", generation: int, *, champion: bool, github: GitHub | None = None) -> dict[str, Any]:
    """Point `best` (alias, latest release) back at an earlier published generation.

    With champion=True the run also resumes self-play/training from that
    generation (appended to the lineage; nothing is removed).
    """
    pub = Publisher(pipe, github)
    index = load_index(pub.registry)
    key = entry_key(pub.run_name, generation)
    entry = next((e for e in index["entries"] if e.get("key") == key), None)
    if entry is None or entry.get("status") != "published":
        raise ValueError(f"{key} is not a published generation in {pub.registry / 'index.json'}")
    if index.get("best") != key:
        index["previous_best"], index["best"] = index.get("best"), key
    pub._update_alias(pub.registry / entry["file"])
    if pub.cfg.github_release and entry.get("release_tag"):
        pub.github.mark_latest(entry["release_tag"])
    index.setdefault("log", []).append({"ts": utc_now(), "action": "rollback", "to": key})
    save_index(pub.registry, index)
    if champion:
        st = pipe.state
        if st.in_progress is not None:
            raise ValueError("a cycle is in progress; finish or let it resume before changing the champion")
        target = next((g for g in st.generations if g.generation == generation), None)
        if target is None:
            raise ValueError(f"generation {generation} not in run state")
        st.generations.append(replace(target, cycle=st.cycle, promoted_at=utc_now(), gate={"rollback": True}))
        save_state(pipe.state_path, st)
    return {"best": index["best"], "previous_best": index["previous_best"], "champion_reset": champion}
