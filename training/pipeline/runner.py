"""The training loop: self-play -> train -> export -> gate -> promote, per cycle.

Differences from the v3 shell loop (deleted Sep 2026; see ADR 0007):
  * self-play writes compact .npz shards (no JSONL parse, ~40x smaller);
  * the replay buffer is a growing window over immutable shards (no rewrites);
  * gating is a sequential test (SPRT) run in arena batches: it controls the
    false-promotion rate that v3's per-game Wilson checks inflated, and still
    stops early on clear wins and clear losses;
  * the next cycle's self-play runs in the background while this cycle trains
    and gates (CPU self-play overlaps GPU training);
  * every stage is checkpointed in state.json, so a crash or preemption (spot
    instances) resumes at the first unfinished stage;
  * champions are immutable, numbered generations with an Elo ladder.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from training.pipeline import replay_index, stats
from training.pipeline.config import PipelineConfig, dump_toml
from training.pipeline.procs import CommandError, Executor, Job, ensure_ort_library
from training.pipeline.state import (
    Generation,
    append_event,
    load_state,
    save_state,
    utc_now,
    write_json_atomic,
)


class _Bundle:
    """A job plus the sidecars it depends on; waiting also stops the sidecars."""

    def __init__(self, job: Job, sidecars: list[Job]) -> None:
        self.job = job
        self.sidecars = sidecars

    def poll(self) -> bool:
        return self.job.poll()

    def wait(self) -> None:
        try:
            self.job.wait()
        except BaseException:
            self.job.terminate()   # an interrupt here must not orphan background self-play
            raise
        finally:
            self.terminate_sidecars()

    def terminate(self) -> None:
        self.job.terminate()
        self.terminate_sidecars()

    def terminate_sidecars(self) -> None:
        for s in self.sidecars:
            s.terminate()
        self.sidecars = []


class Pipeline:
    def __init__(self, cfg: PipelineConfig, executor: Executor, repo_root: Path, *, out: Callable[[str], None] = print) -> None:
        self.cfg = cfg
        self.x = executor
        self.root = repo_root
        self.out = out
        run_dir = cfg.run_dir if cfg.run_dir.is_absolute() else repo_root / cfg.run_dir
        self.dir = run_dir
        self.selfplay_dir = run_dir / "selfplay"
        self.train_dir = run_dir / "train"
        self.models_dir = run_dir / "models"
        self.gating_dir = run_dir / "gating"
        self.history_dir = run_dir / "history"
        self.logs_dir = run_dir / "logs"
        self.state_path = run_dir / "state.json"
        self.events_path = run_dir / "events.jsonl"
        self.state = load_state(self.state_path)
        self.env: dict[str, str] = {}
        self.prefetch: dict[int, _Bundle] = {}

    # ---------------------------------------------------------------- setup

    def log(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        self.out(line)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with (self.logs_dir / "pipeline.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def prepare(self, *, build: bool | None = None) -> None:
        for d in (self.selfplay_dir, self.train_dir, self.models_dir, self.gating_dir, self.history_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        saved = self.dir / "config.toml"
        if saved.exists() and self.state.cycle > 0:
            # Preserve the previous config when a resumed run changes it.
            before = saved.read_text(encoding="utf-8")
            after = dump_toml(self.cfg)
            if before != after:
                self.log(
                    f"WARNING: {saved} differs from the config being run. Cycles 1-{self.state.cycle} "
                    f"were produced under the saved one. Cached arena reports whose config hash no "
                    f"longer matches will be replayed; shards and the champion will not."
                )
                (self.dir / "config-previous.toml").write_text(before, encoding="utf-8")
        saved.write_text(dump_toml(self.cfg), encoding="utf-8")
        for stale in list(self.selfplay_dir.glob("*.npz.tmp")) + list((self.selfplay_dir / "pending").glob("*.npz.tmp")):
            stale.unlink()
        eng = self.cfg.engine
        if eng.backend == "inprocess":
            lib = eng.ort_lib or os.environ.get("ONNXRUNTIME_SHARED_LIBRARY_PATH", "")
            if not lib:
                lib = str(ensure_ort_library(self.root / ".tectonix" / "artifacts"))
            self.env["ONNXRUNTIME_SHARED_LIBRARY_PATH"] = lib
        if eng.build if build is None else build:
            self.build_engine()
        elif not self._abs(self.cfg.gofer_bin()).exists():
            raise CommandError(f"engine binary {self.cfg.gofer_bin()} missing and engine.build=false")
        save_state(self.state_path, self.state)

    def build_engine(self) -> None:
        cmd = ["go", "build", "-o", str(self.cfg.gofer_bin())]
        env = {"CGO_ENABLED": "0"}
        if self.cfg.engine.backend == "inprocess":
            cmd.insert(2, "-tags=onnx")
            env["CGO_ENABLED"] = "1"
        self.log(f"build engine ({self.cfg.engine.backend})")
        self.x.run(cmd + ["./cmd/gofer"], log=self.logs_dir / "build.log", env=env)

    def _abs(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    # ----------------------------------------------------------------- loop

    def run(self, *, max_cycles: int | None = None) -> int:
        """Run cycles until max_cycles / deadline. Returns the number of cycles completed."""
        limit = self.cfg.run.max_cycles if max_cycles is None else max_cycles
        deadline = time.time() + self.cfg.run.deadline_hours * 3600 if self.cfg.run.deadline_hours > 0 else None
        cycle = self.state.in_progress.cycle if self.state.in_progress else self.state.cycle + 1
        done = 0
        try:
            while True:
                if limit and done >= limit:
                    break
                if deadline and time.time() >= deadline:
                    self.log("deadline reached")
                    break
                last = bool(limit and done + 1 >= limit) or bool(deadline and time.time() >= deadline)
                self.run_cycle(cycle, prefetch_next=not last)
                done += 1
                cycle += 1
        finally:
            for bundle in self.prefetch.values():
                bundle.terminate()
            self.prefetch.clear()
        return done

    def run_cycle(self, cycle: int, *, prefetch_next: bool = False) -> dict[str, Any]:
        st = self.state
        st.begin_cycle(cycle)
        save_state(self.state_path, st)
        champ = st.champion
        self.log(f"=== cycle {cycle} (champion: {'gen %d' % champ.generation if champ else 'none, heuristic bootstrap'})")

        if not st.is_done("selfplay"):
            self._timed(cycle, "selfplay", self.stage_selfplay)
        if prefetch_next and self.cfg.run.overlap_selfplay and cycle + 1 not in self.prefetch:
            if not self._shard_path(cycle + 1).exists() and not self._pending_path(cycle + 1).exists():
                self.log(f"prefetch self-play for cycle {cycle + 1} in background")
                self.prefetch[cycle + 1] = self._spawn_selfplay(cycle + 1, self._pending_path(cycle + 1))

        if st.lifetime_rows < self.cfg.replay.min_rows_to_train:
            self.log(f"only {st.lifetime_rows} rows < min_rows_to_train={self.cfg.replay.min_rows_to_train}; skip training this cycle")
            for stage in ("train", "export", "gate", "promote", "publish"):
                st.mark(stage, skipped=True)
        else:
            for stage, fn in (("train", self.stage_train), ("export", self.stage_export), ("gate", self.stage_gate),
                              ("promote", self.stage_promote), ("publish", self.stage_publish)):
                if not st.is_done(stage):
                    self._timed(cycle, stage, fn)

        assert st.in_progress is not None
        record = {"cycle": cycle, "finished_at": utc_now(), **st.in_progress.data}
        write_json_atomic(self.history_dir / f"cycle-{cycle:04d}.json", record)
        st.finish_cycle()
        save_state(self.state_path, st)
        self.prune()
        append_event(self.events_path, cycle=cycle, stage="cycle", status="done", promoted=record.get("promoted", False))
        return record

    def prune(self) -> None:
        """Bound disk use for long runs / CI caches. Champions live in models/gen-*, never pruned."""
        keep = self.cfg.run.keep_train_dirs
        if keep > 0:
            for d in sorted(self.train_dir.glob("cycle-*"))[:-keep]:
                shutil.rmtree(d, ignore_errors=True)
        keep = self.cfg.run.keep_candidates
        if keep > 0:
            for f in sorted(self.models_dir.glob("candidate-*.onnx"))[:-keep]:
                f.unlink(missing_ok=True)

    def _timed(self, cycle: int, stage: str, fn: Callable[[int], dict[str, Any]]) -> None:
        t0 = time.time()
        append_event(self.events_path, cycle=cycle, stage=stage, status="start")
        try:
            info = fn(cycle) or {}
        except Exception as e:
            append_event(self.events_path, cycle=cycle, stage=stage, status="error", seconds=time.time() - t0, error=str(e)[:500])
            raise
        seconds = time.time() - t0
        self.state.mark(stage, **{f"{stage}_seconds": round(seconds, 2)}, **info)
        save_state(self.state_path, self.state)
        append_event(self.events_path, cycle=cycle, stage=stage, status="done", seconds=round(seconds, 2), **_scalars(info))
        self.log(f"cycle {cycle} {stage} done in {seconds:.1f}s")

    # ------------------------------------------------------------- selfplay

    def _shard_path(self, cycle: int) -> Path:
        return self.selfplay_dir / f"cycle-{cycle:04d}.npz"

    def _pending_path(self, cycle: int) -> Path:
        """Where background self-play writes before its cycle starts.

        Prefetched games must not sit in selfplay/ while an earlier cycle trains:
        the trainer reads the whole directory newest-first, so a finished
        prefetch would displace the very data the cycle is meant to learn from,
        and would do it depending on wall-clock timing.
        """
        return self.selfplay_dir / "pending" / f"cycle-{cycle:04d}.npz"

    def _eval_args(self, model1: Path, model2: Path | None, ports: tuple[int, int]) -> list[str]:
        if self.cfg.engine.backend == "inprocess":
            args = ["-eval-backend", "inprocess", "-model", str(model1)]
            if model2 is not None:
                args += ["-model-2", str(model2)]
            return args
        # Preserve model provenance and the arena configuration hash in sidecar mode.
        args = ["-eval-backend", "sidecar", "-onnx-url", f"http://127.0.0.1:{ports[0]}", "-model", str(model1)]
        if model2 is not None:
            args += ["-onnx-url-2", f"http://127.0.0.1:{ports[1]}", "-model-2", str(model2)]
        return args

    def _sidecars(self, models: list[Path], ports: list[int], tag: str) -> list[Job]:
        if self.cfg.engine.backend != "sidecar":
            return []
        started: list[Job] = []
        try:
            for m, p in zip(models, ports):
                started.append(self.x.sidecar(self.cfg.python(), m, p, log=self.logs_dir / f"sidecar-{tag}-{p}.log"))
        except Exception:
            for s in started:
                s.terminate()
            raise
        return started

    def selfplay_command(self, cycle: int, out_path: Path | None = None) -> list[str]:
        sp = self.cfg.selfplay
        champ = self.state.champion
        games = sp.games_per_cycle if champ else (sp.bootstrap_games or sp.games_per_cycle)
        cmd = [
            str(self.cfg.gofer_bin()), "-selfplay",
            "-games", str(games), "-size", str(sp.board_size), "-komi", str(sp.komi),
            "-playouts", str(sp.full_playouts),
            "-selfplay-full-playouts", str(sp.full_playouts),
            "-selfplay-fast-playouts", str(sp.fast_playouts),
            "-selfplay-cap-randomize-p", str(sp.cap_randomize_p),
            "-selfplay-temp-moves", str(sp.temp_moves),
            "-selfplay-parallel", str(self.cfg.selfplay_parallel()),
            "-batch-size", str(self.cfg.engine.batch_size or self.cfg.selfplay_parallel()),
            "-seed", str(self.cfg.run.seed + cycle * 100_003),
            "-o", str(out_path or self._shard_path(cycle)),
        ]
        if champ is None:
            return cmd + ["-selfplay-eval", "heuristic"]
        port = self.cfg.engine.sidecar_base_port
        return cmd + [
            "-selfplay-eval", sp.eval,
            "-selfplay-onnx-fraction", str(sp.onnx_fraction),
            "-eval-timeout", self.cfg.engine.eval_timeout,
        ] + self._eval_args(self._abs(Path(champ.onnx)), None, (port, port))

    def _spawn_selfplay(self, cycle: int, out_path: Path | None = None) -> _Bundle:
        champ = self.state.champion
        port = self.cfg.engine.sidecar_base_port
        sidecars = self._sidecars([self._abs(Path(champ.onnx))], [port], "selfplay") if champ else []
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            job = self.x.spawn(self.selfplay_command(cycle, out_path), log=self.logs_dir / f"selfplay-{cycle:04d}.log", env=self.env)
        except Exception:
            for s in sidecars:
                s.terminate()
            raise
        return _Bundle(job, sidecars)

    def stage_selfplay(self, cycle: int) -> dict[str, Any]:
        path = self._shard_path(cycle)
        bundle = self.prefetch.get(cycle)
        if bundle is not None:
            self.log(f"waiting for prefetched self-play of cycle {cycle}")
            try:
                bundle.wait()
            finally:
                # Popped only after waiting, so an interrupt during the wait still
                # finds the job in self.prefetch and terminates it.
                self.prefetch.pop(cycle, None)
        elif not path.exists() and not self._pending_path(cycle).exists():
            self._spawn_selfplay(cycle).wait()
        pending = self._pending_path(cycle)
        if pending.exists() and not path.exists():
            # This cycle's turn has come: its prefetched games join the replay
            # buffer now, never while an earlier cycle was training.
            pending.replace(path)
        if not path.exists():
            raise CommandError(f"self-play finished but {path} is missing")
        meta = replay_index.read_shard_meta(path)
        rows = int(meta["rows"])
        self.state.lifetime_rows += rows
        return {"selfplay_rows": rows, "selfplay_games": int(meta.get("games", 0)), "selfplay_model": meta.get("model", ""), "lifetime_rows": self.state.lifetime_rows}

    # ---------------------------------------------------------------- train

    def _warm_start(self, cycle: int) -> Path | None:
        """The checkpoint this cycle's training starts from.

        Under ``train.warm_start = "champion"`` that is the last promoted net, so
        a rejected candidate's training is thrown away and the next cycle starts
        again from the same place. Under ``"latest"`` it is the previous cycle's
        checkpoint, promoted or not, so training accumulates across rejections.

        Only the initialisation changes. The champion still plays self-play,
        still defends the gate, and still changes only by winning one.
        """
        if self.cfg.train.warm_start == "latest":
            for c in range(cycle - 1, 0, -1):
                prev = self.train_dir / f"cycle-{c:04d}" / "best.pt"
                if prev.exists():
                    return prev
        champ = self.state.champion
        if champ:
            return self._abs(Path(champ.pt))
        return self._abs(Path(self.cfg.run.init_checkpoint)) if self.cfg.run.init_checkpoint else None

    def stage_train(self, cycle: int) -> dict[str, Any]:
        tc = self.cfg.train
        moved = replay_index.archive_old(self.selfplay_dir, self.cfg.replay, self.state.lifetime_rows)
        if moved:
            self.log(f"archived {len(moved)} old shard(s) outside the replay window")
        window = replay_index.window_rows(self.state.lifetime_rows, self.cfg.replay)
        out = self.train_dir / f"cycle-{cycle:04d}"
        init = self._warm_start(cycle)
        fresh = init is None
        cmd = [
            self.cfg.python(), tc.script,
            "--data", str(self.selfplay_dir),
            "--out-dir", str(out),
            "--window-rows", str(window),
        ]
        if "config" not in tc.args:
            # With a trainer preset (train.args.config) the preset owns the schedule;
            # explicit CLI flags would override it.
            cmd += [
                "--epochs", str(tc.epochs_fresh if fresh else tc.epochs_resume),
                "--lr", str(tc.lr_fresh if fresh else tc.lr_resume),
                "--patience", str(tc.patience),
            ]
        if self.cfg.replay.window_decay > 0:
            cmd += ["--window-decay", str(self.cfg.replay.window_decay)]
        if init is not None:
            cmd += ["--init-from", str(init)]
        if (out / "last.pt").exists():
            cmd.append("--continue")  # a previous attempt was interrupted (crash / spot preemption)
        cmd += trainer_flags(tc.args)
        self.log(f"train on window={window} rows (lifetime {self.state.lifetime_rows}) init={'fresh' if fresh else init.name}")
        self.x.run(cmd, log=self.logs_dir / f"train-{cycle:04d}.log", env=self.env)
        best = out / "best.pt"
        if not best.exists():
            raise CommandError(f"trainer did not write {best}")
        summary_path = out / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        return {"window_rows": window, "train_dir": self._rel(out), "train_summary": summary}

    def _rel(self, p: Path) -> str:
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    # --------------------------------------------------------------- export

    def _candidate_path(self, cycle: int) -> Path:
        return self.models_dir / f"candidate-{cycle:04d}.onnx"

    def stage_export(self, cycle: int) -> dict[str, Any]:
        best = self.train_dir / f"cycle-{cycle:04d}" / "best.pt"
        cand = self._candidate_path(cycle)
        self.x.run([self.cfg.python(), self.cfg.train.export_script, "--checkpoint", str(best), "--out", str(cand)],
                   log=self.logs_dir / f"export-{cycle:04d}.log", env=self.env)
        if not cand.exists():
            raise CommandError(f"export did not write {cand}")
        return {"candidate": self._rel(cand)}

    # ----------------------------------------------------------------- gate

    def arena_command(self, games: int, seed: int, report: Path, baseline: Path | None, challenger: Path) -> list[str]:
        """Arena: challenger (white-eval, reported as wins_challenger) vs baseline; None = heuristic."""
        g = self.cfg.gating
        sp = self.cfg.selfplay
        base = self.cfg.engine.sidecar_base_port
        cmd = [
            str(self.cfg.gofer_bin()), "-arena",
            "-games", str(games), "-size", str(sp.board_size), "-komi", str(sp.komi),
            "-playouts", str(g.playouts),
            "-arena-parallel", str(self.cfg.gating_parallel()),
            "-batch-size", str(self.cfg.engine.batch_size or self.cfg.gating_parallel()),
            "-arena-opening-moves", str(g.opening_moves),
            "-eval-timeout", self.cfg.engine.eval_timeout,
            "-arena-enhanced", "none",
            # The engine has its own in-match promotion stop, which fires on the same
            # win/loss stream this batch feeds to the SPRT. Two stopping rules stacked
            # on one stream is not a test with known error rates, so batches play out.
            "-arena-play-all",
            "-seed", str(seed),
            "-json", str(report),
        ]
        if baseline is None:
            return cmd + ["-black-eval", "heuristic", "-white-eval", "onnx"] + self._eval_args(challenger, None, (base + 1, base + 1))
        return cmd + ["-black-eval", "onnx", "-white-eval", "onnx2"] + self._eval_args(baseline, challenger, (base + 1, base + 2))

    def _arena_config_hash(self, report: Path, games: int, seed: int, baseline: Path | None, challenger: Path) -> str | None:
        """What hash would this arena stamp on its report right now?

        Asked of the engine rather than derived here: the hash covers the model
        file contents and the backend, so only the binary that would run the
        match can answer. Returns None if the engine cannot say, which leaves the
        game-count check as the only gate rather than discarding usable evidence
        over a failed subprocess.
        """
        cmd = self.arena_command(games, seed, report, baseline, challenger) + ["-arena-config-hash"]
        try:
            out = self.x.capture(cmd, env=self.env)
        except (CommandError, OSError) as exc:
            # Only environmental failures degrade to the game-count check. A
            # TypeError or AttributeError here is a bug in this method, and
            # swallowing it would leave the hash check silently never running -
            # which is how this was first shipped.
            self.log(f"could not read arena config hash ({exc}); falling back to game-count check only")
            return None
        return out.strip().splitlines()[-1].strip() if out.strip() else None

    def run_match(self, report: Path, games: int, seed: int, baseline: Path | None, challenger: Path, *, log_name: str) -> dict:
        """Play one arena match, or reuse its report: matches are idempotent across resumes."""
        if report.exists():
            expected = self._arena_config_hash(report, games, seed, baseline, challenger)
            if not self._usable_report(report, games, expected):
                report.unlink(missing_ok=True)
        if not report.exists():
            base = self.cfg.engine.sidecar_base_port
            models = [challenger] if baseline is None else [baseline, challenger]
            sidecars = self._sidecars(models, [base + 1, base + 2], log_name)
            try:
                self.x.run(self.arena_command(games, seed, report, baseline, challenger),
                           log=self.logs_dir / f"{log_name}.log", env=self.env)
            finally:
                for s in sidecars:
                    s.terminate()
        return json.loads(report.read_text(encoding="utf-8"))

    def _usable_report(self, report: Path, games: int, config_hash: str | None = None) -> bool:
        """Is a cached arena report complete, and evidence for *this* configuration?

        A crash while the engine wrote the report leaves a truncated file, and a
        report from a larger batch must not be reused after the configuration
        shrank; either way the match is replayed. Arenas may stop early, so the
        recorded game count is only required not to exceed what we asked for.

        Game count alone is not enough. A resumed run can reach this with a
        different candidate, backend or opening setting, and a report from the
        old one has the right shape and the wrong contents. The engine stamps
        every report with a hash of everything that changes what it measures, so
        when the caller knows the hash it must match.
        """
        try:
            rep = json.loads(report.read_text(encoding="utf-8"))
            played = int(rep.get("game_count", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            self.log(f"discarding unreadable arena report {report.name}; replaying that match")
            return False
        if not 0 < played <= games:
            self.log(f"discarding arena report {report.name}: {played} games recorded, {games} requested")
            return False
        if config_hash is not None:
            got = rep.get("config_hash", "")
            if got != config_hash:
                self.log(
                    f"discarding arena report {report.name}: config hash {got or 'absent'} "
                    f"!= {config_hash}; it measured a different configuration"
                )
                return False
        return True

    def _run_arena(self, cycle: int, batch: int, games: int, report: Path, *, vs_heuristic: bool) -> dict:
        champ = self.state.champion
        baseline = None if vs_heuristic else self._abs(Path(champ.onnx))
        return self.run_match(report, games, self.cfg.run.seed + cycle * 1000 + batch, baseline,
                              self._candidate_path(cycle), log_name=f"gate-{cycle:04d}")

    def _anchor(self, cycle: int, gdir: Path, decision: dict[str, Any]) -> None:
        """Measure this cycle's candidate against the heuristic, not the champion.

        The SPRT compares each challenger to the champion it would replace, which
        is a ladder where every rung is measured against the rung below it: it can
        report steady promotions while going nowhere absolute. This replays the
        seed gate's match -- same evaluators, same flags, same report shape -- so
        generation N's score is directly comparable to generation 1's, and the run
        carries its own evidence of whether it is going anywhere.

        Recorded under the same keys the seed gate uses, so anything reading
        generation 1's ``vs_heuristic_elo`` finds later ones without changing.
        """
        g = self.cfg.gating
        if g.anchor_every <= 0 or cycle % g.anchor_every:
            return
        games = g.anchor_games - g.anchor_games % 2   # colours alternate in pairs
        if games <= 0:
            return
        # Batch 800: clear of the SPRT batches, which count from 1, and of the
        # publish regression match at 900. Sharing a batch number would mean
        # sharing a seed, and two matches on the same openings are one sample.
        rep = self._run_arena(cycle, 800, games, gdir / "anchor-vs-heuristic.json", vs_heuristic=True)
        tally = stats.tally_from_arena(rep)
        elo = tally.elo()[0]
        decision.update({"vs_heuristic": tally.to_dict(), "vs_heuristic_score": tally.score,
                         "vs_heuristic_games": tally.games, "vs_heuristic_elo": elo,
                         "vs_heuristic_kind": "anchor"})
        self.log(f"anchor cycle {cycle}: candidate vs heuristic score {tally.score:.3f} "
                 f"({elo:+.0f} Elo) over {tally.games} games")

    def stage_gate(self, cycle: int) -> dict[str, Any]:
        g = self.cfg.gating
        gdir = self.gating_dir / f"cycle-{cycle:04d}"
        gdir.mkdir(parents=True, exist_ok=True)
        if self.state.champion is None:
            decision = self._seed_gate(cycle, gdir)
        else:
            decision = self._sprt_gate(cycle, gdir)
            self._anchor(cycle, gdir, decision)
        if g.mode == "hold":
            decision["promote"] = False
            decision["reason"] += " (gating.mode=hold: not promoted)"
        write_json_atomic(gdir / "decision.json", decision)
        return {"gate": decision}

    def _sprt_batches(self, cycle: int, gdir: Path, *, vs_heuristic: bool, batch_games: int,
                      prefix: str, label: str) -> tuple[stats.MatchTally, str, float, list]:
        """Play batches against one opponent until the SPRT decides or the cap.

        Shared by the seed gate and the champion gate so the first network faces
        the same test as every later one. The only differences are the opponent
        and the batch size.
        """
        g = self.cfg.gating
        tally = stats.MatchTally()
        steps = []
        verdict, llr = stats.CONTINUE, 0.0
        batch = 0
        while tally.games < g.max_games:
            batch += 1
            games = min(batch_games, g.max_games - tally.games)
            games -= games % 2
            if games <= 0:
                break
            rep = self._run_arena(cycle, batch, games, gdir / f"{prefix}-{batch:02d}.json",
                                  vs_heuristic=vs_heuristic)
            tally = tally.add(stats.tally_from_arena(rep))
            verdict, llr = stats.sprt_decision(tally.wins, tally.losses, tally.draws, elo0=g.elo0, elo1=g.elo1, alpha=g.alpha, beta=g.beta)
            steps.append({"batch": batch, "games": tally.games, "score": round(tally.score, 4), "llr": round(llr, 4), "verdict": verdict})
            self.log(f"{label} cycle {cycle} batch {batch}: {tally.wins}-{tally.losses}-{tally.draws} "
                     f"score={tally.score:.3f} LLR={llr:+.2f} -> {verdict}")
            if verdict != stats.CONTINUE:
                break
        return tally, verdict, llr, steps

    def _seed_gate(self, cycle: int, gdir: Path) -> dict[str, Any]:
        """Decide whether the first network is fit to become the teacher.

        It is not enough for it to be recorded, and not enough for it to win a
        single 40-game match. The champion generates ``selfplay.onnx_fraction``
        of every shard from the next cycle on, so a seed that is merely
        *probably* better poisons the replay window and no later gate can undo
        it: gates protect the champion from replacement, nothing protects the
        data.

        A fixed threshold over one batch is the wrong instrument for that. At 40
        games, "score >= 0.5" admits an evenly-matched network about half the
        time and a genuinely -50 Elo one about a fifth of the time. So the seed
        faces the same sequential test every later challenger faces, against the
        heuristic instead of against a champion, and only an accepted H1 seeds.

        The asymmetry is deliberate and cheap in the right direction: refusing a
        good seed costs one more cycle of heuristic self-play, which is the data
        the run wants anyway; accepting a bad one costs every cycle after it.
        """
        g = self.cfg.gating
        tally, verdict, llr, steps = self._sprt_batches(
            cycle, gdir, vs_heuristic=True, batch_games=g.bootstrap_games,
            prefix="vs-heuristic", label="seed")
        lo, hi = stats.wilson(tally.wins + 0.5 * tally.draws, tally.games)
        seeds = verdict == stats.ACCEPT
        if seeds:
            reason = (f"first network beat the heuristic: SPRT accepted H1 (elo>={g.elo1:g}) "
                      f"after {tally.games} games")
        elif verdict == stats.REJECT:
            reason = (f"first network is not better than the heuristic: SPRT accepted H0 "
                      f"(elo<={g.elo0:g}) after {tally.games} games; keeping the heuristic as "
                      f"the self-play teacher")
        else:
            reason = (f"first network inconclusive against the heuristic at {tally.games} games "
                      f"(score {tally.score:.3f}, wilson_low {lo:.3f}); keeping the heuristic as "
                      f"the self-play teacher rather than seeding on a maybe")
        # Flattened as well as nested: the lineage keeps scalars only, and this
        # match is the only evidence recorded about generation 1's strength.
        decision = {"kind": "seed", "promote": seeds, "would_promote": seeds, "reason": reason,
                    "verdict": verdict, "llr": llr, "llr_bounds": list(stats.sprt_bounds(g.alpha, g.beta)),
                    "wilson_low": lo, "wilson_high": hi, "steps": steps,
                    "vs_heuristic": tally.to_dict(),
                    "vs_heuristic_score": tally.score, "vs_heuristic_games": tally.games,
                    "vs_heuristic_elo": tally.elo()[0]}
        verb = "seed champion" if seeds else "REFUSED to seed"
        self.log(f"gate cycle {cycle}: {verb} (vs heuristic score {tally.score:.3f} "
                 f"over {tally.games} games, {tally.elo()[0]:+.0f} Elo)")
        return decision

    def _sprt_gate(self, cycle: int, gdir: Path) -> dict[str, Any]:
        g = self.cfg.gating
        tally, verdict, llr, steps = self._sprt_batches(
            cycle, gdir, vs_heuristic=False, batch_games=g.batch_games, prefix="batch", label="gate")
        lo, hi = stats.wilson(tally.wins + 0.5 * tally.draws, tally.games)
        if verdict == stats.ACCEPT:
            would, reason = True, f"SPRT accepted H1 (elo>={g.elo1:g}) after {tally.games} games"
        elif verdict == stats.REJECT:
            would, reason = False, f"SPRT accepted H0 (elo<={g.elo0:g}) after {tally.games} games"
        else:
            would = tally.score >= g.promote_win and lo > 0.5
            reason = f"SPRT inconclusive at {tally.games} games; fallback score>={g.promote_win} and wilson_low>0.5 -> {would}"
        bounds = stats.sprt_bounds(g.alpha, g.beta)
        return {"kind": "sprt", "promote": would, "would_promote": would, "reason": reason, "verdict": verdict,
                "llr": llr, "llr_bounds": list(bounds), "wilson_low": lo, "wilson_high": hi, "steps": steps, **tally.to_dict()}

    # -------------------------------------------------------------- promote

    def stage_promote(self, cycle: int) -> dict[str, Any]:
        assert self.state.in_progress is not None
        decision = self.state.in_progress.data.get("gate", {})
        if not decision.get("promote"):
            champ = self.state.champion
            self.log(f"cycle {cycle}: REJECT candidate ({decision.get('reason', '')}); champion stays gen {champ.generation if champ else '-'}")
            return {"promoted": False}
        prev = self.state.champion
        # max+1, not champion+1: after a rollback the champion is not the newest generation.
        gen = max((g.generation for g in self.state.generations), default=0) + 1
        onnx = self.models_dir / f"gen-{gen:04d}.onnx"
        pt = self.models_dir / f"gen-{gen:04d}.pt"
        shutil.copy2(self._candidate_path(cycle), onnx)
        shutil.copy2(self.train_dir / f"cycle-{cycle:04d}" / "best.pt", pt)
        ckpt = self.train_dir / f"cycle-{cycle:04d}" / "best.ckpt"
        if ckpt.exists():
            shutil.copy2(ckpt, self.models_dir / f"gen-{gen:04d}.ckpt")
        # Elo ladder: generation 1 anchors at 0; each promotion adds its measured gain.
        elo = (prev.elo + float(decision.get("elo", 0.0))) if prev else 0.0
        self.state.generations.append(Generation(gen, cycle, self._rel(onnx), self._rel(pt), round(elo, 1), utc_now(), gate=_scalars(decision)))
        self.log(f"cycle {cycle}: PROMOTE -> generation {gen} (ladder Elo {elo:+.0f})")
        return {"promoted": True, "generation": gen, "elo": round(elo, 1)}


    # -------------------------------------------------------------- publish

    def stage_publish(self, cycle: int) -> dict[str, Any]:
        from training.pipeline.publish import Publisher

        assert self.state.in_progress is not None
        data = self.state.in_progress.data
        if not data.get("promoted") or not self.cfg.publish.enabled:
            return {"published": False}
        return Publisher(self).publish_champion(cycle, generation=data.get("generation"))


def trainer_flags(args: dict[str, Any]) -> list[str]:
    """{batch_size: 256, amp: true, compile: false} -> ['--batch-size', '256', '--amp'].

    TOML keys are snake_case like every other key in the configs; the trainer's
    own flags are kebab, so they are translated here rather than in the config.
    """
    out: list[str] = []
    for key, val in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                out.append(flag)
        elif isinstance(val, (list, tuple)):
            out += [flag, *map(str, val)]
        else:
            out += [flag, str(val)]
    return out


def _scalars(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool)) or v is None}


def summarize(state_path: Path) -> dict[str, Any]:
    st = load_state(state_path)
    champ = st.champion
    return {
        "cycle": st.cycle,
        "lifetime_rows": st.lifetime_rows,
        "in_progress": asdict(st.in_progress) if st.in_progress else None,
        "champion": asdict(champ) if champ else None,
        "generations": len(st.generations),
    }
