"""Orchestrator tests with a fake executor (no Go, no PyTorch)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from training.pipeline import replay_index, stats
from training.pipeline.config import ReplayConfig, dump_toml, from_dict, load_config
from training.pipeline.procs import CommandError
from training.pipeline.report import estimate, render
from training.pipeline.runner import Pipeline, trainer_flags
from training.pipeline.state import load_state, read_events

REPO = Path(__file__).resolve().parents[2]
FIXTURE_SHARD = REPO / "training" / "testdata" / "sample-shard.npz"


def _arg(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def write_fake_shard(path: Path, rows: int, games: int = 2, model: str = "heuristic") -> None:
    meta = json.dumps({"format": "gofer-shard", "version": 1, "board_size": 9, "rows": rows, "games": games, "model": model}).encode()
    np.savez_compressed(
        path,
        spatial=np.zeros((rows, 8, 9, 9), np.uint8),
        value=np.zeros(rows, np.float32),
        full_search=np.ones(rows, np.uint8),
        meta=np.frombuffer(meta, np.uint8),
    )


class FakeJob:
    def __init__(self) -> None:
        self.terminated = False

    def wait(self) -> None:
        pass

    def poll(self) -> bool:
        return True

    def terminate(self) -> None:
        self.terminated = True


class FakeExecutor:
    """Simulates gofer / trainer / exporter by writing their output artifacts."""

    # The engine answers -arena-config-hash with a hash of the configuration.
    # The fake derives one the same way: from the argv it is handed. Without
    # this the helper raised AttributeError, the caller swallowed it, and the
    # hash check silently never ran - in the tests as well as in production.
    config_hash = "fakehash00000000"

    def capture(self, cmd, *, env=None):
        # Deliberately NOT recorded in self.calls: this is a query about what a
        # match would be, not a match. Recording it makes every cached report
        # look as though it had been replayed.
        return self.config_hash + "\n"

    def __init__(self, arena=None, rows_per_game: int = 50, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.arena = arena or (lambda cycle, batch, games: (games // 2, games // 2, 0))
        self.rows_per_game = rows_per_game
        self.fail_on = fail_on
        self.sidecars: list[tuple[Path, int, FakeJob]] = []

    def kind(self, cmd: list[str]) -> str:
        if cmd[0] == "go":
            return "build"
        if "-selfplay" in cmd:
            return "selfplay"
        if "-arena" in cmd:
            return "arena"
        if any(c.endswith("export_onnx.py") for c in cmd):
            return "export"
        return "train"

    def run(self, cmd, *, log, env=None) -> None:
        kind = self.kind(cmd)
        self.calls.append(cmd)
        if kind == self.fail_on:
            self.fail_on = None
            raise CommandError(f"injected failure in {kind}")
        if kind == "selfplay":
            games = int(_arg(cmd, "-games"))
            model = "heuristic" if _arg(cmd, "-selfplay-eval") == "heuristic" else "sha256:x"
            write_fake_shard(Path(_arg(cmd, "-o")), games * self.rows_per_game, games, model)
        elif kind == "train":
            out = Path(_arg(cmd, "--out-dir"))
            out.mkdir(parents=True, exist_ok=True)
            (out / "best.pt").write_bytes(b"weights")
            (out / "summary.json").write_text(json.dumps({"val_loss": 1.5}))
        elif kind == "export":
            Path(_arg(cmd, "--out")).write_bytes(Path(_arg(cmd, "--out")).name.encode())
        elif kind == "arena":
            games = int(_arg(cmd, "-games"))
            seed = int(_arg(cmd, "-seed"))
            w, l, d = self.arena(seed // 1000, seed % 1000, games)
            Path(_arg(cmd, "-json")).write_text(json.dumps(
                {"wins_challenger": w, "wins_baseline": l, "draws": d, "game_count": games,
                 "config_hash": self.config_hash}))

    def spawn(self, cmd, *, log, env=None):
        self.run(cmd, log=log, env=env)
        return FakeJob()

    def sidecar(self, python, model, port, *, log):
        job = FakeJob()
        self.sidecars.append((model, port, job))
        return job

    def count(self, kind: str) -> int:
        return sum(1 for c in self.calls if self.kind(c) == kind)


def make_cfg(tmp_path: Path, **overrides):
    data = {
        "run": {"name": "t", "root": str(tmp_path / "runs"), "seed": 0, "overlap_selfplay": False},
        "engine": {"backend": "sidecar", "build": True},
        "selfplay": {"games_per_cycle": 4, "parallel": 2},
        "replay": {"min_window_rows": 100, "max_window_rows": 1000, "min_rows_to_train": 1},
        "gating": {"batch_games": 10, "max_games": 40, "parallel": 2, "bootstrap_games": 10},
        "publish": {"registry": str(tmp_path / "registry")},
    }
    for dotted, val in overrides.items():
        sec, key = dotted.split("__")
        data.setdefault(sec, {})[key] = val
    return from_dict(data)


def make_pipe(tmp_path: Path, ex: FakeExecutor, **overrides) -> Pipeline:
    cfg = make_cfg(tmp_path, **overrides)
    pipe = Pipeline(cfg, ex, REPO, out=lambda _m: None)
    pipe.prepare()
    return pipe


def arena_losing_regression():
    """Gates pass, the regression match (batch 900) loses."""
    def play(cycle, batch, games):
        return (0, games, 0) if batch == 900 else (games, 0, 0)
    return play


# Candidate cycle 2 crushes the champion; cycle 3 loses badly.
def scripted_arena(cycle, batch, games):
    if cycle == 2:
        return games, 0, 0
    if cycle == 3:
        return 0, games, 0
    return games // 2, games // 2, 0


# ------------------------------------------------------------------- runner

def test_three_cycles_seed_promote_reject(tmp_path):
    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex, gating__max_games=100)
    assert pipe.run(max_cycles=3) == 3

    st = load_state(pipe.state_path)
    assert st.cycle == 3 and st.in_progress is None
    assert [g.generation for g in st.generations] == [1, 2]
    assert st.generations[0].elo == 0.0
    assert st.generations[1].elo > 100  # 10-0 sweep -> large measured gain
    assert (pipe.models_dir / "gen-0002.onnx").exists()
    assert st.lifetime_rows == 3 * 4 * 50

    h2 = json.loads((pipe.history_dir / "cycle-0002.json").read_text())
    assert h2["gate"]["verdict"] == stats.ACCEPT and h2["promoted"]
    assert h2["gate"]["games"] < 100  # SPRT stopped before max_games
    h3 = json.loads((pipe.history_dir / "cycle-0003.json").read_text())
    assert h3["gate"]["verdict"] == stats.REJECT and not h3["promoted"]

    # Cycle 1 self-plays with the heuristic; later cycles with the champion over a sidecar.
    sp = [c for c in ex.calls if ex.kind(c) == "selfplay"]
    assert _arg(sp[0], "-selfplay-eval") == "heuristic"
    assert _arg(sp[1], "-selfplay-eval") == "mix" and "-onnx-url" in sp[1]
    # Resuming training from the champion uses the resume schedule.
    tr = [c for c in ex.calls if ex.kind(c) == "train"]
    assert "--init-from" not in tr[0] and _arg(tr[0], "--epochs") == "25"
    assert _arg(tr[1], "--init-from").endswith("gen-0001.pt") and _arg(tr[1], "--epochs") == "15"
    assert all(job.terminated for _, _, job in ex.sidecars)

    events = read_events(pipe.events_path)
    assert sum(1 for e in events if e["stage"] == "cycle") == 3
    page = render(pipe.dir)
    assert "gen 2" in page and "<svg" in page
    assert estimate(pipe.dir, cycles=10)["cycles_measured"] == 3


def test_resume_after_crash_skips_finished_stages(tmp_path):
    ex = FakeExecutor(fail_on="train")
    pipe = make_pipe(tmp_path, ex)
    with pytest.raises(CommandError):
        pipe.run(max_cycles=1)
    st = load_state(pipe.state_path)
    assert st.in_progress is not None and st.in_progress.done == ["selfplay"]

    pipe2 = Pipeline(pipe.cfg, ex, REPO, out=lambda _m: None)
    assert pipe2.run(max_cycles=1) == 1
    assert ex.count("selfplay") == 1  # not regenerated
    assert load_state(pipe.state_path).lifetime_rows == 200  # not double counted


def test_resumed_gate_reuses_finished_batches(tmp_path):
    ex = FakeExecutor(arena=lambda c, b, g: (g // 2, g // 2, 0))
    pipe = make_pipe(tmp_path, ex)
    pipe.run(max_cycles=1)
    gdir = pipe.gating_dir / "cycle-0002"
    gdir.mkdir(parents=True)
    (gdir / "batch-01.json").write_text(json.dumps(
        {"wins_challenger": 9, "wins_baseline": 1, "draws": 0, "game_count": 10,
         "config_hash": FakeExecutor.config_hash}))
    pipe.run(max_cycles=1)
    arena_calls = [c for c in ex.calls if ex.kind(c) == "arena"]
    assert all(not _arg(c, "-json").endswith("batch-01.json") for c in arena_calls)


def test_min_rows_skips_training(tmp_path):
    ex = FakeExecutor()
    pipe = make_pipe(tmp_path, ex, replay__min_rows_to_train=10_000)
    pipe.run(max_cycles=2)
    assert ex.count("train") == 0 and load_state(pipe.state_path).champion is None
    assert json.loads((pipe.history_dir / "cycle-0002.json").read_text())["skipped"] is True


def test_hold_mode_never_promotes(tmp_path):
    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex, gating__mode="hold")
    pipe.run(max_cycles=2)
    st = load_state(pipe.state_path)
    assert st.champion is None
    h = json.loads((pipe.history_dir / "cycle-0001.json").read_text())
    assert h["gate"]["would_promote"] and not h["gate"]["promote"]


def test_overlap_prefetches_next_selfplay(tmp_path):
    ex = FakeExecutor()
    pipe = make_pipe(tmp_path, ex, run__overlap_selfplay=True)
    pipe.run(max_cycles=2)
    # cycle 2's shard was produced during cycle 1 (before cycle 1's training).
    kinds = [ex.kind(c) for c in ex.calls if ex.kind(c) != "build"]
    assert kinds[:3] == ["selfplay", "selfplay", "train"]
    assert ex.count("selfplay") == 2  # the last cycle does not prefetch a third


def test_inprocess_backend_passes_models(tmp_path, monkeypatch):
    monkeypatch.setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "/fake/libonnxruntime.so")
    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex, engine__backend="inprocess")
    pipe.run(max_cycles=2)
    build = ex.calls[0]
    assert "-tags=onnx" in build
    arena = [c for c in ex.calls if ex.kind(c) == "arena"][-1]
    assert "-arena-play-all" in arena  # batches must be the fixed size the SPRT assumes
    assert _arg(arena, "-model").endswith("gen-0001.onnx")
    assert _arg(arena, "-model-2").endswith("candidate-0002.onnx")
    assert not ex.sidecars


def test_trainer_flags():
    assert trainer_flags({"batch-size": 256, "amp": True, "compile": False, "steps": 10}) == ["--batch-size", "256", "--amp", "--steps", "10"]
    assert trainer_flags({"blocks_per": 2}) == ["--blocks-per", "2"]


# -------------------------------------------------------------------- stats

def test_sprt_bounds_and_decisions():
    lo, hi = stats.sprt_bounds(0.05, 0.05)
    assert math.isclose(lo, -hi)
    assert stats.sprt_decision(60, 10, 0, elo0=0, elo1=35, alpha=0.05, beta=0.05)[0] == stats.ACCEPT
    assert stats.sprt_decision(10, 60, 0, elo0=0, elo1=35, alpha=0.05, beta=0.05)[0] == stats.REJECT
    assert stats.sprt_decision(5, 5, 0, elo0=0, elo1=35, alpha=0.05, beta=0.05)[0] == stats.CONTINUE


def test_elo_score_roundtrip_and_wilson():
    for elo in (-200, -35, 0, 35, 200):
        assert math.isclose(stats.score_to_elo(stats.elo_to_score(elo)), elo, abs_tol=1e-6)
    lo, hi = stats.wilson(50, 100)
    assert lo < 0.5 < hi and math.isclose(lo + hi, 1.0)


def test_config_overrides_and_validation(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[gating]\nmax_games = 80\n[train.args]\nbatch-size = 128\n')
    cfg = load_config(p, ["gating.elo1=50", "run.name=\"x\""])
    assert cfg.gating.max_games == 80 and cfg.gating.elo1 == 50 and cfg.run.name == "x"
    assert cfg.train.args == {"batch-size": 128}
    assert load_config(None, []).to_dict() == from_dict({}).to_dict()
    with pytest.raises(ValueError, match="unknown keys"):
        from_dict({"gating": {"max_gmes": 1}})
    with pytest.raises(ValueError, match="even"):
        from_dict({"gating": {"batch_games": 7}})


def test_dump_toml_roundtrips(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.train.args = {"batch-size": 64, "amp": True}
    p = tmp_path / "dump.toml"
    p.write_text(dump_toml(cfg))
    assert load_config(p).to_dict() == cfg.to_dict()


def test_repo_configs_parse():
    for p in sorted((REPO / "configs").glob("pipeline*.toml")):
        load_config(p)


# ------------------------------------------------------------------- replay

def test_read_meta_from_go_written_fixture():
    meta = replay_index.read_shard_meta(FIXTURE_SHARD)
    assert meta["format"] == "gofer-shard" and meta["board_size"] == 9
    z = np.load(FIXTURE_SHARD)
    assert z["spatial"].shape[0] == meta["rows"] and z["spatial"].dtype == np.uint8
    # Side-to-move identity the Go writer guarantees: score == sum(ownership) -/+ komi.
    black = z["globals"][:, 2] == 1
    own = z["ownership"].sum(1).astype(np.float32)
    expect = np.where(black, own - meta["komi"], own + meta["komi"])
    assert np.array_equal(expect, z["score"])
    full = z["full_search"] == 1
    assert full.any() and (~full).any()
    assert np.allclose(z["policy"].sum(1), 1.0, atol=1e-4)


def test_window_grows_sublinearly():
    cfg = ReplayConfig(min_window_rows=1000, max_window_rows=50_000)
    assert replay_index.window_rows(500, cfg) == 500
    assert replay_index.window_rows(1000, cfg) == 1000
    w10, w100 = replay_index.window_rows(10_000, cfg), replay_index.window_rows(100_000, cfg)
    assert 1000 < w10 < 10_000 and w10 < w100 < 100_000
    assert replay_index.window_rows(10**9, cfg) == 50_000


def test_select_window_and_archive(tmp_path):
    d = tmp_path / "sp"
    d.mkdir()
    for c in range(1, 6):
        write_fake_shard(d / f"cycle-{c:04d}.npz", 100)
    shards = replay_index.scan(d)
    assert [s.cycle for s in shards] == [1, 2, 3, 4, 5]
    assert [s.cycle for s in replay_index.select_window(shards, 150)] == [4, 5]
    cfg = ReplayConfig(min_window_rows=100, max_window_rows=100, archive_factor=2.0)
    moved = replay_index.archive_old(d, cfg, lifetime_rows=500)
    assert sorted(p.name for p in moved) == ["cycle-0001.npz", "cycle-0002.npz", "cycle-0003.npz"]
    assert [s.cycle for s in replay_index.scan(d)] == [4, 5]


def test_prune_keeps_champions(tmp_path):
    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex, gating__max_games=100, run__keep_train_dirs=1, run__keep_candidates=1)
    pipe.run(max_cycles=3)
    assert [d.name for d in pipe.train_dir.glob("cycle-*")] == ["cycle-0003"]
    assert [f.name for f in pipe.models_dir.glob("candidate-*.onnx")] == ["candidate-0003.onnx"]
    assert sorted(f.name for f in pipe.models_dir.glob("gen-*.pt")) == ["gen-0001.pt", "gen-0002.pt"]


# ---------------------------------------------- regressions found in review

def test_prefetched_shard_stays_out_of_the_current_window(tmp_path):
    """Overlap must not let the next cycle's games into this cycle's training data."""
    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex, run__overlap_selfplay=True, gating__max_games=100)
    pipe.run_cycle(1, prefetch_next=True)
    # Cycle 2 was generated during cycle 1; it must be staged, not in selfplay/.
    assert [p.name for p in pipe.selfplay_dir.glob("*.npz")] == ["cycle-0001.npz"]
    assert (pipe.selfplay_dir / "pending" / "cycle-0002.npz").exists()
    pipe.run_cycle(2, prefetch_next=False)
    assert sorted(p.name for p in pipe.selfplay_dir.glob("*.npz")) == ["cycle-0001.npz", "cycle-0002.npz"]
    assert ex.count("selfplay") == 2  # cycle 2 came from the prefetch, not a rerun


def test_truncated_arena_report_is_replayed(tmp_path):
    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex, gating__max_games=100)
    pipe.run(max_cycles=1)
    gdir = pipe.gating_dir / "cycle-0002"
    gdir.mkdir(parents=True)
    (gdir / "batch-01.json").write_text('{"wins_challenger": 4, "wins_base')  # killed mid-write
    pipe.run(max_cycles=1)  # must not raise JSONDecodeError
    assert json.loads((gdir / "batch-01.json").read_text())["game_count"] == 10


def test_cached_report_from_a_bigger_batch_is_not_reused(tmp_path):
    ex = FakeExecutor()
    pipe = make_pipe(tmp_path, ex)
    report = pipe.gating_dir / "stale.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"wins_challenger": 30, "wins_baseline": 10, "draws": 0, "game_count": 40}))
    rep = pipe.run_match(report, 10, 1, None, pipe.models_dir / "x.onnx", log_name="t")
    assert rep["game_count"] == 10


def test_regression_baseline_is_the_highest_generation(tmp_path):
    from training.pipeline.publish import Publisher
    from training.pipeline.state import Generation

    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex, publish__regression_lookback=2)
    pipe.state.generations = [
        Generation(g, g, f"models/gen-{g:04d}.onnx", f"models/gen-{g:04d}.pt", 0.0, "t") for g in (1, 2, 3, 4)
    ]
    # A rollback to gen 2 appends a duplicate carrying the older number.
    pipe.state.generations.append(Generation(2, 5, "models/gen-0002.onnx", "models/gen-0002.pt", 0.0, "t"))
    champ = Generation(5, 6, "models/gen-0005.onnx", "models/gen-0005.pt", 0.0, "t")
    pipe.state.generations.append(champ)
    assert Publisher(pipe)._regression_baseline(champ).generation == 3


def test_demote_is_not_replayed_onto_a_newer_champion(tmp_path):
    from training.pipeline.publish import Publisher
    from training.pipeline.state import Generation

    ex = FakeExecutor(arena=arena_losing_regression())
    pipe = make_pipe(tmp_path, ex, publish__regression_games=10, publish__regression_lookback=1,
                     publish__regression_action="demote")
    gens = [Generation(g, g, f"models/gen-{g:04d}.onnx", f"models/gen-{g:04d}.pt", 0.0, "t") for g in (1, 2)]
    pipe.state.generations = list(gens)
    pipe.state.begin_cycle(3)
    pub = Publisher(pipe)
    first = pub.publish_champion(3, generation=2)
    assert first["published"] == "held"
    assert [g.generation for g in pipe.state.generations] == [1, 2, 1]
    # Replaying the interrupted stage must not make the failed generation champion again.
    again = pub.publish_champion(3, generation=2)
    assert again["published"] == "held"
    assert [g.generation for g in pipe.state.generations] == [1, 2, 1]
    assert pipe.state.champion.generation == 1


def test_seed_gate_result_survives_into_the_lineage(tmp_path):
    ex = FakeExecutor(arena=lambda c, b, g: (g, 0, 0))
    pipe = make_pipe(tmp_path, ex)
    pipe.run(max_cycles=1)
    gate = load_state(pipe.state_path).generations[0].gate
    assert gate["vs_heuristic_games"] == 10 and gate["vs_heuristic_score"] == 1.0


def test_anchor_measures_the_candidate_against_the_heuristic(tmp_path):
    """After cycle 1 the SPRT only compares nets to each other; the anchor is the
    one number in the run that does not move when the champion does."""
    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex, gating__anchor_every=1, gating__anchor_games=10)
    pipe.run(max_cycles=2)

    decision = json.loads((pipe.gating_dir / "cycle-0002" / "decision.json").read_text())
    assert decision["vs_heuristic_kind"] == "anchor"
    assert decision["vs_heuristic_games"] == 10
    assert (pipe.gating_dir / "cycle-0002" / "anchor-vs-heuristic.json").exists()

    # It must be the seed gate's match: heuristic on one side, no champion model.
    anchors = [c for c in ex.calls if ex.kind(c) == "arena"
               and _arg(c, "-json").endswith("anchor-vs-heuristic.json")]
    assert len(anchors) == 1 and _arg(anchors[0], "-black-eval") == "heuristic"
    # Its own seed: sharing the regression match's batch 900 would mean sharing
    # its openings, and two matches on one set of openings are one sample.
    assert int(_arg(anchors[0], "-seed")) % 1000 == 800


def test_anchor_is_off_by_default(tmp_path):
    """Default runs keep the old shape: no extra arena, no absolute claim."""
    ex = FakeExecutor(arena=scripted_arena)
    pipe = make_pipe(tmp_path, ex)
    pipe.run(max_cycles=2)

    decision = json.loads((pipe.gating_dir / "cycle-0002" / "decision.json").read_text())
    assert "vs_heuristic_elo" not in decision
    assert not any(_arg(c, "-json").endswith("anchor-vs-heuristic.json")
                   for c in ex.calls if ex.kind(c) == "arena")


def test_config_rejects_odd_bootstrap_games_and_empty_cycles(tmp_path):
    with pytest.raises(ValueError, match="bootstrap_games"):
        make_cfg(tmp_path, gating__bootstrap_games=7)
    with pytest.raises(ValueError, match="games_per_cycle"):
        make_cfg(tmp_path, selfplay__games_per_cycle=0)


# ------------------------------------------------- portability (ORT downloads)

def test_ort_build_covers_arm_and_mac():
    """Apple Silicon and ARM servers (Graviton) must not be locked out."""
    from training.pipeline.procs import ORT_VERSION, ort_build_for

    cases = {
        ("Linux", "x86_64"): "linux-x64",
        ("Linux", "aarch64"): "linux-aarch64",
        ("Linux", "arm64"): "linux-aarch64",      # some distros report arm64
        ("Darwin", "arm64"): "osx-arm64",
        ("Windows", "AMD64"): "win-x64",
    }
    for (system, machine), expected in cases.items():
        build = ort_build_for(system, machine)
        assert build is not None, f"{system}/{machine} has no ORT build"
        name, kind, lib = build
        assert expected in name and ORT_VERSION in name
        assert kind in ("tgz", "zip") and lib.startswith("lib/")


def test_unsupported_platform_points_at_the_sidecar(tmp_path, monkeypatch):
    """Intel Macs have no pinned in-process build; the error must say what does work."""
    from training.pipeline import procs

    assert procs.ort_build_for("Darwin", "x86_64") is None
    monkeypatch.setattr(procs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(procs.platform, "machine", lambda: "x86_64")
    with pytest.raises(procs.CommandError, match="sidecar"):
        procs.ensure_ort_library(tmp_path)


def test_batch_size_matches_stage_parallelism(tmp_path):
    """Both inference stages must size the batch to their own concurrency.

    The batched evaluator runs one worker goroutine that gathers a batch and
    then blocks on the inference call, so nothing else is evaluated while it is
    in flight. Throughput is therefore capped at batch-size evaluations per
    serialised dispatch. Left at the engine default of 8 while the stage runs 32
    games, a 32-way stage does the work of roughly one thread - and the symptom
    is only slowness, so nothing fails and nobody looks.
    """
    pipe = make_pipe(tmp_path, FakeExecutor(),
                     selfplay__parallel=32, gating__parallel=16)

    sp = pipe.selfplay_command(1)
    assert "-batch-size" in sp, "self-play never sized the batch"
    assert sp[sp.index("-batch-size") + 1] == sp[sp.index("-selfplay-parallel") + 1] == "32"

    ar = pipe.arena_command(40, 1, tmp_path / "r.json", None, tmp_path / "c.onnx")
    assert "-batch-size" in ar, "the arena never sized the batch, and it is 81% of a cycle"
    assert ar[ar.index("-batch-size") + 1] == ar[ar.index("-arena-parallel") + 1] == "16"


def test_batch_size_override_is_honoured(tmp_path):
    """engine.batch_size pins it explicitly; 0 means follow the stage."""
    pipe = make_pipe(tmp_path, FakeExecutor(),
                     selfplay__parallel=32, engine__batch_size=64)
    sp = pipe.selfplay_command(1)
    assert sp[sp.index("-batch-size") + 1] == "64"
    assert sp[sp.index("-selfplay-parallel") + 1] == "32"


def test_cached_report_from_a_different_configuration_is_replayed(tmp_path):
    """A cached arena report must match the configuration that would produce it.

    Game count alone is not enough: a resumed run can reach a gate with a
    different candidate, backend or opening setting, and the stale report has
    the right shape and the wrong contents. This check shipped broken once - the
    helper read a misnamed attribute, the caller swallowed the AttributeError,
    and the hash was never compared in production OR in these tests.
    """
    ex = FakeExecutor()
    pipe = make_pipe(tmp_path, ex)
    gdir = pipe.gating_dir / "cycle-0002"
    gdir.mkdir(parents=True)
    stale = gdir / "batch-01.json"
    stale.write_text(json.dumps({"wins_challenger": 9, "wins_baseline": 1, "draws": 0,
                                 "game_count": 10, "config_hash": "a-different-configuration"}))

    assert not pipe._usable_report(stale, 10, ex.config_hash), "stale evidence accepted"
    # The same report under the matching hash is fine, so the check is not just refusing everything.
    stale.write_text(json.dumps({"wins_challenger": 9, "wins_baseline": 1, "draws": 0,
                                 "game_count": 10, "config_hash": ex.config_hash}))
    assert pipe._usable_report(stale, 10, ex.config_hash)


def test_arena_config_hash_surfaces_programming_errors(tmp_path):
    """Only environmental failures may degrade to the game-count check.

    The first version caught bare Exception, so a typo in the helper looked
    exactly like a missing engine and the check quietly stopped running.
    """
    class Broken(FakeExecutor):
        def capture(self, cmd, *, env=None):
            raise TypeError("a bug in the helper, not a missing binary")

    pipe = make_pipe(tmp_path, Broken())
    with pytest.raises(TypeError):
        pipe._arena_config_hash(tmp_path / "r.json", 10, 1, None, tmp_path / "c.onnx")

    class NoEngine(FakeExecutor):
        def capture(self, cmd, *, env=None):
            raise OSError("engine binary missing")

    pipe = make_pipe(tmp_path, NoEngine())
    assert pipe._arena_config_hash(tmp_path / "r.json", 10, 1, None, tmp_path / "c.onnx") is None
