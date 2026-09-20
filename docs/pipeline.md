# Training pipeline (orchestrator v4)

`python -m training.pipeline` runs the whole reinforcement loop from one TOML config: self-play, train, export, gate, promote, publish. It is resumable at every stage, runs the same way on a laptop, a free CI runner, or a rented GPU, and replaces `scripts/train-loop-v3.sh`, whose Python side was removed once nothing depended on it.

```text
          ┌──────────── cycle N ─────────────────────────────────────────────┐
 champion │ self-play ──► shard ──► train ──► export ──► gate ──► promote ──► publish
  gen k   │ (Go, .npz)    window    (PyTorch)  (ONNX +   (SPRT     gen k+1     registry,
          │                         init=gen k  parity)  vs gen k)            alias, Release
          └──────────────────────────────────────────────────────────────────┘
            self-play for cycle N+1 already runs in the background (overlap)
```

The learner (model, trainer, export, presets) is documented in [training/README.md](../training/README.md). The data contract is in [training-data-format.md](training-data-format.md). The design rationale is in [ADR 0007](decisions/0007-pipeline-orchestrator.md).

## Quick start

```bash
pip install -r training/requirements.txt
python -m training.pipeline run --config configs/pipeline-smoke.toml     # ~3 min, 3 cycles, any machine
python -m training.pipeline status --config configs/pipeline-smoke.toml
python -m training.pipeline report --config configs/pipeline-smoke.toml  # runs/smoke/report.html
```

`make pipeline-smoke` does the same. Stop the run at any point (Ctrl-C, `docker stop`, spot preemption) and rerun the same command: it resumes at the first unfinished stage.

## What changed from the v3 shell loop

| | v3 (`train-loop-v3.sh`) | v4 orchestrator |
|---|---|---|
| Self-play data | JSONL, full-search rows only | `.npz` shards (~40x smaller); fast rows kept for value/ownership |
| Replay | one `replay.jsonl`, rewritten on every trim | immutable shards; window grows with total data (KataGo power law) |
| Gating | Wilson bound re-checked after every game (~8% false promotions at zero gain) | SPRT over fixed-size arena batches: ~2.3% false promotions and more power (90% at +50 Elo vs 67%); longer gates near zero gain |
| Arena fairness | seeds linear in the game index, correlated with the colour swap | seeds mixed per game, so role attribution is unbiased ([ADR 0008](decisions/0008-search-correctness.md)) |
| Throughput | stages strictly serial | next cycle's self-play overlaps training and gating |
| Crash / preemption | restart the cycle | per-stage checkpoints; resumes mid-cycle; the trainer continues with `--continue` |
| Champions | `best.onnx` overwritten, one archive copy | immutable numbered generations, Elo ladder, registry, rollback |
| Config | environment variables in a shell script | one typed, validated TOML file, snapshotted into the run dir |
| Visibility | log file | HTML dashboard, `status`, cost/time `estimate` |

## Configs

| Config | Where | Notes |
|---|---|---|
| [pipeline-smoke.toml](../configs/pipeline-smoke.toml) | laptop / CI | tiny; sidecar backend, so no CGO is needed |
| [pipeline-cpu.toml](../configs/pipeline-cpu.toml) | always-on CPU box | legacy 6x64 lineage, `train-cpu` preset |
| [pipeline-actions.toml](../configs/pipeline-actions.toml) | free GitHub Actions runners | see [Free training](#free-training-on-github-actions) |
| [pipeline-gpu.toml](../configs/pipeline-gpu.toml) | one rented GPU | `gpool-10x96`, `train-gpu` preset, overlap on |

Override any key without editing a file: `--set gating.max_games=200 --set 'train.args={config="training/configs/train-gpu.toml", steps=5000}'`. Print the fully resolved config with `python -m training.pipeline config --config <file>`.

**Platforms.** The `inprocess` backend needs a pinned ONNX Runtime shared library, which the orchestrator downloads for Linux x86-64, Linux arm64 (Graviton), macOS arm64 (Apple Silicon) and Windows x86-64. Anywhere else — an Intel Mac, for instance — use `engine.backend = "sidecar"`, which only needs the Python `onnxruntime` wheel; the error message says so.

Sections: `[run]` (name, deadline, overlap, pruning), `[engine]` (backend `inprocess` | `sidecar`), `[selfplay]`, `[replay]` (window), `[train]` (script plus `args` passed through to the trainer), `[gating]` (SPRT), `[publish]`. Unknown keys are errors.

## The stages

**Self-play.** `gofer -selfplay -o runs/<name>/selfplay/cycle-NNNN.npz`. The first cycle uses the heuristic evaluator (`bootstrap_games`); after that it uses the champion. With `overlap_selfplay`, cycle N+1's games start as soon as cycle N's finish, using the champion at that moment (asynchronous, KataGo-style).

**Train.** The trainer gets the shard directory with `--window-rows W`, where `W = c·(1 + β((N/c)^α − 1)/α)` over lifetime rows N (`min_window_rows`=c, α=0.75, β=0.4, capped at `max_window_rows`). Early heuristic-quality data ages out quickly, but the window keeps growing with the run. Shards older than `archive_factor × W` move to `selfplay/archive/`. The champion's weights seed training (`--init-from`, fresh optimizer). A retry after a crash adds `--continue`. When `train.args.config` names a learner preset, the preset owns the schedule.

**Export.** Writes `models/candidate-NNNN.onnx`. The exporter checks ORT against torch parity and fails the stage on a mismatch.

**Gate.** Plays the candidate against the current champion in batches of `batch_games`, colors alternating. After each batch a sequential probability ratio test compares H0 (gain ≤ `elo0`) with H1 (gain ≥ `elo1`) at error rates `alpha`/`beta`. The gate stops at accept, reject, or `max_games`; when inconclusive at `max_games` it falls back to the v3 rule (score ≥ `promote_win` and Wilson low > 0.5). Batches are cached on disk, so a resumed gate replays nothing. `python -m training.pipeline plan-sprt` prints the expected games per gate:

```text
 true Elo  promoted  E[games]      (defaults: elo0=0 elo1=35 alpha=0.05 beta=0.10, batch 40, cap 600)
     -100     0.000        94      clearly worse: rejected fast
        0     0.023       397      no real gain: almost never promoted, but it costs the most games
      +35     0.605       444
      +50     0.899       339
     +100     1.000       154      clearly better: accepted fast
```

These are exact, not estimates: `plan-sprt` runs the same decision code the gate
uses over every reachable (games, wins) state. `--compare` adds the v3 gate's
numbers for context, and `--elos` takes your own list. Raising `max_games` is the
main way to buy power; the settings per deployment are in the configs.

`gating.mode = "hold"` runs gates but never promotes (used for scoring investigations).

**Promote.** Copies the candidate to immutable `models/gen-NNNN.{onnx,pt,ckpt}`. Ladder Elo = previous champion's Elo + gain measured at the gate (generation 1 = 0). The first network seeds the lineage unconditionally after a sanity match against the heuristic.

**Publish.** See the next section.

## Publishing champions (and never losing an old one)

Every promoted generation is **published** automatically. Older bests are never deleted or overwritten:

- **Registry** (`publish.registry`, default `models/champions/`): each best is copied to `gofer-9x9-<run>-genNNNN.onnx` with a `.json` card (gate stats, Elo, sha256). `index.json` lists every entry and tracks `best` and `previous_best`.
- **Alias** (`publish.alias`, e.g. `models/gofer-9x9-best.onnx`): a convenience copy of the current best for GTP/arena commands. The registry, not the alias, holds the history.
- **GitHub Release** (`publish.github_release = true`): one release per generation (tag `gofer-9x9-<run>-genNNNN`, assets: ONNX + card), marked *latest*. Earlier releases stay downloadable. Upload failures never stop training; retry them with `python -m training.pipeline publish --retry`.

**False-best guard.** A candidate can pass its gate by luck, or beat the current champion while losing to older ones (non-transitive strength). With `publish.regression_games > 0`, the new champion also plays the champion from `regression_lookback` generations back before anything is published. If it scores below `regression_min_score`, it is recorded as `held`: the alias and the latest release don't move. With `regression_action = "demote"`, self-play also reverts to the previous champion.

**Rollback.** If a published best turns out to be bad:

```bash
python -m training.pipeline publish  --config configs/pipeline-gpu.toml            # list: status, Elo, release URL
python -m training.pipeline rollback --config configs/pipeline-gpu.toml --to 7     # best -> gen 7 (alias + latest release)
python -m training.pipeline rollback --config configs/pipeline-gpu.toml --to 7 --champion   # and train from gen 7 again
```

Rollback only re-points `best`: nothing is deleted, and the next promotion gets a fresh generation number.

## Free training on GitHub Actions

The repo is public, so standard Actions runners are free (4 vCPU, 16 GB, 6 h per job). [`.github/workflows/train-free.yml`](../.github/workflows/train-free.yml) runs [pipeline-actions.toml](../configs/pipeline-actions.toml) every 6 hours:

1. restore `runs/actions` from the Actions cache
2. train until the config deadline (4.75 h), with a hard stop at 5.3 h (state is checkpointed per stage)
3. save `runs/actions` back to the cache, and upload the dashboard as an artifact
4. publish each new best as a GitHub Release

It's **opt-in**:

```bash
gh variable set TRAIN_FREE_ENABLED --body true      # enable the schedule
gh workflow run train-free.yml                      # or run once by hand
gh variable set TRAIN_FREE_ENABLED --body false     # pause
```

Caveats: CPU only, so progress is slow (use `estimate` after a few cycles). Actions caches are evicted after 7 days without access; with a 6-hour schedule that doesn't happen while enabled, but after a long pause the run restarts from scratch. Published releases are unaffected, and `run.init_checkpoint` can warm-start from one.

## Renting a GPU

When there's budget, one command covers each provider:

- **SkyPilot** (any cloud with credentials; picks the cheapest spot GPU; the run dir lives in a bucket, so preemption loses nothing): edit `RUN_BUCKET` in [infra/cloud/skypilot.yaml](../infra/cloud/skypilot.yaml), then `sky jobs launch infra/cloud/skypilot.yaml`.
- **Docker** (RunPod / Vast / any NVIDIA host): `docker compose -f infra/docker/compose.yaml up gpu`. The prebuilt image `ghcr.io/devomb/gofer-train:gpu` comes from [train-image.yml](../.github/workflows/train-image.yml); run it manually or on a `v*` tag.
- **Plain VM**: `bash infra/cloud/bootstrap.sh --run` installs Go, ORT 1.26.0, torch (CUDA if `nvidia-smi` works), builds the engine, self-tests, and starts the loop.

Bring results home with `bash infra/cloud/sync-run.sh pull user@host:/path/to/Gofer gpu`. It is incremental, because shards and champions are immutable, and it rebuilds the dashboard locally.

Budget before renting: run a few cycles anywhere, then

```bash
python -m training.pipeline estimate --config configs/pipeline-gpu.toml --cycles 100 --hourly-usd 0.40
```

This projects wall-clock and cost from the measured per-stage timings. Set `run.deadline_hours` as a hard budget cap.

## Run directory

```text
runs/<name>/
  config.toml           resolved config snapshot
  state.json            manifest v4: completed cycle, in-progress stages, lifetime rows, generations (Elo ladder)
  events.jsonl          stage start/done/error with durations (dashboard + estimate)
  selfplay/cycle-NNNN.npz, selfplay/archive/
  train/cycle-NNNN/     trainer output (best.pt, best.ckpt, last.pt, metrics.jsonl, summary.json)
  models/candidate-NNNN.onnx, models/gen-NNNN.{onnx,pt,ckpt}
  gating/cycle-NNNN/    batch-NN.json arena reports, decision.json, regression-vs-genNNNN.json
  history/cycle-NNNN.json   everything about one cycle
  logs/                 pipeline.log + per-stage command logs
  report.html
```

`run.keep_train_dirs` / `run.keep_candidates` prune old training outputs and rejected candidates for long runs. Champions are never pruned.

## Command reference

| Command | Purpose |
|---|---|
| `run [--cycles N] [--no-build]` | run or resume the loop |
| `status [--json]` | champion, cycle, in-progress stages |
| `report [--out f.html]` | dashboard (Elo ladder, gates with CIs, time per stage, cycle table) |
| `estimate --cycles N [--hourly-usd X]` | time/cost projection from measured timings |
| `plan-sprt` | expected gate length for the configured SPRT |
| `publish [--retry]` | list published champions; retry failed GitHub releases |
| `rollback --to N [--champion]` | re-point best (and optionally the training champion) |
| `config` | print the resolved config |

All commands take `--config` and repeatable `--set section.key=value`.
