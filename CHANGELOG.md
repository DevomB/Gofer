# Changelog

All notable changes to Gofer are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed

- **The search never left the root on 9x9.** `Arena.Get` returns a pointer into a slice
  that `AddChild` appends to, and `expandLocked` wrote the "expanded" flag through a
  pointer taken before the append. A 9x9 root has 82 children against an initial capacity
  of 64, so the flag was lost on every search: playouts stopped at the root, the
  transposition table answered the rest, and no child was ever visited. The engine played
  the first legal move regardless of playout count. Boards small enough to avoid the
  reallocation — every unit test — were unaffected.
- **Selection preferred the opponent's best move.** `puctScore` used a child's mean without
  negating it, but node values are stored from each node's own side to move, so more search
  made the engine weaker.
- **First-play urgency was a flat constant**, so once a child had been visited nothing could
  outscore it: at 200 playouts one move took every visit. The visit distribution is the
  policy training target, so this would have produced one-hot targets. It is now the node's
  own value minus a reduction that grows with the explored prior, as in KataGo.
- **The search could not see the end of the game.** Two consecutive passes were not terminal
  and finished positions were scored by the evaluator instead of the rules, so the engine
  walked into lost endings. Terminal results are never cached in the transposition table,
  whose key does not cover pass history.
- **Forced playouts flooded the root.** Wu (2020) forces extra playouts only on root
  children that already have visits; Gofer forced a floor on every child, before the search
  instead of after it. On 9x9 that is 82 children, outnumbering a 200-playout budget, and it
  flattened the visit distribution that becomes the policy target. At identical settings the
  top move now holds 0.199 of the visits instead of 0.159, over 24.3 moves instead of 41.5.
- **`-arena-enhanced baseline` silently meant "both".** The mode was resolved by comparing
  evaluator names, and the project's own baseline command uses the same name on both sides.
  Because forced playouts act at the root, this masked the root defects in the very
  measurement meant to detect them. It now follows the baseline role across the colour swap,
  and the baseline command uses `-arena-enhanced none`.
- **Measurement runs stopped early on the statistic they measured.** A match ends when its
  promotion gate decides; `minGamesBeforePromote` guards only the accept branch, so a
  200-game bias test silently played 169, stopping when the challenger fell behind. The same
  rule ran inside every gate batch, stacking a second stopping rule under the SPRT.
  `-arena-play-all` disables it, and the gate and the measurement tests both pass it.
- **Arena role attribution was biased.** Per-game seeds were linear in the game index while
  colours swapped on the same parity, so one role systematically drew correlated openings.
  With two evaluators that are literally the same code, roles split 36/64, 55/26 and 55/30
  across three seeds; after mixing the seeds they split 55/53, 51/69 and 55/61.

The project's reproducible baseline (200 games, identical heuristics, Black 600 playouts
vs White 200) went from **7/200** to **136/200** for the deeper search: -576 Elo to +131,
a sign flip, with the median game going from 11 moves to 55. Repeated on a second seed it
gives 142/200; pooled over both, **278/400** = +143 Elo, 95% CI [+106, +180]. A single
200-game arena carries a 95% interval about 105 Elo wide (+186 is [+131, +241]; +131 is
[+80, +182] -- they overlap over nearly their whole length), so no number in this entry
from one 200-game run should be read as exact. Doubling to 400 games narrows it to 74. A controlled 40-game pair at
identical settings gives 0/40 before and 24/40 after. With fair komi and identical
evaluators, 200 games now split 99-101 by colour.

- Every strength number recorded before these fixes is void, including
  `.tectonix/reports/arena-9x9-baseline.json`, and self-play data produced before them
  carries policy targets from a search that did not search.
- Fair komi for equal heuristic engines at 50 playouts is now about 0.5; the tests that
  encoded the old value (fitted to the broken search) were updated.

- **The challenger never reached the second sidecar.** Building an ONNX evaluator went
  through four functions, two of them mutually recursive, and one worked out which model it
  was building by comparing the URL it had been handed against `ONNXURL2`. On the sidecar
  path the resolved model was discarded and the primary URL used instead, so that
  comparison was unreachable. `-eval-backend sidecar` with `-black-eval onnx -white-eval
  onnx2` pointed both engines at the same sidecar: the arena played the champion against
  itself and returned about 0.500 for every candidate, which is exactly the number a
  symmetric match should produce. The `-onnx-url-2` flag had been documented and defaulted
  to port 8081 throughout. Replaced with one constructor taking a slot, plus a pure
  `resolveONNXSlot` that is tested.
- **The transposition table never checked the key.** Slots were indexed by `hash & mask`
  with no key stored, so any two positions sharing the low 16 bits read each other's value.
  With 65,536 slots a collision is likely after a few hundred distinct positions and a
  single 9x9 search visits far more, so wrong evaluations were being backed up the tree and
  presenting as a weak evaluator. `Entry.Depth` was written as the literal `1` at all three
  store sites and only ever tested for non-zero: an occupancy flag named after a
  search-depth field. Slots now hold the full key and a filled bit. `NewTable` also never
  enforced the power-of-two size its mask assumes.
- **The JSONL path accepted rows the shard writer rejects.** `rows_from_jsonl` left
  ownership as zeros when a row had none; `WriteSampleShard` fails the file. The ownership
  loss is unmasked, so an unlabelled row is not a missing label but a label asserting that
  every point is neutral. Both paths now refuse it.
- Five Go files were committed unformatted, all misaligned struct or composite-literal
  columns from hand edits. `make lint` now runs `gofmt -l`. A `.gitattributes` pins the tree
  to LF so a Windows checkout does not report all 61 files and hide the real ones.

### Added

- Training pipeline v4: `.npz` self-play shards, a resumable orchestrator
  (`python -m training.pipeline`), SPRT gating, a champion registry with rollback, Docker /
  SkyPilot / free-CI deployment, and a paper in `paper/`. See
  [docs/pipeline.md](docs/pipeline.md) and [ADR 0007](docs/decisions/0007-pipeline-orchestrator.md).
- `plan-sprt` reports a gate's exact promotion rate and expected length
  (`training/pipeline/gate_oc.py`). Three short-mode tests guard the search: root children
  are visited, visits spread across moves, and selection prefers the move that is worse for
  the opponent.

### Removed

- The v3 training loop's Python side (`training/cycle.py`, `replay.py`, `manifest.py` and
  their tests, 413 lines), reachable only from `scripts/train-loop-v3.sh` and superseded by
  `python -m training.pipeline` (ADR 0007). The README still advertised it as the training
  path alongside v4.
- 1,526 lines of documentation that recorded plans rather than decisions: `docs/plans/`
  (a superseded v3 spec and a copy-paste agent prompt carrying a dead server IP),
  `optimization-framework.md` and `optimization-scorecard.md` (a self-graded 0-10 rubric
  cross-linked to a quality signal nothing computes), `implementation-blueprint.md`, and the
  four `backlog-*.md` task tables, untouched since July while the project shipped v4. Their
  genuinely open items, all marked "deferred", are a six-row table in `known-issues.md`.
- `docs/failure-modes.md`, merged into `known-issues.md` — two files for "what is wrong with
  this thing", one of them still claiming there was no real ONNX backend.
- `.tectonix/rules.toml` and `README-keys.md`, plus the style guide's Tectonix section:
  layer constraints and a session workflow for a tool that no target, workflow or script
  invokes. `.tectonix/reports/` stays; CI reads `bench-regression.json`.
- `stats.expected_games` and its helper (Wald's SPRT sample-size approximation, superseded
  by the exact dynamic program in `gate_oc`) and `gate_oc.gate_curve`. Neither had a caller
  outside its own test.
- `cmd/gofer/gating.go`, nine lines holding one constant whose two companions already lived
  in `match.go` next to the function applying all three.
- `Board.Neighbors`, a duplicate of `forEachNeighbor` that allocated a slice per call.

### Changed

- `cli.go` and `cmdline.go` are synonyms that held three unrelated things between them.
  Now `evaluators.go` (construction from flag names), `interactive.go` (`-play`, `-watch`,
  `-analyze`) and `commands.go` (flags, dispatch, batch commands).
- `runPlayoutForced` was `runPlayout` with the transposition probe removed and the descent
  seeded one node lower — thirty lines of copied hot-path code where a divergence would have
  been a silent search bug. Both now call `descend`.
- `models/README.md` described the v3 layout, including a `training/state/best.pt` that does
  not exist, and called the tracked bootstrap net an alias for the champion rather than the
  random-weights fixture it is.

## [2.7.1] - 2026-07-07

### Added

- `GATING_MODE=hold` for scoring investigation: arena runs but champion is not swapped (`training/cycle.py`, `train-loop-v3.sh`)
- Chinese area scoring invariant tests (`chinese_scoring_test.go`, `score_symmetry_test.go`, `arena_bias_test.go`)
- Champion ONNX archive on promote: `models/archive/pre-promote-cycle-N.onnx` before overwrite (`train-loop-v3.sh`)
- `scripts/replay-arena-cycle.sh` for Lightsail cycle validation

### Changed

- **Unified komi at 6.5** for self-play and arena; removed `komi9x9Arena` / `normalizeArenaKomi` arena-only remap
- Gating restored to `GATING_MODE=normal` after scoring investigation

### Fixed

- CI: CGO-free build gate, ONNX sidecar smoke, linux-amd64 bench regression baseline (`8a18c70`)

## [2.7.0] - 2026-07-06

### Added

- In-process ONNX Runtime backend (`ORTBackend`, `//go:build onnx`) via `onnxruntime_go` v1.31.0 / ORT 1.26.0
- `-eval-backend inprocess|sidecar`, `-model`, `-model-2` flags; default inference path is now **in-process**
- Parity harness: `scripts/parity-onnx.sh`, `training/parity_harness.py`, `cmd/gofer/onnx_parity_test.go`
- `make build-onnx`; `scripts/lightsail-inprocess-cycle.sh`; `MAX_CYCLES` in `train-loop-v3.sh`
- ADR [0004](docs/decisions/0004-in-process-onnx-inference.md); [docs/known-issues.md](docs/known-issues.md)

### Changed

- Arena early-stop: promotion gate with `minGamesBeforePromote=100`, early reject when max achievable win rate &lt; 0.55, floor 20 games; skipped for identical evaluators; CLI prints `black=`/`white=` stone counts
- 9×9 arena komi workaround (later removed in 2.7.1): default CLI `6.5` remapped to `3.5` in arena only
- `training/export_onnx.py`: default export is policy+value only (`--with-ownership` for three heads)
- `training/inference_server.py`: ORT `intra_op`/`inter_op` threads capped at 1 (init + reload)
- `training/train_bootstrap.py`: validation epoch under `torch.no_grad()`
- `training/requirements.txt`: `onnxruntime==1.26.0` pinned
- Sidecar path retained as fallback (`EVAL_BACKEND=sidecar`, `-eval-backend=sidecar`)

### Fixed

- `lightsail-inprocess-cycle.sh`: no Python ORT pip install on in-process path; uses `.venv311` for parity only
- Documented production RAM: Lightsail instance is `t3.small` (~2 GiB), not 4 GiB (ADR 0004)

## [2.6.0] - 2026-07-01

### Added

- ML pipeline v3: `scripts/train-loop-v3.sh` with replay buffer, manifest, monotonic promote
- Trainer `--resume` / `--init-from`, validation split, val-based `best.pt` (G1, G5)
- Self-play `-selfplay-eval heuristic|onnx|mix` with ONNX sidecar (G4)
- `training/replay.py`, `training/manifest.py`, `scripts/gating.env`
- AWS ops: `start-v3`, `stop-loop`, `fetch-all`, `seed-status` on `aws-run-arena.sh`
- Pytest suite: `training/test_train.py`, `test_replay.py`, `test_export.py`
- ADR 0003: iterative training loop

### Changed

- Sidecar: batched ORT, optional CUDA provider, SIGHUP reload, latency logging
- `remote-arena-gate.sh`: fail without checkpoint when `SELFPLAY_GAMES>0`; `ENFORCE_GATE`
- CI runs `pytest training/`

## [2.5.0] - 2026-06-30

### Added

- Real ONNX inference via HTTP sidecar (`-eval onnx`, `-onnx-url`, `-batch-size`, `-eval-timeout`)
- `SidecarBackend` + `BuildFeaturesV2` (8 planes + 4 globals); schema in `docs/model-input-schema.md`
- Bootstrap 9×9 ResNet in `training/` with `make train-bootstrap`, `make sidecar`, `make export-onnx`
- Committed fixture model `models/gofer-9x9-bootstrap.onnx`
- Self-play exports board-indexed policy (`RootPolicyBoard`) and feature tensors for training
- Arena `-arena-enhanced` flag (`none` / `baseline` / `both`); equal-config ONNX gate via `make reproduce-9x9-onnx-gate`
- Latency harness tests (`latency_test.go`); ONNX integration tests (`onnx_integration` build tag)
- CI: ONNX export, sidecar integration, optional ONNX arena smoke

### Changed

- ADR 0001 updated with sidecar protocol, fallback behavior, latency SLO table
- `BatchedEvaluator` supports configurable `reqTimeout`; `Engine.Close()` stops batch worker
- Arena archived at `.tectonix/reports/arena-9x9-onnx-v25.json` (see win rate in report)

### Not in v2.5.0

- In-process ONNX Runtime (CGO)
- Ownership / score-belief training heads
- KataGo-level 19×19 strength

## [2.0.0] - 2026-06-30

### Added

- `-arena` CLI: champion/challenger matches with Wilson CI, config hash, JSON report
- Self-play schema v1: `policy_opp`, ownership labels, `full_search` flag, JSONL export
- Paper SE-4: fast/full playout caps, forced root playouts, policy target pruning
- `BatchedEvaluator` mock inference queue (`-eval batched` / `mock-batch`)
- `BuildFeaturesV1` feature tensor + golden test (`testdata/features_golden.json`)
- Ownership labels via area-based territory flood (`OwnershipLabel`)
- ADRs: `docs/decisions/0001-inference-backend.md`, `0002-legal-moves-repr.md`

### Changed

- `BenchmarkLegalMoves` allocs/op: ~1158 → **7** (reused trial board + visit marks)
- Arena CI smoke: 20 games per push
- Optimization scorecard: **7/10** composite
- Documented strength gate: baseline heuristic (600 playouts + forced root) beats challenger heuristic (200 playouts) @ 200 games, win_rate_baseline=1.0 (see `.tectonix/reports/arena-9x9-baseline.json`)

### Not in v2.0.0

- Real ONNX/GPU inference (planned v2.5)
- Score belief PDF/CDF training labels
- KataGo-level strength or JSON analysis API

## [1.0.0] - 2026-06-30

### Added

- Interactive terminal play (`-play`) with analyze, undo, and SGF export (`-o`)
- Position analysis CLI (`-analyze`) with think-time and setup moves
- GTP 2.x subset for Sabaki/Lizzie (`-gtp`) with `time_left` think budget
- GTP SGF export on quit via `-o game.sgf`
- Engine-vs-engine demo (`-watch`)
- Self-play training samples and SGF game logs (`-selfplay`, `-sgf-dir`)
- SGF replay and export (`-sgf`, `GameLog`)
- PUCT MCTS with transposition table, parallel playouts, tree reuse
- Heuristic and uniform evaluators
- `cmd/bench` regression runner and CI gate (`make bench-check`)

### Not in v1.0.0

- Neural network training or in-process ONNX inference
- KataGo-level strength or JSON analysis API
- Full time controls (byo-yomi); `time_left` uses remaining seconds as next-move budget
- Benson pass-alive scoring (naive area-flood territory)
- Forced playouts and policy target pruning (paper M10 deferred)
