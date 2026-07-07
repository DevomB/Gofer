# Changelog

All notable changes to Gofer are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/).

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
