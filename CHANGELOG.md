# Changelog

All notable changes to Gofer are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/).

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
