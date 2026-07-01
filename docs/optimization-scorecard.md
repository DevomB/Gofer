# Optimization Scorecard

Live scorecard for Gofer.

Last updated: 2026-06-30 (v2.5 ONNX session)

---

## Summary

| Metric | Value |
|--------|-------|
| **Composite score** | 7 |
| **Tectonix quality_signal** | ≥9000 (maintain at release) |
| **Quality gate** | `quality_signal` ≥ 9000 before tagging a release |

---

## Dimension scores (0–10)

| # | Dimension | Score | Evidence |
|---|-----------|-------|----------|
| 1 | Correctness robustness | 7 | `go test ./...`; GTP, tree reuse, self-play value tests |
| 2 | Algorithmic sophistication | 7 | PUCT, TT, forced playouts, policy pruning, cap randomize |
| 3 | Data-structure efficiency | 6 | `legalityScratch`, `visitMark`, `forEachNeighbor` |
| 4 | Memory efficiency | 7 | LegalMoves **7** allocs/op (was 1158) |
| 5 | Allocation discipline | 7 | benchmem evidence in ADR 0002 |
| 6 | Concurrency effectiveness | 5 | Root-parallel MCTS wired; **~1.0×** at 200 playouts on reference laptop (overhead &gt; gain); revisit at 800+ playouts |
| 7 | Profiling maturity | 5 | `make profile`, `make pgo-profile`, README PGO section |
| 8 | Benchmark coverage | 8 | board, rules, search, ONNX sidecar mock bench |
| 9 | Build/compiler optimization | 2 | PGO via `make pgo-build` (documented) |
| 10 | Observability/regression | 8 | bench-check, arena CI, ONNX arena archive |
| 11 | Protocol/tooling | 9 | GTP, SGF, `-arena`, `-selfplay` JSONL |
| 12 | Idiomaticity | 6 | monolithic `cmd/gofer` |

**Weighted composite:** ~7 / 10

---

## Bench snapshot

| Benchmark | allocs/op (typical) |
|-----------|---------------------|
| BenchmarkLegalMoves | **7** (was 1158) |
| BenchmarkSGFReplay | 45 |
| BenchmarkEvalBatch | see regression JSON |
| BenchmarkSidecarBackendEvalBatch | mock HTTP sidecar |
| BenchmarkBatchedEvaluator | mock queue path |

Regression baseline: `.tectonix/reports/bench-regression.json`

CI and local: `make bench-check`

---

## Key engineering decisions

| Topic | Choice |
|-------|--------|
| Board mutation in search | undo over clone |
| Incremental groups | deferred per [ADR 0002](decisions/0002-legal-moves-repr.md); scratch buffers hit 7 allocs/op |
| Package layout | single `cmd/gofer` package main |
| MCTS tree reuse | `AdvanceTree` + movable root index |
| Parallel search | root-parallel with per-playout RNG |
