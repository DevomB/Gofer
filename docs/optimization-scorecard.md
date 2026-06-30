# Optimization Scorecard

Live scorecard for Gofer.

Last updated: 2026-06-30 (v2 baseline session)

---

## Summary

| Metric | Value |
|--------|-------|
| **Composite score** | 7 |
| **Tectonix quality_signal** | 9141 (2026-06-30) |
| **Equality root cause** | ~6300 typical for monolithic `cmd/gofer`; release gate is **quality_signal**, not per-metric 9000 |
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
| 8 | Benchmark coverage | 8 | board, rules, search (`BenchmarkBestMove`), sgf benches |
| 9 | Build/compiler optimization | 2 | PGO via `make pgo-build` (documented) |
| 10 | Observability/regression | 8 | `make bench-check`, arena CI smoke, Wilson CI |
| 11 | Protocol/tooling | 9 | GTP, SGF, `-arena`, `-selfplay` JSONL |
| 12 | Idiomaticity | 6 | monolithic `cmd/gofer` |

**Weighted composite:** ~7 / 10

---

## Bench snapshot

| Benchmark | allocs/op (typical) |
|-----------|---------------------|
| BenchmarkLegalMoves | **7** (was 1158) |
| BenchmarkSGFReplay | 45 |
| BenchmarkBestMove | see `BenchmarkSearchParallel` (50 playouts, 4 workers) |
| BenchmarkSearchWorkers1/8 | scaling probe at 200 playouts (not in regression gate) |

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
