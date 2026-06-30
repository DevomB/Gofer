# Optimization Scorecard

Live scorecard for Gofer.

Last updated: 2026-06-29

---

## Summary

| Metric | Value |
|--------|-------|
| **Composite optimization level** | 5 |
| **Tectonix quality_signal** | see latest `tectonix health .` |
| **9000+ gate** | required at sign-off |

---

## Dimension scores (0–10)

| # | Dimension | Score | Evidence |
|---|-----------|-------|----------|
| 1 | Correctness robustness | 7 | `go test ./...`; GTP, tree reuse, self-play value tests |
| 2 | Algorithmic sophistication | 5 | PUCT MCTS, TT, tree reuse, parallel playouts |
| 3 | Data-structure efficiency | 4 | Index arena, snapshot LegalMoves |
| 4 | Memory efficiency | 3 | LegalMoves 1158 allocs/op |
| 5 | Allocation discipline | 3 | benchmem tracked |
| 6 | Concurrency effectiveness | 4 | Root-parallel MCTS with virtual loss |
| 7 | Profiling maturity | 4 | `make profile`, `make pgo-profile` |
| 8 | Benchmark coverage | 7 | board, rules, search, sgf benches |
| 9 | Build/compiler optimization | 1 | PGO via `make pgo-build` |
| 10 | Observability/regression | 6 | `make bench-check`, CI regression gate |
| 11 | Protocol/tooling | 7 | GTP extended; `-gtp-playouts`, `-eval` |
| 12 | Idiomaticity | 6 | monolithic `cmd/gofer` |

**Weighted composite:** ~5 / 10

---

## Bench snapshot

| Benchmark | allocs/op (typical) |
|-----------|---------------------|
| BenchmarkLegalMoves | 1158 |
| BenchmarkSGFReplay | 45 |

Regression baseline: `.tectonix/reports/bench-regression.json`

---

## Key engineering decisions

| Topic | Choice |
|-------|--------|
| Board mutation in search | undo over clone |
| Incremental groups | deferred; O(n) liberty scan |
| Package layout | single `cmd/gofer` package main |
| MCTS tree reuse | `AdvanceTree` + movable root index |
| Parallel search | root-parallel with per-playout RNG |
