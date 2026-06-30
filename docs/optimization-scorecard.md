# Optimization Scorecard

Live scorecard for Gofer.

Last updated: 2026-06-30

---

## Summary

| Metric | Value |
|--------|-------|
| **Composite optimization level** | 5 |
| **Tectonix quality_signal** | ≥9000 at sign-off (run `tectonix health .`) |
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
| 7 | Profiling maturity | 5 | `make profile`, `make pgo-profile`, README PGO section |
| 8 | Benchmark coverage | 8 | board, rules, search (`BenchmarkBestMove`), sgf benches |
| 9 | Build/compiler optimization | 2 | PGO via `make pgo-build` (documented) |
| 10 | Observability/regression | 7 | `make bench-check`, CI regression gate (`.github/workflows/ci.yml`) |
| 11 | Protocol/tooling | 8 | GTP + SGF export; `-watch`, `-gtp -o` |
| 12 | Idiomaticity | 6 | monolithic `cmd/gofer` |

**Weighted composite:** ~5 / 10

---

## Bench snapshot

| Benchmark | allocs/op (typical) |
|-----------|---------------------|
| BenchmarkLegalMoves | 1158 |
| BenchmarkSGFReplay | 45 |
| BenchmarkBestMove | see `BenchmarkSearchParallel` |

Regression baseline: `.tectonix/reports/bench-regression.json`

CI and local: `make bench-check`

---

## Key engineering decisions

| Topic | Choice |
|-------|--------|
| Board mutation in search | undo over clone |
| Incremental groups | deferred; O(n) liberty scan |
| Package layout | single `cmd/gofer` package main |
| MCTS tree reuse | `AdvanceTree` + movable root index |
| Parallel search | root-parallel with per-playout RNG |
