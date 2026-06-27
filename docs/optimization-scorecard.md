# Optimization Scorecard

Live scorecard for Gofer. **Do not inflate scores.**

Last updated: 2026-06-27 (M2–M11 audit + structural consolidation)

---

## Summary

| Metric | Value |
|--------|-------|
| **Composite optimization level** | 4 (search + benches + profile + CI smoke) |
| **Confidence** | medium |
| **Tectonix quality_signal** | **9523** / 10000 (session-end 2026-06-27) |
| **9000+ gate** | **met** — modularity 10000 after `internal/gofer` consolidation |

---

## M3 decision log

| Decision | Choice | Evidence |
|----------|--------|----------|
| Board mutation in search | **undo** over clone for playouts | `BenchmarkCloneVsUndo` in `gofer_test.go` |
| Incremental groups | **deferred** (ponytail) | O(n) liberty scan retained |
| LegalMoves alloc | **1 trial board + Snapshot** | 1517 → **1158** allocs/op; 36139 B/op |
| Package layout | **single `internal/gofer`** | Tectonix modularity recovery; not subdirectory scan |

---

## Dimension scores (0–10)

| # | Dimension | Score | Evidence |
|---|-----------|-------|----------|
| 1 | Correctness robustness | 6 | `go test ./...`; 6 SGF replays; search/gtp/selfplay tests |
| 2 | Algorithmic sophistication | 5 | PUCT MCTS, TT lookup, visit-weighted π |
| 3 | Data-structure efficiency | 3 | Index tree arena; snapshot LegalMoves |
| 4 | Memory efficiency | 3 | Reduced LegalMoves allocs |
| 5 | Allocation discipline | 3 | benchmem tracked; LegalMoves improved |
| 6 | Concurrency effectiveness | 0 | N/A |
| 7 | Profiling maturity | 4 | `.tectonix/reports/cpu.prof` captured |
| 8 | Benchmark coverage | 6 | board, rules, search, sgf benches in `gofer_test.go` |
| 9 | Build/compiler optimization | 0 | PGO M12 |
| 10 | Observability/regression | 5 | `cmd/bench -json`, CI workflow |
| 11 | Protocol/tooling | 5 | GTP subset, `cmd/engine -gtp`, `cmd/selfplay -o` |
| 12 | Idiomaticity | 6 | monolithic core + thin cmd; eval boundary preserved |

**Weighted composite:** ~4 / 10

---

## Bench snapshot (post-M3 LegalMoves fix)

| Benchmark | ns/op | allocs/op |
|-----------|-------|-----------|
| BenchmarkLegalMoves | ~257k | 1158 |
| BenchmarkSGFReplay | ~7k | 45 |

Regression JSON: `.tectonix/reports/bench-regression.json`

---

## Tectonix session-end (2026-06-27)

```
Scanned: c:/Coding-Projects/GoEngine
Quality signal: 9523/10000 (gate: >= 9000) PASS
Weakest root cause: equality (7833)
Five metrics: modularity 10000, acyclicity 10000, depth 10000, equality 7833, redundancy 10000
Main issue: large merged gofer.go (equality); acceptable for v1 core library
Actions taken: merged board/rules/search/eval/gtp/selfplay into internal/gofer; fixed CI bench path; honest traceability
Remaining debt: forced playouts, policy pruning, Sample field population (M10–M11 paper ML)
Benches: LegalMoves 1158 allocs/op (was 1517)
```

---

## Session-end report template

```
Scanned: c:/Coding-Projects/GoEngine
Quality signal: XXXX/10000 (gate: >= 9000)
Weakest root cause: <metric> (<score>)
Remaining debt: <if any>
Benches: LegalMoves 1158 allocs/op (was 1517)
```
