# Optimization Scorecard

Live scorecard for Gofer. **Do not inflate scores.**

Last updated: 2026-06-27 (M0+M1 tranche audit)

---

## Summary

| Metric | Value |
|--------|-------|
| **Composite optimization level** | 2 (rules engine + benches; no profiling regression yet) |
| **Confidence** | medium (tests green; limited bench history) |
| **Tectonix quality_signal** | **7081** / 10000 (repo root rescan, post-audit fixes) |
| **9000+ gate** | **blocked** — modularity 2789; see remediation below |

---

## Dimension scores (0–10)

| # | Dimension | Score | Confidence | Evidence | Missing instrumentation |
|---|-----------|-------|------------|----------|------------------------|
| 1 | Correctness robustness | 4 | medium | `go test ./...` green; capture/ko/suicide/pass golden | More SGF corpus (M2) |
| 2 | Algorithmic sophistication | 1 | high | No search | MCTS M5 |
| 3 | Data-structure efficiency | 2 | low | Mutable board + undo; clone bench | M3 incremental groups decision |
| 4 | Memory efficiency | 2 | low | `benchmem` on board/rules benches | Escape analysis |
| 5 | Allocation discipline | 2 | low | LegalMoves clones per candidate | Profile-guided fixes |
| 6 | Concurrency effectiveness | 0 | high | N/A | — |
| 7 | Profiling maturity | 1 | medium | Makefile `profile` target | Representative cpu.prof captured |
| 8 | Benchmark coverage | 5 | medium | `BenchmarkMakeMove`, `BenchmarkLegalMoves`, `BenchmarkCaptureHeavy`, `BenchmarkHashUpdate` | CI regression JSON (M9) |
| 9 | Build/compiler optimization | 0 | high | — | PGO (M12) |
| 10 | Observability/regression | 1 | medium | Tectonix session reports | CI bench gate |
| 11 | Protocol/tooling | 1 | medium | `cmd/engine`, Makefile | GTP M8 |
| 12 | Idiomaticity | 6 | medium | `board`/`rules` split, ponytail comments | — |

**Weighted composite:** ~2 / 10

---

## Tectonix tracking

| Field | Value |
|-------|-------|
| Session baseline | `.tectonix/session-baseline.json` |
| M1 health snapshot | `.tectonix/reports/health-m1.json` |
| Weakest root cause | **modularity (2789)** |
| Bottleneck | modularity |
| Gate status | **blocked** (< 9000) |

### Root causes (2026-06-27 rescan)

| Metric | Score |
|--------|-------|
| acyclicity | 10000 |
| depth | 8000 |
| equality | 6190 |
| modularity | 2789 |
| redundancy | 10000 |

**Main issue:** Small layered repo (`board` ← `rules` ← `cmd`) has high cross-module edge ratio; Tectonix modularity metric penalizes thin multi-package cores until more cohesive modules exist.

**Remediation (real fixes, not gaming):**
1. Grow `internal/search` + `internal/tree` as sibling consumers of `board` (M4+) — increases intra-layer cohesion
2. Promote `sgf_parse` → `internal/sgf` when parser grows (M2) with tests wired
3. Add `internal/eval` boundary before ML (M7) — stabilizes dependency fan-in to `board`
4. Re-run `tectonix session-end` after M4 skeleton lands; expect modularity lift with search packages

---

## Next highest-leverage improvements (ordered)

1. M2: Tromp-Taylor + fuller SGF parser + superko option
2. First `pprof` capture on `BenchmarkLegalMoves` → document in scorecard
3. M3: bench clone vs undo; decide incremental groups
4. M4: search skeleton — primary path to Tectonix modularity recovery
5. M9: `cmd/bench` regression JSON + CI

---

## Known bottlenecks

| Area | Status | Notes |
|------|--------|-------|
| Liberty scan | active | ponytail in `chinese_rules.go` |
| LegalMoves O(n²) clones | active | `BenchmarkLegalMoves` |
| Search | not built | — |

---

## Do not optimize yet

- MCTS parallelism
- PGO (until representative game bench)
- NN inference batching
- Changes solely to raise Tectonix score without structural meaning

---

## Optimization / performance / complexity debt

| Debt | Type | Notes |
|------|------|-------|
| Tectonix < 9000 at M1 | structural | Documented; unblock expected M4+ |
| Seki scoring simplified | optimization | ponytail in Score |
| Ko: single-stone capture ban | correctness | ponytail; strict liberty ko in M2 |
| SGF parser minimal | complexity | promote M2 |

---

## Score history

| Date | Composite | quality_signal | Notes |
|------|-----------|----------------|-------|
| 2026-06-27 | 0 | 10000 (0 Go files) | Docs baseline |
| 2026-06-27 | 2 | 6731 | M0+M1 code; gate blocked |

---

## Sign-off checklist (M1 tranche)

- [x] `go test ./...` green
- [x] Golden ko/capture/pass tests pass (`internal/testdata/*.sgf`)
- [x] Benchmarks exist (`internal/board`, `internal/rules`)
- [x] `cmd/bench` runs all benches
- [ ] `tectonix rescan` quality_signal ≥ 9000 — **blocked (6731)**
- [x] `tectonix session-end` report filed
- [x] Traceability rows updated for M1 items
- [x] This scorecard updated with measured evidence

**M1 functional sign-off:** yes (rules engine works). **M1 Tectonix gate:** not met — continue structural work at M2/M4 without declaring tranche fully closed per plan §6.3.
