# Go Engineering Standard

How to write idiomatic, performance-aware Go in Gofer.

---

## Philosophy

- Idiomatic first, clever second.
- Explicit over magical.
- Small focused packages; dependency direction is documentation.
- No abstraction until pressure proves need (lazy-developer ladder).
- Optimize hot paths only with benchmark or profile evidence.
- Do not import Java/C++ OOP habits (deep inheritance, factory forests).

### Lazy-developer ladder (pre-edit checklist)

1. Does this need to be built? (YAGNI)
2. Does it already exist here? Reuse it.
3. Does stdlib do it?
4. Does an installed dep do it?
5. Can it be one line?
6. Only then: minimum new code.

### Ponytail comments

Any intentional shortcut with a known ceiling:

```go
// ponytail: naive liberty flood-fill per capture check.
// Ceiling: O(n) per candidate move on dense boards.
// Upgrade: incremental group liberties (backlog-core-engine).
```

### One runnable check

Non-trivial logic ships with the smallest test or bench that fails if logic breaks. No heavy fixtures for one-liners.

---

## Style Rules

- **Naming:** `MixedCaps`, no stuttering (`board.Board` ok as package type `Board`).
- **Packages:** one responsibility; name matches directory; no `util` junk drawer.
- **Errors:** return `error`; wrap with `%w` at boundaries; no panic in rules engine except impossible internal states in tests.
- **Constructors:** `NewBoard(size)` when zero value insufficient; prefer useful zero values for DTOs.
- **Interfaces:** at system boundaries (`eval.Evaluator`, `rules.Ruleset`); not in select loop.
- **Receivers:** pointer receiver if mutation; value if small immutable.
- **Slices:** preallocate with known capacity `make([]T, 0, maxMoves)`.
- **Arrays:** use for fixed topology when it helps (e.g. `[362]float32` policy later).
- **Sync:** `sync.Mutex` over channels for hot-path stats; channels for work queues only.
- **Tests:** table-driven `tests := []struct{...}`; `t.Parallel()` only when safe.
- **Benchmarks:** `BenchmarkFoo` in `*_test.go`; stable names across commits.

---

## Performance-Sensitive Go Rules

- Avoid interface dispatch in inner search loop unless benchmarked acceptable.
- Avoid `map` in per-move inner loop; use slices indexed by `Point.Index()`.
- Minimize heap escapes: check `go build -gcflags=-m` on hot packages.
- Pointer vs value: small nodes as indices into arena, not `*Node` forests.
- Preallocate move lists, child slices in tree expansion.
- Contiguous memory: `[]Stone` board, SoA for node stats if needed.
- No channel choreography in playout loop.
- Goroutines: root parallel MCTS or inference worker — measure wall-clock win.
- Beware false sharing: pad counters if multithreaded visit stats.
- `sync.Pool`: only with measured allocs/op reduction; easy to make slower.
- Cache: row-major index `y*size+x`.
- Generics: ok for small helpers; skepticism in hottest loops until Go 1.22+ proves out.
- No reflection in hot path.

---

## Benchmarking Rules

- Every optimization claim needs `go test -bench=. -benchmem` before/after.
- Stable fixtures in `cmd/gofer/testdata` or const positions in `_test.go`.
- Report `ns/op`, `B/op`, `allocs/op`.
- Distinguish throughput (positions/sec) vs latency (ms/move).
- Synthetic microbenches + representative macro (19×19 capture fight).
- Keep benchmark names stable for trend tracking.

---

## Profiling Rules

```bash
go test -bench=BenchmarkLegalMoves -cpuprofile=cpu.prof ./cmd/gofer/
go tool pprof -top cpu.prof
```

- CPU + alloc profiles for hot-path changes.
- Store profiles under `.tectonix/reports/` or `profiles/` with date.
- Compare before/after for every non-trivial optimization.
- Annotate workload (board size, position type).

---

## Build Rules

| Target | Command |
|--------|---------|
| Normal | `go build ./...` |
| Test | `go test ./...` |
| Race | `go test -race ./...` |
| Bench | `go test -bench=. -benchmem ./...` |
| Release | `go build -ldflags="-s -w"` |
| PGO | See below |

### Profile-Guided Optimization (PGO)

Go supports feeding representative CPU profiles to the compiler (`default.pgo`). Documented gains on representative programs are roughly **2%–14%** — not a substitute for algorithm choice.

Workflow:
1. Generate profile from representative workload (`make pgo-profile` or `cmd/bench`).
2. Copy to `default.pgo` at module root (or use `make pgo-profile` which writes it directly).
3. `go build -pgo=default.pgo -o bin/gofer ./cmd/gofer`
4. Re-bench; record delta in scorecard.
5. Refresh PGO when search/eval code changes materially.

**Caveat:** Microbench-only PGO can mis-optimize other paths — prefer macro workload.

---

## Code Review Rules

- Correctness: tests for new rules/search behavior.
- Hot-path changes: bench numbers required.
- Reject "looks faster" without data.
- Tradeoff note required when code becomes less obvious.
- Ponytail required for known-ceiling shortcuts.
- Tectonix: no acyclicity/modularity regression without justification.
- No new dependencies without reason in decision log.

---

## Tectonix Integration

| When | Action |
|------|--------|
| Session start | `tectonix session-start .` |
| Before refactor | `tectonix health .`, `git-stats` on hotspots |
| Before risky edit | `tectonix test-gaps .` |
| Architecture change | `tectonix dsm .`, `check-rules .` |
| Session end | `tectonix session-end .` |
| Milestone done | `quality_signal >= 9000` at repo root |

Report weakest root cause and what structural fix was applied.

---

## Makefile Conventions (M0+)

```makefile
test:  go test ./...
bench: go test -bench=. -benchmem ./...
race:  go test -race ./...
lint:  go vet ./...
profile: go test -bench=BenchmarkLegalMoves -cpuprofile=cpu.prof ./cmd/gofer/
```

Extend as packages land; no fake targets.
