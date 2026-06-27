# Implementation Blueprint

Module: `github.com/DevomB/gofer` | Rules v1: **Chinese** | Paper reference: arXiv:1902.10565

---

## 1. Project Goals

### Long-term vision
A serious Go engine in idiomatic Go: rules-correct, search-strong, optionally neural-guided, benchmark-driven, with self-play/training hooks — inspired by Stockfish-style engineering discipline and KataGo-style ML architecture (without copying KataGo code).

### Near-term deliverables (this tranche)
- 10 planning docs complete
- M0 repository foundation
- M1 Chinese-rules board engine with undo, tests, benchmarks
- Tectonix quality_signal ≥ 9000 at repo root before sign-off

### Explicit non-goals (v1 tranche)
- Neural network training or GPU inference
- GTP server (M8)
- MCTS (M5)
- Superko (unless trivial add in M1)
- Full SGF export
- Score maximization, JSON analysis API (post-paper, v2+)

---

## 2. Engine Modes

| Mode | Description | Milestone |
|------|-------------|-----------|
| **rules-only** | Legal moves, play, score — no search | M1 |
| **search-only** | MCTS with uniform/heuristic leaf | M4–M5 |
| **heuristic evaluator** | Hand-tuned eval for search | M7 |
| **neural-guided** | NN prior/value via `eval` | M11 |
| **analysis** | Deep eval + PV for positions | M8+ |
| **self-play generator** | Games → training samples | M10 |
| **benchmark** | `cmd/bench` + regression | M9 |

---

## 3. Milestone Ladder

### M0: Repository foundation
- **Objective:** Compilable module, CI-friendly Makefile, package skeleton
- **Acceptance:** `go build ./...`, `go test ./...` pass (may be empty tests)
- **Tests:** smoke test in `board` package
- **Benchmarks:** skeleton `BenchmarkBoardAt` in `board`
- **Bottlenecks:** N/A
- **DoD:** `go.mod`, Makefile, `.tectonix/rules.toml`, `cmd/engine`, `internal/board` coords

### M1: Rules-correct board engine (Chinese)
- **Objective:** Place/capture/ko/score/undo on 19×19
- **Acceptance:** golden tests pass; simple ko enforced; suicide allowed only if capturing
- **Tests:** table-driven legality, capture, ko, scoring; 3+ golden positions
- **Benchmarks:** MakeMove, Undo, LegalMoves, HashUpdate
- **Bottlenecks:** naive liberty scan
- **DoD:** `internal/rules/chinese` complete for standard play

### M2: Tromp-Taylor + superko + SGF replay
- **Objective:** Second ruleset; SGF import validates engine moves
- **Acceptance:** SGF corpus replays without mismatch
- **Tests:** SGF replay tests in `internal/sgf`
- **Benchmarks:** SGF replay throughput
- **DoD:** `rules/tromp` package; positional ko option

### M3: Fast state mutation
- **Objective:** Incremental groups OR proven undo faster than copy
- **Acceptance:** bench shows ≥2× MakeMove vs M1 naive OR documented ponytail with plan
- **Benchmarks:** clone vs undo comparison
- **DoD:** decision log entry for board repr

### M4: Search skeleton
- **Objective:** Selection/expansion/backup without NN
- **Tests:** fixed-seed tree shape tests
- **Benchmarks:** single playout iteration
- **DoD:** `internal/search`, `internal/tree` arenas

### M5: Basic MCTS/PUCT
- **Objective:** PUCT with heuristic eval; root Dirichlet noise optional
- **Tests:** visit count monotonicity; symmetry on empty board
- **Benchmarks:** selection step, expansion, full playout
- **DoD:** matches paper PUCT formula (c_PUCT=1.1)

### M6: Transposition table
- **Objective:** Zobrist-keyed TT; graph-search compatibility research
- **Benchmarks:** TT hit rate on repeated subtrees
- **DoD:** measurable hit rate on ladder/capture fixtures

### M7: Evaluator abstraction
- **Objective:** `eval.Evaluator` + heuristic + mock
- **Benchmarks:** eval call overhead, batch throughput stub
- **DoD:** search uses interface only at boundary

### M8: Protocol support
- **Objective:** GTP 2.x core commands; analysis CLI stub
- **Tests:** GTP command parsing
- **Benchmarks:** GTP parse throughput
- **DoD:** `genmove`, `play`, `boardsize`, `komi`

### M9: Benchmark suite
- **Objective:** `cmd/bench` runs all benches; optional regression JSON
- **DoD:** Makefile `bench`, `profile` targets work

### M10: Self-play data generation
- **Objective:** playout cap randomization; sample export schema
- **DoD:** writes JSON/flat samples from self-play

### M11: Model integration
- **Objective:** inference adapter; batched eval worker
- **DoD:** engine runs with external ONNX or mock weights

### M12: Optimization passes
- **Objective:** PGO build, escape analysis fixes, concurrency tuning
- **DoD:** documented before/after profiles; scorecard ≥6

---

## 4. Repository Layout

```
gofer/
├── cmd/engine/          # GTP (M8)
├── cmd/analyze/         # analysis CLI (M8+)
├── cmd/selfplay/        # M10
├── cmd/bench/           # M9
├── internal/
│   ├── board/           # state, coords, hash, undo
│   ├── rules/           # Ruleset interface
│   │   └── chinese/     # v1 primary
│   ├── search/          # M4+
│   ├── tree/            # node arena
│   ├── eval/            # M7+
│   ├── model/           # M11
│   ├── gtp/             # M8
│   ├── sgf/             # M2+
│   ├── analysis/        # post-paper M8+
│   ├── selfplay/        # M10
│   └── training/        # sample schema M10+
├── internal/testdata/   # golden SGF, positions
├── docs/
├── .tectonix/rules.toml
├── Makefile
├── go.mod
└── README.md
```

---

## 5. Package Contracts

### `internal/board`
- **Responsibilities:** Coordinates, `Move`, `Board` grid, `Side`, Zobrist, undo stack
- **Must NOT:** rule legality, scoring logic
- **Public API:** `New(size)`, `At(c)`, `Play(m)` via rules only — board exposes mutation primitives
- **Hot path:** `SetStone`, `RemoveStone`, `Hash`, undo push/pop
- **Tests:** coord round-trip, hash determinism

### `internal/rules`
- **Responsibilities:** `Ruleset` — `LegalMoves`, `Play`, `Score`, `Result`
- **Must NOT:** MCTS, GTP parsing
- **API:** `type Ruleset interface { ... }` at package root; implementations in subpackages
- **Hot path:** `LegalMoves` — ponytail allowed with benchmark
- **Tests:** per-ruleset golden files

### `internal/search`
- **Responsibilities:** MCTS driver, PUCT, root noise, pruning hooks
- **Must NOT:** import `training`, `model` weights
- **API:** `Search(board, eval, cfg) Move`
- **Hot path:** `Select`, `Expand`, `Backup` — no interface dispatch in inner loop
- **Tests:** seeded RNG tree tests

### `internal/eval`
- **Responsibilities:** `Evaluator` interface, heuristic, mock
- **Must NOT:** board mutation
- **API:** `Evaluate(pos) (policy, value, err)`
- **Hot path:** batch API separate from single-pos interface

### `internal/gtp`
- **Responsibilities:** stdin/stdout protocol
- **Must NOT:** search internals

---

## 6. Board Representation Options

| Option | Go fit | Search fit | Verdict |
|--------|--------|------------|---------|
| **Mutable + undo** | Excellent — slice stack | Best for MCTS | **Default [GOFER]** |
| Copy-make | Simple, idiomatic | Alloc-heavy | Benchmark vs undo |
| Immutable persistent | Functional style | Poor GC pressure | Reject for v1 |
| Bitboards | Less natural in Go | Good for 19×19 masks | Optional later for legality masks |
| Union-find groups | Standard | Speeds capture | M3 candidate |

**Zobrist:** `uint64` per (cell, color, komi bucket); increment on change.

**Superko v1:** simple ko only; hash+ko-ban point in undo record.

---

## 7. Search Architecture

```mermaid
flowchart TD
  root[Root node]
  select[PUCT select]
  expand[Expand one child]
  eval[Evaluator leaf]
  backup[Backup value]
  root --> select
  select -->|child exists| select
  select -->|frontier| expand
  expand --> eval
  eval --> backup
```

- **Node:** `VisitCount`, `ValueSum`, `Prior`, `Children []child` or index arena
- **Root:** noise blend, forced playouts, pruning on target export only
- **Concurrency:** virtual loss optional M5+; benchmark before enabling
- **Tree reuse:** retain tree on same position (GTP) — M8
- **TT:** separate from node tree — M6

---

## 8. Evaluation Layer

```go
type Evaluator interface {
    Evaluate(ctx context.Context, pos Position) (EvalResult, error)
}
```

- `HeuristicEvaluator` — material/territory proxy for M7
- `MockEvaluator` — fixed policy/value for tests
- `BatchedEvaluator` — wraps remote/ONNX with queue (M11)
- Hot path: concrete type in search loop; interface at construction only

---

## 9. Protocols And Tooling

| Protocol | Priority | Package |
|----------|----------|---------|
| GTP 2.x | M8 | `internal/gtp` |
| JSON analysis | M8+ post-paper | `internal/analysis` |
| SGF import | M2 | `internal/sgf` |
| SGF export | M2+ | `internal/sgf` |
| `cmd/bench` | M9 | `cmd/bench` |
| `cmd/selfplay` | M10 | `cmd/selfplay` |

Makefile targets: `test`, `bench`, `race`, `lint`, `profile`, `pgo-build`, `selfplay`, `analyze`

---

## 10. Test Strategy

- **Unit:** table-driven for rules, coords, hash
- **Property:** legality ⊂ board empty or capture; hash collision spot checks
- **Golden:** known positions from testdata
- **SGF replay:** moves match reference (M2)
- **Regression:** bench JSON compare (M9)
- **Seeds:** `rand.NewSource(0)` in search tests

---

## 11. Performance Strategy

### Microbenchmarks
`go test -bench=. -benchmem ./internal/board/...` etc.

### Macrobenchmarks
Full game playout, GTP session replay (M9)

### pprof
```bash
go test -cpuprofile=cpu.prof -bench=BenchmarkPlay -benchtime=3s ./internal/rules/chinese/
go tool pprof -top cpu.prof
```

### PGO
```bash
go test -cpuprofile=default.pgo -bench=BenchmarkFullGame ./...
go build -pgo=default.pgo -o bin/engine ./cmd/engine
```
Refresh when hot paths change. Microbench-only profiles can mislead — use representative mix.

### Build tags
`//go:build debug` for expensive assertions only in dev

---

## 12. Decision Log Format

Store in `docs/decisions/NNNN-title.md`:

```markdown
# Title
## Context
## Options
## Decision
## Why
## Performance impact
## Revisit trigger
```

Example triggers: bench regression >5%; new ruleset; MCTS parallelism added.
