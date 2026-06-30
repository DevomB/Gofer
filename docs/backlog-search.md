# Backlog: Search

Epics for `cmd/gofer` search (`mcts.go`, `arena.go`, `tt.go`), MCTS/PUCT, paper mechanisms (deferred to M5+).

---

## Epic SE-1: Tree storage (M4)

| ID | Task | Deps | Acceptance | Bench | Test |
|----|------|------|------------|-------|------|
| SE-1.1 | Node arena (index-based) | board, rules | No per-node heap alloc in steady state | `BenchmarkNodeAlloc` | arena reuse |
| SE-1.2 | Child slice pre-cap | SE-1.1 | Expand adds one child | — | child count |
| SE-1.3 | Pointer vs index benchmark | SE-1.1 | Decision log entry | compare benches | — |

**Risk:** Arena growth — doubling slice reallocation.

---

## Epic SE-2: Search skeleton (M4)

| ID | Task | Deps | Acceptance | Bench | Test |
|----|------|------|------------|-------|------|
| SE-2.1 | Selection to leaf | SE-1 | Deterministic with seed | `BenchmarkSelect` | tree shape |
| SE-2.2 | Single-child expansion | SE-2.1 | One new node per playout | `BenchmarkExpand` | — |
| SE-2.3 | Value backup | SE-2.2 | Mean updates correct | — | numeric |
| SE-2.4 | Full playout iteration | SE-2.3 | N playouts complete | `BenchmarkPlayout` | — |

---

## Epic SE-3: PUCT MCTS (M5)

| ID | Task | Deps | Acceptance | Bench | Test |
|----|------|------|------------|-------|------|
| SE-3.1 | PUCT formula c=1.1 | SE-2 | Matches paper | — | numeric |
| SE-3.2 | FPU unvisited children | SE-3.1 | c_FPU=0.2 | — | — |
| SE-3.3 | Root Dirichlet noise | SE-3.1 | 0.75/0.25 blend | — | statistical |
| SE-3.4 | Root temperature 1.03 | SE-3.3 | Optional flag | — | — |
| SE-3.5 | Integrate `eval.Evaluator` | eval M7 | Search calls eval at leaf | `BenchmarkEvalOverhead` | mock eval |

---

## Epic SE-4: Paper training mechanisms (M10)

| ID | Task | Deps | Acceptance |
|----|------|------|------------|
| SE-4.1 | Playout cap randomization | SE-3, selfplay | p=0.25 full, fast otherwise |
| SE-4.2 | Forced root playouts | SE-3 | k=2, sqrt formula |
| SE-4.3 | Policy target pruning | SE-4.2 | Pruned π export |
| SE-4.4 | Tree reuse on same position | SE-3, gtp | GTP genmove retains tree |

---

## Epic SE-5: Transposition & graph search (M6+)

| ID | Task | Deps | Acceptance | Bench |
|----|------|------|------------|-------|
| SE-5.1 | Zobrist TT | board hash | Hit on transposition | `BenchmarkTTLookup` |
| SE-5.2 | TT store depth/bounds | SE-5.1 | Correct cutoff | — |
| SE-5.3 | Graph-search prototype | SE-5.1 | DAG merge research doc | — |

---

## Epic SE-6: Concurrency (M5+)

| ID | Task | Deps | Acceptance | Risk |
|----|------|------|------------|------|
| SE-6.1 | Root parallel playouts | SE-3 | Speedup on `-bench` | lock contention |
| SE-6.2 | Virtual loss | SE-6.1 | No duplicate expand storm | measure |
| SE-6.3 | Mutex profile | SE-6.1 | `mutex.prof` analyzed | — |

**Benchmark hook:** `BenchmarkSearchParallel` with `GOMAXPROCS` sweep.

---

## Epic SE-7: Post-paper search (v2+)

| ID | Task | Status |
|----|------|--------|
| SE-7.1 | Dynamic score maximization | deferred [POST-PAPER] |
| SE-7.2 | LCB move selection | deferred |
