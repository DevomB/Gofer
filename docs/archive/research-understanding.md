# Research Understanding

Source: David J. Wu, *Accelerating Self-Play Learning in Go* (arXiv:1902.10565, Nov 2020). Describes **KataGo's** self-play improvements — we learn from this paper; **Gofer is not KataGo**.

Labels used below: **[PAPER]** fact from the paper; **[POST-PAPER]** later KataGo/ecosystem, not in paper; **[GOFER]** our project decision.

---

## 1. Executive Summary

**[PAPER]** The paper's goal is to reduce the compute required for AlphaZero-style self-play in Go while matching or exceeding strong baselines (ELF OpenGo, Leela Zero). KataGo reaches ELF's strength after ~19 days on <30 V100 GPUs (~1.4 GPU-years) versus ELF's ~74 GPU-years — roughly a **50×** efficiency gain versus comparable AlphaZero replications.

**Why it matters for Go engines:** Go has a huge branching factor and long horizons. Pure policy networks are weak; MCTS + neural guidance is the modern paradigm. This paper shows that **training efficiency** and **engine/runtime efficiency** are first-class — not afterthoughts. Techniques here directly inform how Gofer should structure search, self-play data generation, evaluation heads, and optimization discipline.

**Directly relevant to Gofer:**
- PUCT/MCTS search loop and root exploration (noise, forced playouts, policy target pruning)
- Playout cap randomization for self-play throughput
- Neural net architecture patterns (global pooling, multi-head outputs)
- Auxiliary targets (opponent policy, ownership, score belief)
- Progressive model scaling and gating between training generations
- Rules/board-size randomization for generalization
- Performance-conscious threading and batching (match settings in paper)

**Research-only or deferred for Gofer v1:**
- Full GPU training pipeline and convolutional residual net training **[PAPER]** — deferred to M10–M11
- Global pooling / auxiliary heads implementation **[PAPER]** — training-side; engine needs inference adapter only at M11
- Tromp-Taylor self-play rules **[PAPER]** — Gofer uses Chinese rules v1 **[GOFER]**, Tromp-Taylor M2+
- Dynamic score maximization, JSON analysis API **[POST-PAPER]** — v2+ protocol/search objectives
- Graph search beyond standard MCTS tree **[GOFER]** — future exploration

---

## 2. Engine Concepts Extracted

### Game state representation
- **[PAPER]** Board as spatial tensor input (b×b×18 binary features) plus 10 global scalars (komi, ko type, suicide, history, etc.)
- **[GOFER]** Runtime board: dense `[]Stone` grid, group/liberty tracking, Zobrist hash, undo stack — not the NN feature tensor (deferred v2: `inference.go` / external feature builder)

### Search
- **[PAPER]** MCTS with PUCT selection; single-node expansion per playout; backup of value estimates
- Root-only Dirichlet noise and temperature; forced playouts + policy target pruning at root only
- Playout caps: full search (600→1000 nodes) on fraction p of moves; fast search (100→200) otherwise

### Neural network guidance
- **[PAPER]** ResNet trunk; policy prior P(c); value estimate V from outcome head
- FPU for unvisited children: V(c) = V(n) - c_FPU·√P_explored, c_FPU=0.2 (0 at noisy root)

### Output heads
- Policy (current player + pass), opponent auxiliary policy
- Value: win/loss/(no result), score mean/std, ownership per intersection, score belief distribution (pdf + cdf)

### Auxiliary targets
- Opponent next-move policy (regularization, w=0.15)
- Ownership per point (credit assignment)
- Score belief pdf/cdf (sharper gradients than binary win/loss alone)

### Training loop
- Self-play → sample positions → SGD (batch 256, momentum 0.9)
- Moving window replay (250k → 22M samples)
- Progressive net size upscaling when loss catches up
- SWA snapshots + gating (100/200 wins vs current net)

### Gating
- Candidate nets from EMA of weight snapshots must beat incumbent in 200 games before promotion to self-play

### Inference backends
- **[PAPER]** GPU batched inference during matches (batch size 16 mentioned)
- **[GOFER]** Pluggable `eval.Evaluator` — heuristic/mock first; ONNX/external later — not specified in paper

### Performance tuning
- **[PAPER]** Thread count and batch size materially affect wall-clock (2–3× cited in KataGo docs ecosystem)
- **[GOFER]** Benchmark threads/batch size; never guess

### Data generation
- Self-play games with mixed full/fast search; only full-search positions in training set for policy
- ~241M samples from 4.2M games in main run

### Rules handling
- **[PAPER]** Tromp-Taylor variant; random ko/suicide/komi; smaller boards mixed in
- **[GOFER]** Chinese rules primary v1; rules interface for Tromp-Taylor M2+

### Score maximization
- **[POST-PAPER]** Jane Street / later KataGo: secondary objective for handicap play — **not in this paper**; defer to search v2+

### Analysis engine support
- **[POST-PAPER]** KataGo JSON analysis engine with batched position eval — **not in this paper**; deferred v2+ (no `cmd/gofer -analyze` JSON API yet)

---

## 3. Architecture Extracted From The Paper

### Self-play pipeline
```
Initialize random weights → loop:
  Self-play games (MCTS + NN, mixed caps)
  → record training samples (policy targets pruned, aux targets)
  → train on replay window
  → periodic SWA candidate → gating matches → maybe promote net
  → optionally upscale net architecture
```

### Search loop (per playout)
1. Start at root; while child exists, select argmax PUCT(c)
2. At frontier, expand one child, evaluate leaf with NN (or terminal outcome)
3. Backup value along path

### Policy/value interaction
- NN provides prior P and leaf value; MCTS refines policy target from visit counts
- Value head trains on game outcome (+ aux ownership/score)
- Policy target **decoupled** from exploration via pruning

### Root-specific vs tree-wide
| Mechanism | Scope |
|-----------|-------|
| Dirichlet noise | Root only |
| Temperature 1.03 | Root only |
| Forced playouts | Root children only |
| Policy target pruning | Root visit distribution |
| FPU c_FPU=0 | Root when noise on |
| PUCT formula | Entire tree |

### Randomness injection
- Root Dirichlet noise (exploration)
- Playout cap randomization per turn (full vs fast)
- Rules/board-size randomization across games (training)
- Opening temperature in evaluation matches (external to core loop)

### Pruning / target modification
- Forced visits subtracted from policy target unless move proven good by PUCT
- Single-visit children pruned from target

### Model size scaling
- Start small (6×96), grow to (10×128), (15×192), (20×256) when loss catches up
- Concurrent training of next size on same data before switch

---

## 4. Features To Build

### Must-build v1 (M0–M3)
- Chinese rules engine (legal moves, capture, ko, scoring)
- Board state + undo + Zobrist hash
- GTP skeleton (M8 but protocol design now)
- Benchmark harness
- Evaluator interface + heuristic/mock

### Should-build v2 (M4–M9)
- PUCT MCTS with root noise
- Transposition table
- SGF import/replay validation
- Basic analysis CLI
- Bench regression tracking

### Advanced / later (M10–M12)
- Self-play with playout cap randomization
- Policy target pruning + forced playouts
- Training sample schema + gating harness
- Neural inference integration
- Tromp-Taylor rules + paper-aligned features (ladders, pass-alive)

### Research / optional
- Global pooling in custom training net (training pipeline, not engine core)
- Score maximization play objective **[POST-PAPER]**
- Graph search / transposition-aware DAG search
- JSON batched analysis API **[POST-PAPER]**
- Policy surprise weighting **[POST-PAPER KataGoMethods]**

---

## 5. Important Algorithms And Mechanisms

| Mechanism | What | Why | Signal improved | Runtime / Training / Both | Difficulty | Dependencies |
|-----------|------|-----|-----------------|---------------------------|------------|--------------|
| **PUCT/MCTS** | UCT with policy prior | Focus search on plausible moves | Policy + value quality | Both | Medium | Board, legal moves, eval |
| **Playout cap randomization** | Full search p=20% of moves (fast=50, full=200 via `gating.env`), fast otherwise | More games for value; quality policy on full moves | Value + policy balance | Training | Medium | Search, self-play scheduler |
| **Forced playouts** | Minimum root visits per child | Discover noise-suggested moves | Exploration | Training (search runtime) | Medium | Root PUCT override |
| **Policy target pruning** | Remove forced visits from training π | Decouple π from exploration noise | Policy convergence | Training | Medium | Visit stats at root |
| **Global pooling** | Mean/size-scaled mean/max → channel bias | Non-local context (ko, global strategy) | NN representation | Training (+ inference) | Hard | Conv net training code |
| **Aux policy targets** | Predict opponent's reply | Regularization | Policy generalization | Training | Easy-Medium | Multi-head loss |
| **Ownership targets** | Per-point final ownership | Localized credit assignment | Value/ownership | Training | Medium | Scoring engine for labels |
| **Score belief heads** | Pdf + cdf over final margin | Finer than win/loss | Value calibration | Training | Hard | Scoring, wide output head |
| **Gating** | 100/200 win test for new net | Prevent policy collapse | Training stability | Training | Medium | Match engine, two nets |
| **Dynamic score maximization** | **[POST-PAPER]** Play for points | Handicap strength | Match play | Runtime search | Unknown | Score head, search utility |
| **Rules randomization** | Vary ko/suicide/komi | Single net generalizes | Training robustness | Training | Medium | Rules engine variants |
| **Board-size randomization** | 9–19 mixed | Generalization | Training | Training | Medium | Variable-size board |
| **Branch generation** | Standard MCTS expand-one-child | Tree growth | Search | Runtime | Low | Node storage |

---

## 6. What We Can Implement Without Full ML Infrastructure

Immediately buildable:
- **Rules engine** — Chinese v1, Tromp-Taylor hooks
- **GTP protocol** — parse/play commands (M8)
- **Search skeleton** — selection/expansion/backup without NN
- **Playout engine** — random/heuristic playouts for testing
- **Benchmark harness** — `cmd/bench`, `testing.B`, Makefile targets
- **Inference abstraction** — `eval.Evaluator` with `Heuristic` and `Mock` implementations
- **Test corpus** — golden positions, SGF replay tests
- **Zobrist hashing** — transposition-ready board

---

## 7. What Requires ML / GPU / External Tooling

Requires external stack:
- **Training data generation at scale** — GPU self-play farm (M10+)
- **Neural net training** — PyTorch/JAX or sidecar; not pure Go requirement
- **GPU inference** — CUDA/TensorRT/ONNX Runtime; batching worker
- **Model format** — ONNX, native checkpoint, or gRPC to Python trainer
- **Gating matches** — two nets + many parallel games

**Pure Go vs bridge tradeoffs:**
| Approach | Pros | Cons |
|----------|------|------|
| Pure Go inference | Single binary, deploy simple | No training; limited GPU kernels |
| ONNX Runtime via cgo | Portable inference | Build complexity |
| External process (gRPC/stdio) | Use any trainer | Latency, ops burden |

**[GOFER]** Start with mock/heuristic eval; add ONNX or sidecar at M11 when model exists.

---

## 8. Ambiguities / Open Questions

| Question | Severity | Resolution path | Impact on order |
|----------|----------|-----------------|-----------------|
| Chinese vs Tromp-Taylor scoring deltas in training labels | Medium | Implement both scorers; document diff | M1 Chinese, M2 Tromp-Taylor |
| Superko variant for v1 (simple ko only?) | Medium | **[GOFER]** simple ko v1; positional/situational M2+ | M1 scope |
| Score maximization mechanics | Low (deferred) | Study post-paper KataGo docs | v2+ search |
| Analysis API schema | Low | Mirror KataGo JSON when needed | M8+ |
| Graph search definition | Medium | Prototype after TT + MCTS stable | M6+ |
| Inference format | High (before M11) | Decide when first model exists | M11 blocker |
| Incremental liberties vs recompute | Medium | Benchmark on capture-heavy positions | M3 |
| Root parallelism vs tree parallelism | Medium | Profile lock contention | M5+ |

Do not hallucinate answers; track in `research-traceability.md`.

---

## 9. Implementation Implications For A Go Codebase

> **v1 layout (2026-06):** All engine code is in `cmd/gofer` (`package main`). Logical boundaries below map to files, not separate packages.

| Area | `cmd/gofer` files | Responsibility |
|------|-------------------|----------------|
| board | `board.go`, `move.go`, `point.go`, `zobrist.go`, `groups.go` | Grid, colors, moves, komi, hash, undo — **rules-agnostic** |
| rules | `chinese_rules.go`, `tromp_rules.go`, `superko.go`, `rules.go` | `Ruleset` interface; legal moves, scoring |
| search | `mcts.go`, `arena.go`, `tt.go` | MCTS/PUCT, playout caps, root noise — **no NN types** |
| tree | `arena.go` | Node arena, visit counts, child slices |
| eval | `evaluator.go` | `Evaluator` interface; heuristic, uniform, mock |
| analysis | `cli.go` (`-analyze`) | Position analysis CLI (post-paper JSON API deferred) |
| gtp | `gtp.go` | GTP 2.x command loop |
| selfplay | `selfplay.go`, `sample.go` | Game generation, cap randomization, sample export |
| training | `sample.go`, `inference.go` (mock) | Sample schema, gating — not in-engine training |
| bench | `cmd/bench` | Benchmark regression runner (exec, no import of gofer) |
| model | `inference.go` (mock) | Feature/inference adapters (M11, external) |
| sgf | `sgf.go`, `sgf_parse.go`, `game.go` | Parse/replay/export |
| CLI | `main.go`, `cmdline.go`, `cli.go` | `-play`, `-analyze`, `-watch`, `-selfplay`, `-gtp`, `-sgf` |

**Dependency rule:** types compose in one `package main`; `eval` injected into `Engine`; never import training from hot board paths.

---

## 10. Performance Implications

Likely hotspots:
1. **Legal move generation** — O(n²) naive; dominates if unoptimized
2. **Capture/liberty resolution** — BFS/union-find on dense fights
3. **Board undo** — must be O(1) amortized per move for search
4. **Ko/superko detection** — hash + history
5. **Node allocation** — arena vs pointer tree
6. **Transposition lookup** — hash map vs open addressing
7. **Policy normalization** — softmax over 362 moves
8. **Inference batching** — queue + worker goroutine
9. **Root parallel MCTS** — virtual loss, lock contention
10. **Scoring** — flood-fill territory at endgame

---

## 11. Optimization Opportunities

**Algorithmic:** TT, playout caps, policy pruning (training), better move ordering

**Data structure:** Flat board, incremental groups, index-based tree nodes, open-addressing TT

**Memory:** Arena allocators, pre-sized child slices, avoid board copy per node

**Concurrency:** Root parallel only first; batched inference worker; avoid channels in select loop

**Compiler/build:** `-trimpath`, PGO with representative `pprof` CPU profile (Go docs: ~2–14% on representative programs)

**Profiling-informed:** Every hot-path change needs `pprof` + `benchmem` before/after

---

## 12. Optimization Risks

- Premature abstraction (interface every function)
- Pointer-heavy MCTS tree → GC pressure
- `map` in innermost playout loop
- Goroutine per playout
- Channel-based search coordination
- Hidden board copies in `Play()` API
- Copy-make trees without measuring undo alternative
- Claiming optimization without benchmarks
- Gaming Tectonix score (scope manipulation)

---

## 13. Development Order Recommendation

1. **M0** — `go.mod`, Makefile, package skeleton, Tectonix rules
2. **M1** — Chinese rules + undo + tests + first benches
3. **M2** — Tromp-Taylor + superko options + SGF replay tests
4. **M3** — Fast move gen + liberty incremental experiments (benchmarked)
5. **M4** — Search skeleton (no NN)
6. **M5** — PUCT MCTS + root noise
7. **M6** — Transposition table
8. **M7** — Evaluator abstraction + heuristic
9. **M8** — GTP + basic analysis
10. **M9** — Bench suite + regression thresholds
11. **M10** — Self-play + cap randomization
12. **M11** — Model integration + training sample export
13. **M12** — Profile-guided optimization passes

Rationale: correctness before search; search before ML; measurement throughout.

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **MCTS** | Monte Carlo Tree Search — builds a game tree via simulated playouts |
| **PUCT** | Polynomial UCT — UCT with policy prior term (AlphaZero selection formula) |
| **Playout** | One descent from root to leaf + backup |
| **Policy prior** | NN probability distribution over moves |
| **Policy target (π)** | Training label derived from MCTS visit distribution |
| **Value target** | Win/loss (and auxiliaries) from game outcome |
| **FPU** | First-play urgency — prior for unvisited children |
| **Dirichlet noise** | Root exploration noise blended into policy prior |
| **Playout cap** | Max tree nodes per move (full vs fast) |
| **Policy target pruning** | Remove exploration-only visits from π |
| **Forced playouts** | Minimum root visits per child for exploration |
| **Global pooling** | Aggregate spatial features to bias all locations |
| **Gating** | Champion/challenger net promotion via match play |
| **SWA** | Stochastic Weight Averaging — average checkpoints for stability |
| **Komi** | Points added to compensate White for moving second |
| **Ko** | Repetition rule preventing immediate recapture |
| **Area scoring** | Chinese: stones + surrounded territory count |
| **Territory scoring** | Japanese: empty territory only (not v1) |
| **Zobrist hashing** | XOR hash for board positions |
| **TT** | Transposition table — cache subtrees by hash |
| **Virtual loss** | Temporary visit inflation for parallel search |
| **PGO** | Profile-Guided Optimization — compiler uses CPU profile |
| **Known shortcut** | Comment documenting intentional simplification + upgrade path |
