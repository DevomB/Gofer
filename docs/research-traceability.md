# Research Traceability

Maps paper/KataGo research concepts → Gofer implementation. Status updated at milestones.

**Legend:** `[PAPER]` arXiv:1902.10565 | `[POST-PAPER]` later KataGo/docs | `[GOFER]` project-only

| Research concept | Implementation target | Milestone | Status | Evidence | Risks | Deferred notes |
|------------------|----------------------|-----------|--------|----------|-------|----------------|
| PUCT/MCTS | `internal/search` | M5 | not started | — | Lock contention if parallel early | — |
| Playout cap randomization | `internal/selfplay` | M10 | deferred | — | Config complexity | [PAPER] p=0.25, N/n caps |
| Forced playouts | `internal/search` root | M10 | deferred | — | Must not poison play π | [PAPER] k=2, sqrt formula |
| Policy target pruning | `internal/search` + `internal/training` | M10 | deferred | — | Needs visit metadata export | [PAPER] |
| Global pooling | `internal/model` (training) | M11 | deferred | — | Training-only | [PAPER] not runtime engine |
| Auxiliary policy targets | `internal/training` schema | M11 | deferred | — | π_opp recording | [PAPER] w=0.15 |
| Ownership head | `internal/training` + `eval` | M11 | deferred | — | Label from scorer | [PAPER] |
| Score belief (pdf/cdf) | `internal/training` | M11 | deferred | — | Wide output head | [PAPER] |
| Score maximization | `internal/search` utility | v2+ | deferred [POST-PAPER] | — | Not in paper | Jane Street blog |
| Gating (100/200) | `internal/training` gating | M11 | deferred | — | Match infrastructure | [PAPER] Appendix E |
| SWA snapshots | external trainer | M11 | deferred | — | Not in Go engine | [PAPER] |
| Rules randomization | `internal/selfplay` | M10 | deferred | — | Multi-ruleset | [PAPER] |
| Board-size randomization | `internal/board` + selfplay | M10 | deferred | — | Variable size | [PAPER] 9–19 |
| Game-specific NN features | `internal/model` features | M11 | deferred | — | Ladders, pass-alive | [PAPER] §4.2 |
| Dirichlet root noise | `internal/search` | M5 | not started | — | Root only | [PAPER] |
| Root temperature | `internal/search` | M5 | not started | — | 1.03 | [PAPER] |
| FPU | `internal/search` | M5 | not started | — | c_FPU=0.2 | [PAPER] |
| Progressive net scaling | external training | M11 | deferred | — | — | [PAPER] |
| Chinese rules | `internal/rules` (`chinese_rules.go`) | M1 | done | `rules_test.go`, golden SGF | Seki simplification | [GOFER] v1 primary |
| Tromp-Taylor rules | `internal/rules` (`TrompTaylor`) | M2 | deferred | panic stub in `rules.go` | Paper self-play rules | [PAPER] |
| Simple ko | `internal/rules` | M1 | done | `TestSimpleKo`, `ko.sgf` | — | [GOFER] |
| Superko | `internal/rules` | M2 | deferred | — | Positional/situational | [GOFER] |
| Analysis API | `internal/analysis` | M8+ | deferred [POST-PAPER] | — | JSON batch eval | KataGo software |
| Batched evaluation | `internal/eval` | M11 | deferred | — | Latency | [PAPER] match settings |
| Graph search | `internal/search` TT/DAG | M6+ | research | — | Complexity | [GOFER] future |
| Branch generation | MCTS expand-one | M4 | not started | — | Standard MCTS | [PAPER] |
| Zobrist hashing | `internal/board` | M1 | done | `BenchmarkHashUpdate` | — | [GOFER] |
| Transposition table | `internal/search` | M6 | not started | — | — | [GOFER] |
| Heuristic evaluator | `internal/eval` | M7 | not started | — | — | [GOFER] |
| GTP protocol | `internal/gtp` | M8 | not started | — | — | [GOFER] |
| SGF replay | `internal/sgf` | M1 | started | `rules_test.go` golden, `sgf_test.go`, `testdata/*.sgf` | Minimal parser | [GOFER] tree parser M3 |
| Pass-alive optimization | `internal/rules/tromp` | M2 | deferred | — | Benson's algorithm | [PAPER] |
| Policy surprise weighting | `internal/training` | v2+ | deferred [POST-PAPER] | — | — | KataGoMethods.md |

## Review cadence

- Update status column at each milestone close
- Add evidence links (test file, bench name, PR)
- Never mark `[PAPER]` mechanism "done" without test or explicit scope cut documented
