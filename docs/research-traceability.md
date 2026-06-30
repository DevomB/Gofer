# Research Traceability

Maps paper/KataGo research concepts → Gofer implementation. Status updated at milestones.

**Legend:** `[PAPER]` arXiv:1902.10565 | `[POST-PAPER]` later KataGo/docs | `[GOFER]` project-only

**Package note:** Engine code lives in `cmd/gofer` (single `package main` binary: `-gtp`, `-selfplay`, default demo).

| Research concept | Implementation target | Milestone | Status | Evidence | Risks | Deferred notes |
|------------------|----------------------|-----------|--------|----------|-------|----------------|
| PUCT/MCTS | `cmd/gofer` (mcts) | M5 | done | `gofer_test.go` TestPUCTFormula, TestDeterministicPlayout | Low playouts in GTP | c=1.1 |
| Playout cap randomization | `cmd/gofer` `selfplay.go` | M10 | done | `selfplay.go` `CapRandomizeP=0.25` | Config complexity | [PAPER] ponytail |
| Forced playouts | MCTS root | M10 | deferred | — | Must not poison play π | [PAPER] k=2 not implemented |
| Policy target pruning | search + Sample export | M10 | deferred | — | Needs visit metadata export | [PAPER] |
| Global pooling | external trainer | M11 | deferred | — | Training-only external | [PAPER] not runtime |
| Auxiliary policy targets | Sample schema | M11 | started | `Sample.PolicyOpp` field only | π_opp not recorded in self-play | [PAPER] w=0.15 |
| Ownership head | Sample + eval | M11 | started | `Sample.Ownership` field only | Labels not populated | [PAPER] |
| Score belief (pdf/cdf) | Sample schema | M11 | started | `Sample.ScorePDF/CDF` fields only | Not populated | [PAPER] |
| Score maximization | search utility | v2+ | deferred [POST-PAPER] | — | Not in paper | Jane Street blog |
| Gating (100/200) | GatingHarness | M11 | done | `gofer_test.go` TestGatingHarness | Match infrastructure | [PAPER] ponytail |
| SWA snapshots | external trainer | M11 | deferred | — | Not in Go engine | [PAPER] |
| Rules randomization | `selfplay.go` | M10 | done | `selfplay.go` RulesRandomize | Multi-ruleset | [PAPER] |
| Board-size randomization | `selfplay.go` | M10 | done | `selfplay.go` | Variable size | [PAPER] 9–19 |
| Game-specific NN features | model features | M11 | deferred | — | Ladders, pass-alive | [PAPER] §4.2 |
| Dirichlet root noise | `mcts.go` | M5 | done | `mcts.go` blendDirichlet | Root only | [PAPER] |
| Root temperature | SearchConfig | M5 | done | `RootTemperature` 1.03 | — | [PAPER] |
| FPU | mcts | M5 | done | `gofer_test.go` TestPUCTFormula | c_FPU=0.2 | [PAPER] |
| Progressive net scaling | external training | M11 | deferred | — | — | [PAPER] |
| Chinese rules | chinese_rules | M1 | done | `gofer_test.go`, golden SGF | Seki simplification | [GOFER] v1 primary |
| Tromp-Taylor rules | tromp_rules | M2 | done | TestTrompTaylor*, TestTrompReplayCorpus | Benson ponytail | [PAPER] |
| Simple ko | rules | M1 | done | TestSimpleKo, ko.sgf | — | [GOFER] |
| Superko | WithSuperko | M2 | done | TestSuperkoWrapper | Positional | [GOFER] |
| Analysis API | — | M8+ | deferred [POST-PAPER] | — | JSON batch eval | KataGo software |
| Batched evaluation | Inference mock | M11 | done | `inference.go`, mock batch | Latency | [PAPER] sidecar |
| Graph search | TT lookup in MCTS | M6 | done | TestTTHitRateAfterSearch | replace-always ponytail | [GOFER] |
| Branch generation | MCTS expand-one | M4 | done | mcts expand | Standard MCTS | [PAPER] |
| Zobrist hashing | Board | M1 | done | BenchmarkHashUpdate | — | [GOFER] |
| Transposition table | tt.go | M6 | done | TestTTHitRateAfterSearch | replace-always ponytail | [GOFER] |
| Heuristic evaluator | evaluator.go | M7 | done | Heuristic leaf eval | — | [GOFER] |
| GTP protocol | gtp.go | M8 | done | TestGTPBoardsize, `cmd/gofer -gtp` | Subset | [GOFER] |
| SGF replay | sgf.go | M2 | done | TestReplayCorpus, BenchmarkSGFReplay | 6 golden SGFs | [GOFER] |
| Pass-alive optimization | tromp Score | M2 | started | ponytail in Score | Benson's algorithm | [PAPER] M3 |
| Policy surprise weighting | — | v2+ | deferred [POST-PAPER] | — | — | KataGoMethods.md |

## Review cadence

- Update status column at each milestone close
- Add evidence links (test file, bench name, PR)
- Never mark `[PAPER]` mechanism "done" without test or explicit scope cut documented
