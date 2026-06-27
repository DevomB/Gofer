# Backlog: ML Integration

Epics for `internal/eval`, `internal/model`, `internal/training`, `internal/selfplay`. Training deferred; interfaces first.

---

## Epic ML-1: Evaluator boundary (M7)

| ID | Task | Deps | Acceptance | Bench | Test |
|----|------|------|------------|-------|------|
| ML-1.1 | `eval.Evaluator` interface | board | Policy+value from position | — | compile |
| ML-1.2 | `HeuristicEvaluator` | ML-1.1 | Legal-ish moves | `BenchmarkEvalHeuristic` | smoke |
| ML-1.3 | `MockEvaluator` fixed output | ML-1.1 | Deterministic search tests | `BenchmarkEvalMock` | search integration |
| ML-1.4 | `Position` feature view (stones, to-move) | board | No NN tensors yet | — | — |

**Risk:** Leaking NN types into `board` — forbidden.

---

## Epic ML-2: Inference backends (M11)

| ID | Task | Deps | Acceptance |
|----|------|------|------------|
| ML-2.1 | Model format decision doc | — | ONNX vs sidecar chosen |
| ML-2.2 | `ONNXEvaluator` or gRPC client | ML-2.1 | Load test net |
| ML-2.3 | Batched inference queue | ML-2.2 | Batch size 16 throughput bench |
| ML-2.4 | Feature builder (18 planes + globals) | model | Matches paper Appendix A layout |

**Bench:** `BenchmarkEvalBatch` — positions/sec.

---

## Epic ML-3: Training sample schema (M10–M11)

| ID | Task | Deps | Acceptance |
|----|------|------|------------|
| ML-3.1 | Sample JSON schema | selfplay | policy π, value, ownership label fields |
| ML-3.2 | Export from self-play | SE-4, ML-3.1 | Writes valid samples |
| ML-3.3 | Opponent policy aux field | ML-3.1 | π_opp recorded |
| ML-3.4 | Ownership/score labels | rules score | Labels from final position |

---

## Epic ML-4: Gating harness (M11)

| ID | Task | Deps | Acceptance |
|----|------|------|------------|
| ML-4.1 | Champion/challenger match runner | search | 200 games configurable |
| ML-4.2 | Win threshold 100/200 | ML-4.1 | Promote/reject |
| ML-4.3 | SWA snapshot ingest | external | Document interface only |

---

## Epic ML-5: Training pipeline (external)

| ID | Task | Status |
|----|------|--------|
| ML-5.1 | GPU trainer (PyTorch) | deferred — external repo |
| ML-5.2 | Global pooling in net | deferred — training only |
| ML-5.3 | Progressive net scaling | deferred |
| ML-5.4 | Loss weights (paper Appendix B) | deferred |

---

## Epic ML-6: Post-paper features

| ID | Task | Status |
|----|------|--------|
| ML-6.1 | JSON analysis API | M8+ [POST-PAPER] |
| ML-6.2 | Policy surprise weighting | deferred [POST-PAPER] |

## Risk notes

- **Latency:** external inference — measure p99 before real-time play
- **Version skew:** model schema version in sample header
- **Pure Go training:** out of scope; do not block engine on it
