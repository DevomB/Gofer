# Backlog: ML Integration

Epics adapted for monolithic `cmd/gofer` + `training/` (v2.5).

---

## Epic ML-1: Evaluator boundary — done

| ID | Task | Status |
|----|------|--------|
| ML-1.1 | `Evaluator` interface | done — `evaluator.go` |
| ML-1.2 | `Heuristic` | done |
| ML-1.3 | Mock / batched queue | done — `inference.go` |
| ML-1.4 | Feature view | done — `features.go` v1+v2 |

---

## Epic ML-2: Inference backends — done (v2.5)

| ID | Task | Status |
|----|------|--------|
| ML-2.1 | Backend decision | done — ADR 0001 sidecar |
| ML-2.2 | Sidecar client | done — `onnx_sidecar.go` |
| ML-2.3 | Batched queue | done — `BatchedEvaluator` |
| ML-2.4 | Feature builder v2 | done — `BuildFeaturesV2` |

---

## Epic ML-3: Training sample schema — done (v1 + features)

| ID | Task | Status |
|----|------|--------|
| ML-3.1 | Sample JSON schema | done |
| ML-3.2 | Export from self-play | done + `features_spatial/global` |
| ML-3.3 | Opponent policy | done |
| ML-3.4 | Ownership labels | done (engine-side) |

---

## Epic ML-4: Gating harness — done

| ID | Task | Status |
|----|------|--------|
| ML-4.1 | Match runner | done — `RunMatch` |
| ML-4.2 | Win threshold | done — Wilson CI + arena JSON |
| ML-4.3 | SWA ingest | deferred — external |

---

## Epic ML-5: Training pipeline — bootstrap done

| ID | Task | Status |
|----|------|--------|
| ML-5.1 | PyTorch trainer | done — `training/train_bootstrap.py` |
| ML-5.2 | Global pooling | deferred |
| ML-5.3 | Progressive scaling | deferred |
| ML-5.4 | Full paper loss weights | partial — policy + value |

---

## Epic ML-6: Post-paper — deferred

| ID | Task | Status |
|----|------|--------|
| ML-6.1 | JSON analysis API | deferred |
| ML-6.2 | Policy surprise weighting | deferred |
