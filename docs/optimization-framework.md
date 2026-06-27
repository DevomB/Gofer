# Optimization Framework

How Gofer quantifies optimization maturity on a **0–10 composite scale**, cross-linked to **Tectonix `quality_signal` (0–10000)**.

---

## Composite Level Interpretation

| Level | Meaning |
|-------|---------|
| 0 | Prototype / docs only |
| 2 | Rules-correct, performance-naive |
| 4 | Reasonable baseline; microbenches exist |
| 6 | Active profiling; regression guards |
| 8 | Highly optimized serious engine |
| 10 | Elite mature engine; years of measured tuning |

**Tectonix mapping (structural quality):**

| Composite | Typical quality_signal | Notes |
|-----------|------------------------|-------|
| 0–2 | N/A or unstable | Little/no Go code |
| 4 | 6000–7500 | Clean packages, tests, no cycles |
| 6 | 7500–8500 | Benches + profiles + rules pass |
| 8 | 8500–9500 | Hot paths tuned with evidence |
| **Tranche sign-off** | **≥9000** | Required before milestone "done" |

**Anti-gaming (mandatory):** Scan repo root only. No subdirectory scans, file deletion tricks, or disabling plugins to inflate score. Improvements must address `health.root_causes`.

---

## Dimensions (0–10 each)

### 1. Correctness robustness
| Level | Evidence |
|-------|----------|
| 0 | No tests |
| 5 | Unit tests on happy path |
| 10 | Golden + SGF + edge cases; CI green |

**Anti-pattern:** Fixing symptoms per caller. **Quick win:** Golden ko/capture positions. **Advanced:** Property tests on undo/hash.

### 2. Algorithmic sophistication
| 0 | Random play |
| 5 | PUCT MCTS |
| 10 | TT, cap randomization, pruning per paper |

### 3. Data-structure efficiency
| 0 | Maps everywhere |
| 5 | Flat slices, sized move buffers |
| 10 | Arenas, incremental liberties if measured |

### 4. Memory efficiency
| 0 | Clone per move |
| 5 | Undo stack |
| 10 | Measured allocs/op at target on hot benches |

### 5. Allocation discipline
| 0 | Heap alloc in inner loop |
| 5 | `benchmem` tracked |
| 10 | Zero allocs/op on select path or documented ponytails |

### 6. Concurrency effectiveness
| 0 | Decorative goroutines |
| 5 | Root parallel with virtual loss |
| 10 | Measured speedup ≥1.5× at fixed strength |

### 7. Profiling maturity
| 0 | Never profiled |
| 5 | Ad-hoc pprof |
| 10 | Versioned profiles; before/after per optimization |

### 8. Benchmark coverage
| 0 | None |
| 5 | Board + rules benches |
| 10 | Full catalog in blueprint; regression thresholds |

### 9. Build / compiler optimization
| 0 | Default build only |
| 5 | `-ldflags="-s -w"` release |
| 10 | PGO with representative profile; documented refresh |

### 10. Observability / regression prevention
| 0 | No tracking |
| 5 | Manual bench comparison |
| 10 | CI bench smoke + scorecard updates |

### 11. Protocol / tooling maturity
| 0 | No CLI |
| 5 | GTP subset |
| 10 | GTP + analysis + selfplay + bench CLI |

### 12. Idiomatic Go under performance pressure
| 0 | Java-in-Go |
| 5 | Clear packages; interfaces at edges |
| 10 | Hot paths concrete; ponytails documented |

---

## Weighted Scoring Rubric

```
composite = (
  correctness       * 0.15 +
  algorithmic       * 0.10 +
  data_structures   * 0.10 +
  memory            * 0.10 +
  allocations       * 0.10 +
  concurrency       * 0.05 +
  profiling         * 0.10 +
  benchmarks        * 0.10 +
  build             * 0.05 +
  observability     * 0.05 +
  tooling           * 0.05 +
  idiomatic_go      * 0.05
)
```

**Confidence score (0–1):** Fraction of dimensions with `measured` evidence tags.

**Status tags:** `speculative` | `measured` | `regression-tested`

**Evidence links:** bench output path, pprof file, Tectonix report JSON, commit hash.

**Caps (honesty):**
- No benchmarks → composite **≤5**
- No regression guards → **≤6**
- Poor alloc discipline on hot paths → **≤7**
- Unmeasured concurrency claims → concurrency dim **≤4**

---

## Debt Types

| Debt | Definition | Paydown |
|------|------------|---------|
| **Optimization debt** | Known faster approach not implemented | Bench-driven sprint |
| **Performance debt** | Measured regression vs baseline | Profile + fix root cause |
| **Complexity debt** | Abstraction without measured benefit | Delete or simplify |

---

## When NOT To Optimize

- Before correctness tests exist
- Before baseline `benchmem` exists
- When change is <5% and adds complexity
- When feature is not on critical path (YAGNI)
- When Tectonix shows modularity/acyclicity regressing — fix structure first

---

## Tectonix Workflow Integration

1. `session-start` before work
2. `health` → fix weakest root cause before micro-optimizing
3. `test-gaps` before editing riskiest files
4. `session-end` after work — report quality_signal delta
5. **9000+ gate** at tranche sign-off via real fixes, not gaming
