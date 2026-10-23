# Backlog: Optimization

Epics for benchmarks, profiles, PGO, Tectonix-driven structural fixes, regression guards.

---

## Epic OP-1: Microbenchmarks (M1+)

| ID | Task | Package | Measures |
|----|------|---------|----------|
| OP-1.1 | Board clone vs undo | board | alloc strategy |
| OP-1.2 | MakeMove / Undo | rules/chinese | hot path |
| OP-1.3 | LegalMoves | rules/chinese | move gen |
| OP-1.4 | Capture-heavy fixture | rules/chinese | worst case |
| OP-1.5 | Score | rules/chinese | endgame |
| OP-1.6 | HashUpdate | board | TT prep |
| OP-1.7 | Node expand/select | search | M5 |
| OP-1.8 | Playout iteration | search | M5 |
| OP-1.9 | Eval call / batch | eval | M7 |
| OP-1.10 | GTP parse | gtp | M8 |
| OP-1.11 | SGF replay | sgf | M2 |

**Caveat:** Bench noise on laptops — `-count=5`, compare on same machine.

---

## Epic OP-2: Profiling workflow (M1+)

| ID | Task | Acceptance |
|----|------|------------|
| OP-2.1 | Makefile `profile` target | Generates cpu.prof from representative bench |
| OP-2.2 | Document commands in go-engineering-standard | Done |
| OP-2.3 | Store baseline profile in `.tectonix/reports/` | Named by date |

---

## Epic OP-3: Escape analysis (M3+)

| ID | Task | Acceptance |
|----|------|------------|
| OP-3.1 | `go build -gcflags=-m` on board/rules | Log escapes in decision doc |
| OP-3.2 | Fix unintended escapes in MakeMove | alloc/op reduced in bench |

---

## Epic OP-4: Data structure experiments (M3)

| ID | Task | Acceptance |
|----|------|------------|
| OP-4.1 | Pointer-rich vs index nodes | Bench comparison + decision |
| OP-4.2 | Preallocation strategies | Document winner |
| OP-4.3 | Hash map vs open addressing TT | M6 |

---

## Epic OP-5: PGO (M12)

| ID | Task | Acceptance |
|----|------|------------|
| OP-5.1 | Representative `default.pgo` from game bench | Profile captured |
| OP-5.2 | `make pgo-build` | Builds with `-pgo=` |
| OP-5.3 | Measure gain | 2–14% possible; report actual % or none |

---

## Epic OP-6: Regression thresholds (M9)

| ID | Task | Acceptance |
|----|------|------------|
| OP-6.1 | `cmd/bench` JSON output | ns/op, allocs |
| OP-6.2 | Compare vs committed baseline | CI or manual gate |
| OP-6.3 | Fail if >10% regression without approval | Documented |

---

## Epic OP-7: Tectonix structural optimization

| ID | Task | Acceptance |
|----|------|------------|
| OP-7.1 | `tectonix session-start/end` per milestone | Reports saved |
| OP-7.2 | Fix weakest root cause each session | health delta |
| OP-7.3 | `quality_signal >= 9000` at release | No gaming |
| OP-7.4 | `tectonix check-rules` clean | Layer violations = 0 |
| OP-7.5 | Address `riskiest_untested` before hot edits | test-gaps |

### Tectonix iteration loop

```
health → weakest metric → structural fix → test-gaps → rescan → session-end
```

| Root cause low | Typical fix |
|----------------|-------------|
| Modularity | Split packages, reduce cross-imports |
| Acyclicity | Break import cycles |
| Depth | Remove pass-through wrappers |
| Equality | Split god files |
| Redundancy | Delete duplicate bodies |

---

## Epic OP-8: When NOT to optimize

Track in scorecard — review monthly. See `optimization-framework.md` Section 6.

## Makefile targets (target state)

```makefile
test bench race lint profile pgo-build
```
