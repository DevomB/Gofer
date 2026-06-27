# Backlog: Core Engine

Epics, tasks, dependencies, acceptance criteria, risks, hooks.

---

## Epic E1: Board foundation (M0–M1)

| ID | Task | Deps | Acceptance | Bench | Test | Risk |
|----|------|------|------------|-------|------|------|
| CE-01 | Coordinate system `Point`, `Move`, pass | — | 19×19 index bijection | — | unit | off-by-one |
| CE-02 | `Stone` enum, empty board | CE-01 | `NewBoard(19)` | BenchmarkPlay setup | unit | — |
| CE-03 | Move encoding | CE-01 | Pass + point moves | — | unit | — |
| CE-04 | Zobrist hashing | CE-02 | Hash stable on undo | BenchmarkHash | hash roundtrip | collision (acceptable) |
| CE-05 | Undo stack | CE-02 | 100 plies roundtrip | BenchmarkUndo | property | stack bugs |
| CE-06 | `Ruleset` interface | — | Chinese implements | — | compile | over-abstraction |

## Epic E2: Chinese rules (M1)

| ID | Task | Deps | Acceptance | Bench | Test | Risk |
|----|------|------|------------|-------|------|------|
| CE-10 | Liberty counting / capture | CE-02 | Capture 1-lib group | BenchmarkLegalMovesCapture | capture golden | perf |
| CE-11 | Suicide if no capture (Chinese) | CE-10 | Illegal suicide table | — | table | rule edge |
| CE-12 | Simple ko ban | CE-04 | Ko ban after recapture | — | ko golden | superko gap |
| CE-13 | Legal move generation | CE-10–12 | Empty board 361 moves | BenchmarkLegalMoves | golden count | alloc |
| CE-14 | Chinese area scoring | CE-10 | Known score positions | BenchmarkScore | score test | territory algo |
| CE-15 | Tromp-Taylor hook points | CE-06 | Interface extensible | — | compile | — |

## Epic E3: SGF & testdata (M1–M2)

| ID | Task | Deps | Acceptance | Bench | Test | Risk |
|----|------|------|------------|-------|------|------|
| CE-20 | SGF parse minimal | CE-02 | Load test SGF | BenchmarkSGFReplay | replay | parser scope |
| CE-21 | Golden position corpus | CE-13 | 3+ positions in testdata | — | golden | — |

## Epic E4: Tromp-Taylor (M2)

| ID | Task | Deps | Acceptance | Bench | Test | Risk |
|----|------|------|------------|-------|------|------|
| CE-30 | Tromp rules package | CE-15 | Differs from Chinese documented | — | divergence tests | Benson delay |

---

## Benchmark hooks

- `internal/board/board_bench_test.go`
- `internal/rules/chinese_bench_test.go` (`BenchmarkLegalMoves`, `BenchmarkPlay`, `BenchmarkCaptureHeavy`)

## Test hooks

- `internal/rules/rules_test.go` golden SGF + unit tests
- `internal/testdata/golden/*.json` or embedded SGF
