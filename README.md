# Gofer v1.0.0

A Go engine with Chinese rules, MCTS search, GTP, and self-play — inspired by [Wu et al. 2020](https://arxiv.org/abs/1902.10565). Not KataGo.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Quick start

```bash
make build          # -> bin/gofer
make test
make bench-check    # same regression gate as CI
```

### Play in the terminal

```bash
bin/gofer -play -size 9 -color b
# commands: play D4 | genmove | analyze | board | score | undo | quit
# save game on quit: bin/gofer -play -o game.sgf
```

### Analyze a position

```bash
bin/gofer -analyze -size 9 -playouts 400
bin/gofer -analyze -size 9 -think-time 3s -top 8
bin/gofer -analyze -size 9 -moves D4,Q16,pass
```

### GTP (Sabaki, Lizzie, KaGo tools)

Engine command:

```text
bin/gofer -gtp -size 9 -think-time 5s -eval heuristic
```

Save game on quit:

```text
bin/gofer -gtp -size 9 -o game.sgf
```

Or fixed playouts:

```text
bin/gofer -gtp -gtp-playouts 800 -eval heuristic
```

Sabaki: **Engine → Manage engines → Add** → path above.

### Watch engine vs engine

```bash
bin/gofer -watch -size 9 -playouts 50
```

### Self-play

```bash
bin/gofer -selfplay -games 5 -size 9 -playouts 100 -o samples.json
bin/gofer -selfplay -games 3 -size 9 -sgf-dir games/
```

### SGF replay

```bash
bin/gofer -sgf cmd/gofer/testdata/simple.sgf
```

## Defaults

| Board | Default playouts (when `-playouts 0`) |
|-------|---------------------------------------|
| 9×9   | 400                                   |
| 13×13 | 800                                   |
| 19×19 | 1600                                  |

`-think-time` overrides playout count for that move (GTP `time_left`, `-play`, `-analyze`, `-gtp`).

## Evaluators

- **`heuristic`** (default) — stones, liberties, territory estimate, move priors for PUCT
- **`uniform`** — random-ish MCTS baseline

## Profile-guided build (optional)

```bash
make pgo-profile    # generates default.pgo (gitignored)
make pgo-build      # bin/gofer with -pgo=
make bench-check    # compare before/after
```

ponytail: `pgo-profile` uses `BenchmarkLegalMoves` microbench. Ceiling: may mis-optimize search paths. Upgrade: profile from `-selfplay` macro workload.

## Project layout

| Path | Role |
|------|------|
| `cmd/gofer/` | Engine binary (rules, MCTS, GTP, CLI) |
| `cmd/bench/` | Benchmark regression runner |
| `docs/` | Blueprint, traceability, scorecard |

## Docs

- [`docs/implementation-blueprint.md`](docs/implementation-blueprint.md) — milestones
- [`docs/research-traceability.md`](docs/research-traceability.md) — paper ↔ code
- [`docs/optimization-scorecard.md`](docs/optimization-scorecard.md) — benches & quality gate

## v1 scope (done)

- Chinese + Tromp-Taylor rules, SGF import/export
- PUCT MCTS, transposition table, parallel playouts
- GTP subset, terminal play/analyze/watch, self-play samples

## Not in v1 (explicit non-goals)

- Neural network training or ONNX inference in-process
- KataGo-level strength or full analysis API
- Full time controls (byo-yomi); `time_left` uses remaining time as next-move budget
