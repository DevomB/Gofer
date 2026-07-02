# Gofer v2.6.0

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

Sabaki demo (9×9, think-time, SGF on quit):

1. Build: `make build`
2. Sabaki → **Engines → Manage engines → Add**
3. Command: `bin/gofer -gtp -size 9 -think-time 5s -eval heuristic -o demo.sgf`
4. New game 9×9, play; on quit the engine writes `demo.sgf`

Plot arena gating curve: `./scripts/plot-gating.sh .tectonix/reports/arena-9x9-baseline.json`

Nightly 200-game arena workflow: [`.github/workflows/arena-nightly.yml`](.github/workflows/arena-nightly.yml) (also `make reproduce-9x9-baseline`).

### Watch engine vs engine

```bash
bin/gofer -watch -size 9 -playouts 50
```

### ONNX inference (v2.5)

Start the sidecar, then use `-eval onnx`:

```bash
make sidecar    # python training/inference_server.py on :8080
bin/gofer -gtp -size 9 -eval onnx -onnx-url http://127.0.0.1:8080
```

### Arena (gating)

Equal-config ONNX strength check (200 games, same playouts):

```bash
make sidecar
make reproduce-9x9-onnx-gate
```

Legacy asymmetric heuristic gate (600 vs 200 playouts):

```bash
bin/gofer -arena -games 200 -size 9 -playouts 400 \
  -black-playouts 600 -white-playouts 200 \
  -black-eval heuristic -white-eval heuristic -seed 42 \
  -arena-enhanced baseline \
  -json .tectonix/reports/arena-report.json
```

### Self-play

```bash
bin/gofer -selfplay -games 5 -size 9 -playouts 200 -selfplay-eval mix -o samples.jsonl
bin/gofer -selfplay -games 3 -size 9 -sgf-dir games/
```

### ML training loop v3

Persistent replay buffer, resume training, monotonic arena promote:

```bash
SEED_FROM_CYCLE2=1 WEEK_DAYS=14 bash scripts/train-loop-v3.sh
# Lightsail: bash scripts/aws-run-arena.sh IP start-v3
```

See [docs/decisions/0003-iterative-training-loop.md](docs/decisions/0003-iterative-training-loop.md).

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
- **`batched`** / **`mock-batch`** — batched mock inference queue
- **`onnx`** — ONNX Runtime via HTTP sidecar (`-onnx-url`, `-batch-size`, `-eval-timeout`)

## Profile-guided build (optional)

```bash
make pgo-profile    # generates default.pgo (gitignored)
make pgo-build      # bin/gofer with -pgo=
make bench-check    # compare before/after
```

`make pgo-profile` profiles `BenchmarkLegalMoves` only; self-play would be a better macro workload if PGO gains matter.

## Project layout

| Path | Role |
|------|------|
| `cmd/gofer/` | Engine binary (rules, MCTS, GTP, CLI) |
| `cmd/bench/` | Benchmark regression runner |
| `training/` | PyTorch bootstrap trainer + ONNX sidecar |
| `models/` | Exported ONNX weights |
| `docs/` | Blueprint, traceability, scorecard |

## Docs

- [`docs/implementation-blueprint.md`](docs/implementation-blueprint.md) — milestones
- [`docs/research-traceability.md`](docs/research-traceability.md) — paper ↔ code
- [`docs/optimization-scorecard.md`](docs/optimization-scorecard.md) — benches & quality gate

## Included

- Chinese + Tromp-Taylor rules, SGF import/export
- PUCT MCTS, transposition table, parallel playouts
- GTP subset, terminal play/analyze/watch, self-play samples
- Arena gating, Wilson CI, training sample export
- ONNX sidecar inference (`-eval onnx`), bootstrap 9×9 net

## Not yet

- KataGo-level strength or full analysis API
- In-process ONNX (CGO); sidecar is v2.5 default
- Full time controls (byo-yomi); `time_left` uses remaining time as next-move budget
