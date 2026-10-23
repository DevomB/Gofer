# Architecture

```
            ┌──────────────── Go engine (bin/gofer) ────────────────┐
            │                                                        │
 SGF ──► internal/sgf      internal/game  ◄── rules, scoring          │
            │                   ▲                                    │
            │                   │ boards                             │
            │   internal/nn ◄───┴──── internal/mcts ◄── gtp, play,   │
            │   features, batcher,      PUCT search      arena,      │
            │   ONNX Runtime backend                     selfplay    │
            └────────────────────────────────────┬───────────────────┘
                     ▲ gen-N.onnx                 │ .npz shards, arena JSON
                     │                            ▼
            ┌──────── training/ (PyTorch) ────────────────────────────┐
            │ loop.py: selfplay → train → export → evaluate → repeat  │
            │ data.py  model.py  train.py  export.py  elo.py          │
            └─────────────────────────────────────────────────────────┘
```

The Go binary does everything that needs to be fast or that plays games;
Python only trains networks. They communicate through files: ONNX models one
way, NumPy shards and JSON match results the other.

## Packages

| Package | Responsibility |
|---------|----------------|
| `internal/game` | Board with incremental chains and exact liberties (bitsets), legality (suicide, simple ko, positional superko via per-chain Zobrist hashes), Tromp-Taylor area scoring and ownership. |
| `internal/sgf` | SGF parsing (main line, setup stones) and writing. |
| `internal/nn` | Feature encoding (schema 3), the `Evaluator` interface, `Net` (batches concurrent requests across worker backends, random symmetry, position cache), ONNX Runtime backend (CPU, CUDA, TensorRT), network-free `Uniform` and `Rollout` evaluators. |
| `internal/mcts` | PUCT search with virtual-loss leaf batching, Dirichlet root noise, tree reuse, exact terminal scoring and the conservative pass rule. |
| `internal/selfplay` | Concurrent self-play games with playout-cap randomization, resignation with false-positive tracking, `.npz` shard writer. |
| `internal/arena` | Matches between any two players (networks, rollout/uniform baselines, external GTP engines such as GNU Go) with paired openings, dead-stone adjudication, Wilson intervals, Elo and SPRT. |
| `internal/gtp` | GTP v2 server. |
| `cmd/gofer` | CLI: `gtp`, `play`, `selfplay`, `arena`, `bench`. |

## Search

Each node stores `W`, the sum of backed-up values **from the perspective of
the player who made the move into that node**, so a parent picks the child
maximizing `Q + U` directly:

```
Q = W / N                     (unvisited: parent value − FPU reduction · sqrt(visited prior mass))
U = c_puct · P · sqrt(N_parent) / (1 + N)
```

A network value `v` is for the side to move at the leaf, so the leaf receives
`−v`, its parent `+v`, alternating to the root. Children are allocated once
per expansion and never appended to, so node pointers stay valid.

A search gathers up to `Batch` leaves per round, marking each path with a
virtual loss so the next descent diverges, and evaluates them with a single
`Evaluator.Eval` call. Many games run concurrently against one `nn.Net`, whose
workers pull requests from a shared queue, so the network sees large batches
(GPU) or all cores stay busy (CPU).

Terminal positions (two passes, or the move cap) are scored exactly with area
scoring. Under the conservative pass rule a player may only pass when the
opponent just passed, which makes passing an exactly scored terminal move, or
when every other move would fill one of its own eyes. Games therefore end
with dead stones captured, and area scoring is exact.

## Network inputs (schema 3)

13 planes of size × size, from the side to move's point of view:

| Plane | Meaning |
|-------|---------|
| 0, 1 | own stones, opponent stones |
| 2 | all ones (lets zero-padded convolutions see the edge) |
| 3, 4 | own chains with 1 / 2 liberties |
| 5, 6 | opponent chains with 1 / 2 liberties |
| 7 | simple-ko point |
| 8–12 | the last five moves |

4 globals: komi from the side to move's perspective / 15, opponent just
passed, we passed before that, move number / board area.

Outputs: policy logits over points + pass, value in [−1, 1] for the side to
move, and (training only) ownership per point.

## Model

`training/model.py`: residual trunk (default 6 blocks × 64 channels) where
every third block adds a global-pooling bias, a fully convolutional policy
head with a pooled pass logit, and value/ownership heads on pooled features
(KataGo-style). Unlike v2's flat fully connected heads it has no
size-specific dense layers, so it has fewer parameters and generalizes
across the board.

## Training data

`gofer selfplay` writes `.npz` shards with one row per full-search move:
`spatial` (uint8 planes), `global`, `policy` (visit distribution), `value`
(game result for the side to move), `q` (root search value), `ownership` and
`ownership_weight` (0 for resigned games), `margin`.
