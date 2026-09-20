# Training data format: Gofer shard v1

Self-play writes **shards**: standard NumPy `.npz` archives (deflated ZIP of `.npy` arrays), one per self-play run, one board size per shard. Any `np.load` reads them; no custom parser is needed.

```bash
bin/gofer -selfplay -games 200 -size 9 -o runs/x/selfplay/cycle-0001.npz   # .npz => shard
bin/gofer -selfplay -games 200 -size 9 -o samples.jsonl                   # legacy JSONL still works
```

Writers: Go `WriteSampleShard` ([cmd/gofer/shard.go](../cmd/gofer/shard.go)) and the learner's `gofer_train.shards pack`, which converts legacy JSONL. Readers: the learner (`--data <dir|file.npz>`) and the orchestrator's replay index ([training/pipeline/replay_index.py](../training/pipeline/replay_index.py)). **Both writers must produce this exact layout**, and a change needs a version bump.

## Arrays

`N` rows, board size `S` (9), `P = S*S + 1` policy entries (the last one is pass).

| Array | dtype | shape | Meaning |
|---|---|---|---|
| `spatial` | uint8 | `[N, 8, S, S]` | 0/1 input planes, same order as `BuildFeaturesV2`: own stones, opponent stones, empty, ko, black-to-move, last 3 moves |
| `globals` | float32 | `[N, 4]` | komi/10, move/(S²+1), is-black, is-white |
| `policy` | float32 | `[N, P]` | MCTS root visit distribution (sums to 1) |
| `policy_opp` | float32 | `[N, P]` | opponent's reply policy (the **next** ply's target); all-zero when that ply was a fast search or the game ended |
| `value` | float32 | `[N]` | game result from side-to-move: +1 win, -1 loss, 0 draw |
| `score` | float32 | `[N]` | final area-score margin from side-to-move, komi included |
| `ownership` | int8 | `[N, S*S]` | final owner from side-to-move: +1 own, -1 opponent, 0 neutral |
| `full_search` | uint8 | `[N]` | 1 = full-cap search (policy target valid), 0 = fast search |
| `game_id` | int32 | `[N]` | game index within the shard |
| `move_num` | int16 | `[N]` | ply number |
| `meta` | uint8 | `[K]` | UTF-8 JSON (below) |

`meta` example:

```json
{"format": "gofer-shard", "version": 1, "board_size": 9, "rows": 327, "games": 4,
 "git_commit": "...", "model": "sha256:4f1c... | heuristic", "komi": 6.5, "seed": 1,
 "created_at": "2026-09-18T20:03:16Z"}
```

`model` identifies the network that generated the games (first 8 bytes of the ONNX sha256), so the replay buffer can tell champions apart.

## Invariants

These are tested in [cmd/gofer/shard_test.go](../cmd/gofer/shard_test.go) and [training/pipeline/test_pipeline.py](../training/pipeline/test_pipeline.py):

- **One perspective.** `value`, `score`, `ownership` and `policy_opp` all use the side-to-move frame, like the own/opp input planes. Exactly: `score == ownership.sum(1) - komi` for black-to-move rows and `+ komi` for white-to-move rows.
- **Fast rows are kept.** Unlike `-full-only` JSONL, shards keep playout-cap-randomization fast rows, flagged `full_search=0`. The learner trains value/ownership/score on every row and masks the policy losses to `full_search == 1`. That gives about 5x more value data at no self-play cost. Pass `-full-only=true` explicitly to drop them.
- **Atomic.** Shards are written to `<path>.tmp` and renamed (retrying on Windows sharing violations), so a crash never leaves a truncated `.npz` for the trainer.
- **Validated at write time.** A row is rejected, rather than silently written, if a plane is not 0/1, if any float is NaN or infinite, or if `ownership` is missing: the learner applies its ownership loss to every row, so an all-zero map would teach "neutral everywhere". Self-play always supplies ownership; SGF conversion writes JSONL, not shards.
- **Immutable.** The replay buffer never rewrites shards. The window is "newest shards that fit", and aging out moves a file to `selfplay/archive/`.

## Compared with JSONL

The same 4 games (327 rows) take **19 KB** as a shard and **780 KB** as JSONL, about 40x smaller. Loading is a single `np.load` per column, with no JSON float parsing.

Legacy JSONL gained three additive fields: `game_id`, `score_margin` (side-to-move) and `policy_next` (the correct next-ply target). Two legacy fields keep their old meaning for compatibility: `ownership` is **absolute** (Black = +1), and `policy_opp` is the **previous** ply's policy. The learner corrects both when it loads JSONL.
