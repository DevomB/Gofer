# Pipeline orchestrator v4: shards, SPRT gating, resumable stages, champion registry

## Status

Accepted Sep 2026. Supersedes the loop mechanics of [ADR 0003](0003-iterative-training-loop.md). `scripts/train-loop-v3.sh` stays as the legacy path. The learner side is covered by [ADR 0006](0006-learner-v4.md).

## Context

The v3 loop (`train-loop-v3.sh`) proved the champion/challenger design, but its mechanics limited both speed and the ability to rent compute cheaply:

- **Data path.** Self-play emitted JSONL with ~700 JSON floats per row, and every cycle rewrote the whole `replay.jsonl` to trim it. The trainer parsed JSON on every load. Fast-search rows from playout-cap randomization were thrown away.
- **Label bugs in JSONL.** `ownership` was in the absolute frame (Black = +1) while `value` and the input planes are side-to-move. `policy_opp` held the *previous* ply's policy, not the opponent's reply.
- **Gating cost.** Every gate paid for 200 games, even when the candidate was obviously better or worse.
- **Serial stages.** CPU self-play sat idle while the GPU trained, and the reverse.
- **Fragile runs.** A crash or spot preemption restarted the cycle, and the state was spread across env vars, a manifest and fixed paths.
- **Champion history.** `best.onnx` was overwritten in place; one archive copy was the only rollback.

The project has no compute budget today, so the infrastructure has to (a) run meaningfully on free resources, (b) make a future rented GPU cheap to use, including spot instances, and (c) never lose a good network.

## Decision

1. **Gofer shard v1** ([format](../training-data-format.md)): Go writes `.npz` directly (`-o x.npz`). The binary planes are uint8, all labels share the side-to-move frame, `policy_opp` is the true next-ply target, fast rows are kept with a `full_search` flag, and writes are atomic. Result: ~40x smaller than JSONL and no parsing.
2. **Replay = immutable shards plus a growing window** (KataGo's power law over lifetime rows). The orchestrator owns `--window-rows` because only it knows lifetime totals. Aging out is a file move.
3. **SPRT gating** in arena batches (elo0/elo1/alpha/beta), with the v3 Wilson rule as the fallback at `max_games`. Batches are cached, so a resumed gate costs nothing extra.
4. **One Python orchestrator** (`python -m training.pipeline`) driven by a typed TOML config. Every stage is recorded in `state.json` (manifest v4) and resumes at the first unfinished stage. SIGTERM stops cleanly. The trainer resumes with `--continue`.
5. **Overlapped self-play**: cycle N+1's games start while cycle N trains and gates.
6. **Immutable generations plus a publish stage**: `gen-NNNN` files, an Elo ladder, a registry that is never pruned, an optional per-generation GitHub Release, an anti-regression check against an older generation before publishing (to catch false bests), and `rollback`.
7. **Deployment targets from one codebase**: the free GitHub Actions schedule (opt-in), a Docker image (GHCR), a SkyPilot task (spot, bucket-backed run dir), and a plain-VM bootstrap.

The orchestrator depends on the learner only through its CLI contract: `train_bootstrap.py --data --out-dir --window-rows --init-from/--continue [--config preset]` and `export_onnx.py --checkpoint --out`. It is tested with a fake executor (no Go or PyTorch), plus a real end-to-end smoke run in CI.

## Consequences

- Throughput: less data I/O, gates end early when the result is clear, and self-play overlaps training. How much wall-clock each saves depends on the hardware, so measure with `python -m training.pipeline estimate` instead of assuming.
- The value/ownership heads see about 5x more rows per game, with no extra self-play.
- With `overlap_selfplay`, the trainer may already see the next cycle's shard if it finishes first. That is intended (newer data from the same champion), but it makes cycles less reproducible. Turn overlap off for controlled experiments.
- SPRT with `elo1 = 35` needs ~31 net wins to accept, so small `max_games` caps often end at the fallback rule. Pick `elo1` to match the expected per-generation gain (`plan-sprt`).
- Free Actions training is CPU-only and depends on the Actions cache (7-day eviction when idle). Releases are the durable record.
- The legacy JSONL fields keep their old meaning for compatibility. The learner corrects them on load.
