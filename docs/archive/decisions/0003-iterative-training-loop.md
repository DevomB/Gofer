# ADR 0003: Iterative training loop (ML pipeline v3)

## Status

Accepted — 2026-07-01

## Context

`scripts/weekly-train-loop.sh` retrained from random init each cycle on a single
`samples-cycleN.jsonl`, always deployed the latest ONNX, and never loaded
`gofer-9x9-best.onnx` into training or self-play. Win rates bounced (33% → 58% → 35%)
instead of climbing monotonically.

## Decision

Replace the weekly loop with `scripts/train-loop-v3.sh`:

1. **Persistent state** under `training/state/` (`best.pt`, `manifest.json`) and
   `training/data/replay.jsonl` (FIFO cap 50k rows).
2. **Resume training** from `best.pt` every cycle (`--resume`); save `best.pt` on
   minimum validation loss only.
3. **Monotonic promote**: export `gofer-9x9-candidate.onnx`, arena vs heuristic;
   promote to `gofer-9x9-best.onnx` only if win rate beats historical best + 2pp
   or reaches `WIN_TARGET` (0.75). On reject, restore sidecar to `best.onnx` and
   revert `best.pt` from pre-cycle backup.
4. **Self-play mix** (70% ONNX / 30% heuristic via odd/even games) after PR3.
5. **Gating constants** in `scripts/gating.env` (single source for shell scripts).

## Consequences

- Cycle-2 seed (`SEED_FROM_CYCLE2=1`) bootstraps manifest at 58% floor.
- Arena uses 400 playouts; self-play uses 200 (reduced distribution shift).
- Wilson CI logged for diagnostics; promotion uses point estimate + margin.
- `weekly-train-loop.sh` is a **legacy alias** that execs `train-loop-v3.sh` (do not extend).

## Alternatives considered

- **Per-cycle scratch training** — rejected (G1 regression).
- **Always deploy latest ONNX** — rejected (G8).
- **90% win gate** — out of scope (needs 10k+ games, larger net).
