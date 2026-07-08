# Net-size ablation (Phase 2, Piece 2)

## Status

Investigated Jul 2026 — **not adopted**. Live architecture remains **6×64** (`GoferBootstrapNet` default).

## Context

With replay capped at 50k rows, a 6×64 trunk (1.2M params) may exceed what the data volume needs. Phase 2 Piece 2 compared smaller nets offline on a frozen replay snapshot before any live architecture change.

Piece 1 (playout-cap randomization) already reduced self-play wall-clock without touching model size. Switching architecture would forfeit `--resume` warm-start from the current champion checkpoint.

## Experiment

- **Dataset:** frozen copy of Lightsail `replay.jsonl` (50,000 rows, pre–Piece-1-cap-randomization cycles). See `training/data/ablation/README.md`.
- **Training:** fresh init per candidate, 25 epochs, lr=0.01, val_split=0.1, split_seed=42 (fixed). No early stop. Script: `training/net_size_ablation.py`.
- **Candidates:** 6×64 (baseline), 4×48, 6×32, 4×32.
- **Seed variance (Jul 2026):** 4×48 and 6×32 re-trained with init seeds 11, 42, 99 (`--variance` mode).

## Findings

### Single-seed ablation (val loss, ONNX size, CPU latency)

| Config | Best val loss | ONNX | Latency (median) |
|--------|---------------|------|------------------|
| 6×64 baseline | 3.2285 | 4.64 MB | 0.90 ms |
| **4×48** | **3.1989** | 2.61 MB | 0.41 ms |
| 6×32 | 3.2247 | 1.58 MB | 0.32 ms |
| 4×32 | 3.2302 | 1.44 MB | 0.31 ms |

4×48 showed the best val loss with ~2× faster inference and smaller ONNX. Gaps were small (~0.9% vs 6×64).

### Seed variance (init seeds 11, 42, 99)

| Config | Mean val loss | Spread (max−min) |
|--------|---------------|------------------|
| 4×48 | 3.2032 | **0.0063** |
| 6×32 (seeds 42, 99) | 3.2210 | 0.0083 |
| 6×32 (all seeds) | 3.8475 | 1.8838 |

**6×32 seed 11 diverged** to val loss **5.10** after 25 epochs and never recovered — init fragility; disqualified as a live candidate without further stabilization work.

4×48 vs 6×64/6×32 val-loss edge is **real but thin** (~0.02 absolute, ~3–4× the 4×48 run-to-run spread). Not reliable enough to justify a live switch on val loss alone.

### Re-bootstrap cost (from `train-v3.log`, parallel=2)

Architecture change breaks `--resume` (shape mismatch). Estimated cost to recover **current champion playing strength** (cycle-24 level):

- **~4–5 calendar days**, **~20–22 cycles** from cold start (SEED → first decisive PROMOTE ~1–1.5 days / ~6 cycles; then ~3 more days to cycle-24 strength).
- Median **~3.7 h/cycle** on Lightsail t3.small.

## Decision

**Keep 6×64** as the live architecture.

Inference latency and file-size wins from 4×48 do not justify ~4–5 days of lost champion strength when Piece 1 already addressed training wall-clock without an architecture change.

**Reconsider only if** a from-scratch retrain becomes necessary for other reasons (e.g. new board size, feature schema change, or deliberate reset).

## Artifacts

- Full ablation: `.tectonix/reports/net-size-ablation/results.json` (Lightsail + local copy).
- Seed variance: `.tectonix/reports/net-size-ablation/seed-variance.json`.
- Re-run instructions: module docstring in `training/net_size_ablation.py`.
