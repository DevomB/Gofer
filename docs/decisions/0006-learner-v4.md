# ADR 0006: Learner v4 (gofer_train)

## Status

Accepted, Sep 2026. Replaces the internals of `train_bootstrap.py` and `export_onnx.py`; their CLIs are unchanged. Orchestration is covered separately in ADR 0007.

## Context

The v3 learner had these problems:

- It rebuilt a Python tensor from JSON lists for every row, every epoch.
- It split train/val by row, which leaked near-duplicate positions from the same game into val.
- It trained the policy on fast-cap moves. After playout-cap randomization (ADR 0005 Piece 1), most rows are fast-cap, and their visit distributions are noise.
- It stored JSONL ownership in absolute colour while every input feature is side-to-move.
- It had no augmentation, no LR schedule and no weight averaging.
- It could not resume an interrupted run.
- It could not change architecture without the 4–5 day re-bootstrap cost that ADR 0005 used to keep 6x64.

## Decision

Build the `training/gofer_train/` package:

1. **Shard v1 data.** Rows come from `.npz` shards (written directly by Go; the format is owned by the orchestrator side) or from converted legacy JSONL. The replay window is held on the training device and gathered by index.
2. **Correct targets.**
   - Policy loss is masked to `full_search` rows. Value, ownership and score train on all rows.
   - Ownership is side-to-move.
   - Val split is by game.
   - Every batch gets a random D4 symmetry per sample.
3. **Trainer.**
   - Warmup + cosine schedule, Nesterov SGD or AdamW, decoupled weight decay.
   - EMA weights, which are exported.
   - AMP, gradient clipping, and `torch.compile` on CUDA.
   - Atomic checkpoints and `--continue` for preemptible GPUs.
   - `summary.json` for the orchestrator.
4. **Architectures.**
   - `legacy` stays bit-compatible with the champion.
   - New `gpool` family: global-pooling bias, convolutional/pooled heads that don't depend on board size, plus score and opponent-policy auxiliary heads.
   - Architecture metadata is stored in `best.ckpt` and in the ONNX `metadata_props`.
5. **Distillation.** `--teacher` lets a new architecture start from the champion's knowledge instead of from zero.
6. **Export.** Always parity-checked against ONNX Runtime; optional int8.

## Consequences

- Existing callers (the v3 shell loop, `make train-bootstrap`, `net_size_ablation.py`) keep working. `best.pt` is still a plain state_dict.
- The default architecture is still `legacy-6x64` when no `--arch` is given. Moving to `gpool` is an explicit, gated decision (distill → arena vs champion), not a silent switch.
- CPU training speed per step is roughly unchanged, because it is compute-bound. The wins are data volume (shards are ~40× smaller than JSONL), GPU throughput, and sample efficiency.
- Validation loss is not comparable with v3 numbers: the split is different and fast-cap rows are masked out of the policy loss.

## Open questions

- **Is gpool-6x64 stronger?** It has 0.45M params versus 1.2M for legacy 6x64. Whether it is stronger at equal self-play cost needs an arena run on real hardware: distill it, then gate it.
- **Does int8 hold strength?** The int8 ONNX needs its own gate before use in self-play.
