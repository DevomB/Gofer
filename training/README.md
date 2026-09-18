# Gofer learner (`training/`)

This directory turns self-play games into an ONNX network the Go engine can load.
The Python learner lives in [`gofer_train/`](gofer_train/). The legacy entry points
(`train_bootstrap.py`, `export_onnx.py`, `model.py`) are thin wrappers around it,
so every existing script and `make` target keeps working.

```text
gofer -selfplay ──► shards (.npz) ──► train_bootstrap.py ──► best.pt ──► export_onnx.py ──► cand.onnx ──► arena/gate
      (Go)          or legacy JSONL      (gofer_train)                    (parity-checked)
```

For pipeline orchestration (cycles, gating, promotion, cloud runs), see
[docs/pipeline.md](../docs/pipeline.md). This README covers only the learner.

## Quick start

```bash
pip install -r training/requirements.txt

# Train on a shard directory (newest 150k rows, recency-weighted), warm-starting from the champion
python training/train_bootstrap.py --config training/configs/train-cpu.toml \
    --data training/data/shards --out-dir training/state/run --init-from training/state/best.pt

# Export + ORT parity check (non-zero exit if torch and ONNX disagree)
python training/export_onnx.py --checkpoint training/state/run/best.pt --out models/gofer-9x9-candidate.onnx

# Tests (CPU, ~40 s)
python -m pytest training -q
```

## Data

| Source | How it's read | Notes |
|---|---|---|
| GOFER SHARD v1 (`.npz`) | `gofer_train.shards.read_shard` | Written directly by `gofer -selfplay -o x.npz`. Compressed files are ~40× smaller than JSONL and load ~4× faster. Contract: [docs/training-data-format.md](../docs/training-data-format.md) |
| Legacy JSONL | `gofer_train.shards.rows_from_jsonl` | Converted to the shard conventions: ownership flipped to side-to-move, game ids derived from `move_num` resets, and the old `policy_opp` field ignored because it holds the *previous* move's policy |

`--data` accepts files, shard directories, or several of each. The loader:

- **Window.** `--window-rows N` keeps the newest N rows. Shards are ordered by mtime, and whole old shards are skipped rather than read.
- **Recency weighting.** With `--window-decay F`, the newest rows are drawn eᶠ times as often as the oldest.
- **Game-level validation split.** Whole games go to either train or val. The old row-level split leaked near-duplicate positions from the same game into val.
- **Device-resident.** The window is loaded onto the training device once (uint8 planes: 50k rows ≈ 32 MB). Batches are gathered by index, so the loop has no DataLoader and no per-row tensor building.
- **8-fold D4 symmetry.** Each sample gets a random dihedral transform on the fly. Planes, policy (pass untouched) and ownership are transformed together, and a test checks they stay aligned.

Pack JSONL or inspect shards:

```bash
cd training
python -m gofer_train.shards pack --in data/replay.jsonl --out data/shards/ --rows-per-shard 25000
python -m gofer_train.shards inspect data/shards/
```

## Losses

| Head | Target | Rows | Weight |
|---|---|---|---|
| policy | MCTS visit distribution | `full_search == 1` only (playout-cap randomization: fast-search visits are too noisy to imitate) | 1.0 |
| value | game result, side-to-move | all rows | 1.5 |
| ownership | final owner per point | all rows | 0.15 |
| score *(gpool)* | final margin / 20, Huber | rows with a known margin | 0.05 |
| policy_opp *(gpool)* | opponent's reply policy | full-search rows where the reply is known | 0.15 |
| distill | teacher policy (KL) + value (MSE) | all rows, only with `--teacher` | 1.0 |

Masking lets self-play keep its fast-cap moves. They carry about 5× more value and
ownership data at no extra cost, and they no longer dilute the policy target.
Weights are set under `[train.loss]` in a config.

## Architectures

| Preset | Params | Notes |
|---|---|---|
| `legacy-6x64` | 1.22 M | The current champion's layout (flatten → FC heads). State dicts are unchanged, so `--init-from training/state/best.pt` works. |
| `legacy-4x48` | 0.68 M | ADR 0005's candidate |
| `gpool-6x64` | 0.45 M | KataGo-style. Global-pooling bias blocks, fully-convolutional pooled heads, and score + opponent-policy auxiliary heads. The parameter count doesn't depend on board size. Residual branches are zero-initialized to prevent the seed-divergence seen in ADR 0005. |
| `gpool-10x96` | 1.6 M | Default for GPU runs |
| `gpool-15x128` | 4.3 M | Larger GPU runs |

`--arch kind-BxC` (for example `gpool-8x80`) builds any size. The chosen architecture
is saved in `best.ckpt`, and `export_onnx.py` reads it back. Older bare state dicts
have their architecture inferred from tensor shapes.

**Switching architecture without losing strength.** ADR 0005 kept 6x64 only because a
new architecture meant 4–5 days of re-bootstrapping. Distillation removes that cost:

```bash
python training/train_bootstrap.py --config training/configs/distill-gpool.toml \
    --teacher training/state/best.pt --data training/data/shards --out-dir training/state/distill
```

The student learns from the replay targets and from the champion's own policy and value.
It then goes through the normal arena gate against the champion.

## Trainer

- Step-based schedule: linear warmup, then cosine decay to `min_lr_frac`. `--steps` overrides `--epochs`.
- SGD with Nesterov momentum (default) or AdamW. Weight decay applies to weights only, not to BN or biases.
- **EMA weights** (`ema_decay`, default 0.999). Evaluation and `best.pt` both use the EMA weights, which are smoother and usually a little stronger at no cost.
- AMP (`bf16` where supported, otherwise `fp16` with a grad scaler), gradient clipping, `--channels-last`, and `--compile` (CUDA only).
- **Crash-safe.** Checkpoints are written atomically. `--ckpt-every N` saves periodically. `--continue` resumes an interrupted run with its optimizer, scheduler, EMA and step counter, so a preempted spot GPU loses at most N steps.
- `--resume` and `--init-from` keep their v3 meaning: warm-start weights, fresh optimizer. A missing auxiliary head (older checkpoints have no ownership head) starts fresh instead of failing.

Outputs in `--out-dir`:

| File | Contents |
|---|---|
| `best.pt` | Plain state_dict (EMA). Back-compatible with every consumer. |
| `best.ckpt` | `{arch, state_dict, step, val}`, self-describing |
| `last.pt` | Full resume state, plus the legacy `epoch` / `train_loss` / `val_loss` keys |
| `metrics.jsonl` | One line per eval: train/val losses, policy accuracy, value sign accuracy, lr, samples/s |
| `summary.json` | Stable keys for the orchestrator: `status, best_val_loss, best_step, best_epoch, final_val_loss, val{…}, steps, epochs, samples_seen, samples_per_sec, train_rows, val_rows, val_games, arch, params, device, best_pt, wall_sec` |

Presets in [`configs/`](configs/):

| Preset | Target |
|---|---|
| `train-cpu.toml` | Lightsail or a laptop. Keeps the legacy-6x64 lineage. |
| `train-gpu.toml` | One rented GPU: gpool-10x96, 1M-row window, bf16, compile, save every 500 steps |
| `distill-gpool.toml` | One-off architecture migration |

Flags given on the command line override the preset's values.

## Export

```bash
python training/export_onnx.py --checkpoint run/best.pt --out cand.onnx [--quantize-int8] [--with-ownership] [--json]
```

- Input and output names are unchanged (`spatial_input`, `global_input` → `policy_logits`, `value`). The auxiliary heads are stripped.
- Every export is checked against PyTorch with ONNX Runtime; a difference above 1e-3 exits with an error.
- The ONNX file carries `gofer.arch`, `gofer.weights_sha` and `gofer.step` as metadata.
- `--quantize-int8` also writes `cand.int8.onnx` (dynamic int8 weights) for CPU inference. Gate it in the arena before using it.

## Benchmark

```bash
PYTHONPATH=training python -m gofer_train.bench --data training/data/samples.jsonl --steps 60
```

This prints load time and training samples/s for the legacy DataLoader loop and for the
new trainer. On CPU both are compute-bound and run at about the same speed. The new
trainer's gains are elsewhere:

- data size and load time
- GPU throughput, since the loop has no DataLoader or host-to-device copies and uses AMP
- sample efficiency: 8× augmentation, masked policy on 5× more value data, and EMA

## Files

| Path | Role |
|---|---|
| `gofer_train/nets.py` | Architectures, presets, architecture inference, lenient auxiliary-head loading |
| `gofer_train/shards.py` | Shard v1 read/write, JSONL conversion, window loading, `pack`/`inspect` CLI |
| `gofer_train/data.py` | Device-resident rows, game split, recency sampler, D4 symmetry |
| `gofer_train/losses.py` | Masked multi-head loss and metrics |
| `gofer_train/trainer.py` | Trainer, config, checkpoints, EMA, summary |
| `gofer_train/bench.py` | Throughput benchmark |
| `train_bootstrap.py` / `export_onnx.py` / `model.py` | Back-compatible entry points |
| `dataset.py`, `net_size_ablation.py` | Legacy row-level path, kept so ADR 0005 stays reproducible |
