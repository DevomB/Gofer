# Training

## Setup

```bash
scripts/setup.sh          # CPU machine: GNU Go, ONNX Runtime, .venv, bin/gofer
scripts/setup.sh gpu      # NVIDIA machine: GPU ONNX Runtime and CUDA PyTorch
make test
```

## Run

```bash
.venv/bin/python -m training.loop --run runs/9x9 --gofer bin/gofer --hours 8
```

The run directory holds everything; rerun the same command to resume after an
interruption (each phase records completion in `state.json` only after its
outputs exist).

```
runs/9x9/
  config.json      frozen LoopConfig (edit before the first generation, or use --set)
  state.json       current generation, completed phases, all match results
  log.txt          human-readable log
  metrics.jsonl    per-generation self-play, training and evaluation metrics
  models/          gen-NNNN.onnx  (gen-0000 is the random initial network)
  ckpt/            gen-NNNN.pt    (model + optimizer state)
  data/gen-NNNN/   self-play shards (.npz)
  sgf/gen-NNNN/    every self-play game
  eval/            arena results (JSON) and ratings.json
```

Each generation:

1. **Self-play** `games_per_gen` games with the current network. Every move
   gets either a full search (`full_visits`, Dirichlet noise, recorded as a
   training row) with probability `full_prob`, or a cheap one (`fast_visits`,
   not recorded): KataGo's playout-cap randomization.
2. **Train** on a replay window of the newest rows (growing from `window_min`
   by `window_growth` per generation to `window_max`), drawing
   `sample_reuse` samples per new row, with a random board symmetry per
   sample. The loss on the new data *before* training (logged as
   "previous net on new data") is an honest held-out metric.
3. **Export** the network to ONNX and verify it against PyTorch.
4. **Evaluate**: new vs previous generation, and every `anchor_every`
   generations against fixed anchors (GNU Go by default). All results feed a
   Bradley-Terry fit in `eval/ratings.json` with `gen-0000` at 0 Elo.

There is no gating: like AlphaZero and KataGo, the newest network always
plays the next self-play. (`gofer arena -sprt 0,35` is available if you want
a promotion gate.)

Override settings when creating a run, e.g.
`--set games_per_gen=1000 --set full_visits=400 --set train.lr=1e-3`.

## Throughput

`bin/gofer bench -model runs/9x9/models/gen-0000.onnx` measures playouts per
second with 64 concurrent searches. On a 4-vCPU cloud VM (CPU inference):

| Network | Parameters | Playouts/s | Network positions/s |
|---------|-----------:|-----------:|--------------------:|
| 4×48 | 185k | 13,100 | 12,000 |
| 6×64 | 477k | 6,400 | 6,000 |
| 10×96 | 1.7M | 2,100 | 1,800 |

Search itself costs about 4 µs per playout (500k playouts/s on 4 cores with a
free evaluator), so inference dominates and throughput scales with cores. On
a GPU, use `-device cuda` (or `tensorrt`), and give the self-play enough
concurrent games to fill batches: `gofer selfplay -device cuda -workers 2
-batch 512 -parallel 1024`.

## Choosing a machine

Self-play inference is the bottleneck. With the 6×64 network on CPU, cost is
roughly proportional to cores, so a 32-vCPU CPU-optimized VM is about 8× the
4-vCPU numbers above. A GPU becomes worth it as the network grows (10×128 and
up) or for 13×13/19×19, where CPU inference falls far behind.
