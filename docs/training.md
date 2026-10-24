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
  champion/        gofer-9x9.onnx (the current champion) and gen-NNNN/ per champion
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
5. **Champion** (anchor generations only): if the new network is rated above
   the champion, it plays the champion directly (`champion_games` games at
   `champion_visits`, stopped early by SPRT `champion_sprt`) and becomes the
   new champion if it wins. The first network rated at least
   `champion_min_elo` becomes the first champion.

There is no gating for training: like AlphaZero and KataGo, the newest
network always plays the next self-play. The champion is only the best
network so far, the one worth downloading.

## Champions and releases

Every champion is copied to `champion/gen-NNNN/` (network, `champion.json`
with its rating, match and anchor results, and release notes), and
`champion/gofer-9x9.onnx` always holds the current one.

Pass `--release-repo OWNER/REPO` (or set `GOFER_RELEASE_REPO`) and each new
champion is also published as a GitHub Release, tagged
`model-<run>-gNNNN`, with the network as `gofer-9x9.onnx`. Because the asset
name never changes, this link always downloads the newest champion:

```
https://github.com/OWNER/REPO/releases/latest/download/gofer-9x9.onnx
```

Publishing uses the GitHub CLI on the training machine: run `gh auth login`,
or set `GH_TOKEN` to a fine-grained token with *Contents: read and write* on
the repository. If a release cannot be published (no network, expired
token), it stays pending and the loop retries it on the next generation;
`python -m training.release --run runs/9x9 --repo OWNER/REPO` shows the
champion and publishes anything pending. Only the newest pending champion is
published, so a long outage does not produce a burst of stale releases.

The repository's committed `models/gofer-9x9.onnx` (used by the website and
the examples) is not touched; update it deliberately when a release is worth
promoting.

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
