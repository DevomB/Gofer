# ML pipeline v3 — implementation spec

Scope: fix G1–G13, ship PR1–PR5, restart Lightsail loop from cycle-2
`best.pt` / `gofer-9x9-best.onnx`. Budget: ≤ $20/mo on `small_3_0` ($12/mo).

---

## 1. Problem statement

`scripts/weekly-train-loop.sh` retrains `GoferBootstrapNet()` from random init each cycle
on a single `samples-cycleN.jsonl`. Arena uses the new ONNX regardless of outcome.
`gofer-9x9-best.onnx` is copied on improvement but never loaded into training or
self-play. Result: win rate is not a learning curve; it is independent samples
(33% → 58% → 35%).

Cycle-2 artifacts on server `54.90.212.111`:

- `training/checkpoints/cycle2/best.pt` (1.5 MB)
- `models/gofer-9x9-best.onnx` (1.5 MB, policy_size=82)
- `training/data/samples-cycle{1,2}.jsonl`

---

## 2. Gap register (G1–G13)

| ID | Location | Defect | Fix (PR) |
|----|----------|--------|----------|
| G1 | `train_bootstrap.py:19` | `GoferBootstrapNet()` every run | `--resume training/state/best.pt` (PR1) |
| G2 | `weekly-train-loop.sh:76-79` | `gofer-9x9-best.onnx` write-only | Load as init + sidecar parent (PR2) |
| G3 | `weekly-train-loop.sh:50` | Single-cycle JSONL | Append to `replay.jsonl`, FIFO cap 50k rows (PR2) |
| G4 | `selfplay.go:79` | `Heuristic{}` hardcoded | `-selfplay-eval mix` + ONNX sidecar (PR3) |
| G5 | `train_bootstrap.py:37` | `best.pt` overwritten every epoch | Save on min val loss; `last.pt` separate (PR1) |
| G6 | loop selfplay 100 pl, arena 400 pl | Distribution shift | Self-play export at 200 pl (PR2 script) |
| G7 | `inference.go:179`, loop `WIN_TARGET` | 0.55 vs 0.75 vs none | Single `gating.env` / constants (PR5) |
| G8 | loop export always → `bootstrap.onnx` | Worse net deployed | Promote only if arena beats `manifest.best_win_rate + 0.02` (PR2) |
| G9 | no `training/test_*.py` | No train/export regression | pytest in CI (PR5) |
| G10 | `remote-arena-gate.sh:41` | Random ONNX if missing | Require checkpoint or fail (PR5) |
| G11 | `inference_server.py` | Ignores batch dim; CPU only | Batch ORT run; optional CUDA provider (PR4) |
| G12 | loop | No manifest / crash resume | `training/state/manifest.json` (PR2) |
| G13 | `aws-run-arena.sh fetch` | One JSON only | `fetch-all` for cycle reports (PR4) |

Deferred (not in v3): ownership / `policy_opp` heads in loss (exported in JSONL, unused in trainer).

---

## 3. Target artifact graph

```
training/state/manifest.json     # {best_win_rate, cycle, replay_rows, promoted_at}
training/state/best.pt           # min-val-loss weights (monotonic promote)
training/state/last.pt           # final epoch (debug)
training/data/replay.jsonl       # FIFO append-only buffer
models/gofer-9x9-best.onnx       # deployed ORT model (opset 18)
models/gofer-9x9-candidate.onnx   # pre-arena export this cycle
models/gofer-9x9-bootstrap.onnx   # alias → best.onnx (CLI compat)
```

ONNX I/O (schema v2, 9×9 bootstrap):

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `spatial_input` | `[B, 8, 9, 9]` NCHW | float32 |
| `global_input` | `[B, 4]` | float32 |
| `policy_logits` | `[B, 82]` | float32 |
| `value` | `[B, 1]` tanh in PyTorch | float32 |

Sidecar applies softmax on policy; Go `BuildFeaturesV2` must match golden
`cmd/gofer/testdata/features_golden_v2.json`.

---

## 4. Cycle state machine (replaces weekly loop)

File: `scripts/train-loop-v3.sh`

```
INIT (once):
  cp cycle2/best.pt → training/state/best.pt
  cat samples-cycle1.jsonl samples-cycle2.jsonl → replay.jsonl
  manifest.best_win_rate = 0.58

EACH cycle N:
  1. selfplay NEW_GAMES → samples-cycleN.jsonl
  2. replay.append(cycleN); replay.trim(max=50000)     # O(n) append, O(n) trim if over cap
  3. train --data replay.jsonl --resume best.pt
  4. export candidate.onnx from best.pt (post-train)
  5. sidecar --model candidate.onnx
  6. arena 200 × (heuristic vs onnx), 400 pl, seed 42+N
  7. if win_rate > manifest.best_win_rate + 0.02 OR win_rate >= WIN_TARGET:
       cp weights → best.pt; cp candidate → best.onnx; update manifest
     else:
       sidecar --model best.onnx; log regression; no overwrite
  8. N += 1; stop if WIN_TARGET or deadline
```

Promotion uses Wilson CI in JSON for logging only; decision is point estimate + 2pp
margin over historical best (200-game variance ≈ ±7pp at 50%).

---

## 5. PR breakdown

### PR1 — trainer (`training/train_bootstrap.py`)

- `--resume PATH` | `--init-from PATH`
- 90/10 train/val split (shuffle seed fixed)
- Track `best_val_loss`; write `best.pt` only on improvement
- Write `last.pt` every epoch
- LR: `0.01` fresh, `0.001` resume; epochs 25 / 15
- Early stop patience 5 on val loss
- `training/test_train.py`: assert `best.pt` ≠ `last.pt` when val diverges

Commit: `fix(training): resume checkpoint and val-based best.pt (G1, G5)`

### PR2 — replay + loop (`training/replay.py`, `scripts/train-loop-v3.sh`)

- `append_jsonl(src, dst)` — stream copy, no full load
- `trim(path, max_lines=50000)` — keep tail (FIFO)
- `training/state/manifest.json` read/write
- Monotonic promote (G2, G3, G8, G12)
- Self-play `-playouts 200` in script
- `SEED_FROM_CYCLE2=1` init path for first deploy

Commit: `feat(training): replay buffer and train-loop-v3 monotonic promote (G2,G3,G6,G8,G12)`

### PR3 — self-play eval (`cmd/gofer/selfplay.go`, `cmdline.go`)

- `-selfplay-eval heuristic|onnx|mix` (default `mix`)
- `-selfplay-onnx-fraction 0.7`
- Reuse `parseEvaluator` / `-onnx-url` / `-eval-timeout`
- Odd games: both sides ONNX; even: both heuristic (throughput + diversity)

Commit: `feat(selfplay): wire -eval onnx and mix mode (G4)`

### PR4 — sidecar + AWS ops

- `inference_server.py`: batch ORT `session.run` on stacked `[B,8,9,9]`
- SIGHUP reload model path
- `aws-run-arena.sh`: `start-v3`, `stop-loop`, `fetch-all`, `seed-status`

Commit: `feat(ops): batched sidecar and aws-run-arena v3 commands (G11, G13)`

### PR5 — gating, CI, docs

- `scripts/gating.env`: `PROMOTE_MIN=0.55`, `WIN_TARGET=0.75`, `MARGIN=0.02`
- `remote-arena-gate.sh`: fail if no checkpoint when `SELFPLAY_GAMES>0`
- `training/test_export.py`: PyTorch vs ORT max abs diff < 1e-4 on zeros input
- CI: `pytest training/`
- `docs/decisions/0003-iterative-training-loop.md`

Commit: `chore(ml): unify gating, training tests, ADR 0003 (G7,G9,G10)`

---

## 6. Lightsail ($12/mo cap)

| Item | Spec | $/mo |
|------|------|------|
| Instance | `small_3_0`, us-east-1, 2 vCPU, 2 GB | 12 |
| Snapshot | 1 before code deploy | ~0.50 |
| GPU burst | Optional LS Research GPU-XL ≤3 hr | ≤7 |
| **Ceiling** | | **≤19** |

Do not run GPU 24/7 (~$2.37/hr). Bootstrap net (32ch, 4 ResBlock) trains in
minutes on CPU; bottleneck is arena MCTS + ORT HTTP (~6 h / 200 games on 2 GB).

Keep instance `gofer-v25-arena` / `54.90.212.111`. Resize to `medium_3_0` ($24)
only if budget raised.

Deploy sequence after PR1+PR2 merge:

```bash
bash scripts/aws-run-arena.sh 54.90.212.111 stop-loop
git pull  # on server via start-v3
SEED_FROM_CYCLE2=1 WEEK_DAYS=14 NEW_SELFPLAY_PER_CYCLE=200 \
  bash scripts/aws-run-arena.sh 54.90.212.111 start-v3
```

---

## 7. Commit policy during integration

One commit per PR unit above. Do not squash G1/G5 fix with loop rewrite.
Tag after PR2: `ml-v3-loop-seed` (cycle-2 baseline runnable).

Do not commit: `.tectonix/session-baseline.json`, `*.pem`, `lightsail-key.json`,
large `.pt` / `.onnx` artifacts (fetch to `.tectonix/artifacts/` locally).

---

## 8. Acceptance checks

| Check | Command |
|-------|---------|
| Trainer resume | `pytest training/test_train.py` |
| ORT parity | `pytest training/test_export.py` |
| Go unit | `go test ./... -short` |
| Live sidecar | `go test -tags=onnx_integration ./cmd/gofer/...` |
| Loop smoke | `SELFPLAY_GAMES=5 WEEK_DAYS=0.001 bash scripts/train-loop-v3.sh` (local) |
| Arena gate | challenger win rate ≥ previous best + 0.02 or ≥ 0.75 |

---

## 9. Out of scope (v3)

- 19×19 / progressive channel scaling (ML-5.3)
- SWA / EMA (ML-4.3)
- Ownership auxiliary loss
- 90% win-rate target (needs 10k+ self-play games; not bounded by this spec)
