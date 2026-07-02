# COPY-PASTE PROMPT — Gofer ML Pipeline v3 Implementation

Paste everything below the line into a new Cursor agent session.

---

## YOUR MISSION

Implement **Gofer ML Pipeline v3** in repo `C:/Coding-Projects/GoEngine` (GitHub: `DevomB/Gofer`). Fix gaps **G1–G13**, ship **PR1–PR5** in order, run Tectonix session tracking, commit after each PR, deploy to existing Lightsail box when PR1+PR2 are green. Budget: **≤ $20/mo AWS**.

**Authoritative spec on disk:** `docs/plans/ml-pipeline-v3.md` (commit `f7b6319`+). Read it first. This prompt adds operational context the spec omits.

---

## REPO CONTEXT (v2.5 baseline — already shipped)

Monolithic Go engine: `cmd/gofer/`. Python training: `training/`. No separate packages.

| Area | Key files |
|------|-----------|
| ONNX sidecar client | `cmd/gofer/onnx_sidecar.go` — `SidecarBackend`, HTTP to Python |
| Batched MCTS eval | `cmd/gofer/inference.go` — `BatchedEvaluator`, heuristic fallback |
| Features v2 | `cmd/gofer/features.go` — `BuildFeaturesV2`, 8 spatial planes + 4 globals |
| Self-play | `cmd/gofer/selfplay.go` — **BUG: line 79 hardcodes `Heuristic{}`** |
| Arena / gating | `cmd/gofer/match.go`, `cmd/gofer/inference.go` — `GatingHarness.Pass` @ 0.55 |
| CLI | `cmd/gofer/cmdline.go`, `cmd/gofer/cli.go` |
| Trainer | `training/train_bootstrap.py` — **BUG: fresh net every run, best.pt = last epoch** |
| Export | `training/export_onnx.py` — opset 18, inputs `spatial_input`/`global_input` |
| Sidecar | `training/inference_server.py` — ORT CPU, `/health`, `/v1/eval` |
| Model | `training/model.py` — `GoferBootstrapNet`, 32ch, 4 ResBlock, 9×9, policy_size=82 |
| Dataset | `training/dataset.py` — JSONL loader, ignores `ownership`/`policy_opp` |
| Broken loop | `scripts/weekly-train-loop.sh` — replace with `scripts/train-loop-v3.sh` |
| AWS driver | `scripts/aws-run-arena.sh` — extend, do not rewrite from scratch |
| One-shot gate | `scripts/remote-arena-gate.sh` |
| Feature schema | `docs/model-input-schema.md` |
| Inference ADR | `docs/decisions/0001-inference-backend.md` |
| ML backlog | `docs/backlog-ml-integration.md` |

**Do NOT commit:** `.tectonix/session-baseline.json`, `*.pem`, `lightsail-key.json`, large `.pt`/`.onnx` blobs. SSH key path: `.tectonix/gofer-v25-run.pem` (gitignored).

---

## ROOT PROBLEM (why v3 exists)

`weekly-train-loop.sh` each cycle:
1. Self-play with heuristic only → `samples-cycleN.jsonl`
2. `GoferBootstrapNet()` **from random init** (G1)
3. Train on **that cycle's file only** (G3)
4. Export → overwrite `models/gofer-9x9-bootstrap.onnx` **always** (G8)
5. Arena; if best rate, copy to `gofer-9x9-best.onnx` but **never load it next cycle** (G2)

Observed arena win rates (200 games, heuristic vs ONNX, 400 playouts, 9×9):
- Cycle 1: 33.0%
- Cycle 2: **58.0%** (seed model — keep this)
- Cycle 3: 35.5% (regression from retrain-from-scratch)
- Cycle 4: training loss spiked epoch 7 (~3.5 → ~5.0); exported last epoch (G5)

Cycle 2 `promoted=true` in JSON only means challenger ≥ 55% (`GatingHarness`). Weekly loop stop target is **75%** (`WIN_TARGET`) — inconsistent (G7).

---

## V3 TARGET BEHAVIOR (one lineage, monotonic deploy)

**Persistent server state:**
```
training/state/manifest.json
training/state/best.pt          # min val-loss; matches deployed ONNX
training/state/last.pt            # debug
training/data/replay.jsonl        # FIFO cap 50_000 rows
models/gofer-9x9-best.onnx        # production ORT model
models/gofer-9x9-candidate.onnx   # this cycle pre-arena
models/gofer-9x9-bootstrap.onnx # copy/symlink of best.onnx (CLI compat)
```

**ONNX I/O (schema v2, bootstrap 9×9):**
| Tensor | Shape | Notes |
|--------|-------|-------|
| `spatial_input` | `[B, 8, 9, 9]` NCHW | 648 floats/board |
| `global_input` | `[B, 4]` | komi/10, move_frac, black, white |
| `policy_logits` | `[B, 82]` | pass = index 81 |
| `value` | `[B, 1]` | tanh in PyTorch; sidecar returns raw |

Golden: `cmd/gofer/testdata/features_golden_v2.json`. Sidecar softmax on policy.

**Invariants after every cycle:**
1. `gofer-9x9-best.onnx` = strongest **arena-verified** net
2. `training/state/best.pt` weights match that ONNX
3. Training always `--resume training/state/best.pt` (unless `--fresh` one-shot)
4. Replay buffer append-only, FIFO trim at 50k
5. Arena loser **does not** replace deployed model

**Cycle state machine (`scripts/train-loop-v3.sh`):**
```
INIT (once, SEED_FROM_CYCLE2=1):
  cp training/checkpoints/cycle2/best.pt → training/state/best.pt
  cat samples-cycle1.jsonl samples-cycle2.jsonl → replay.jsonl
  manifest: best_win_rate=0.58, best_cycle=2, version=3

EACH cycle N:
  1. Self-play NEW_SELFPLAY_PER_CYCLE games → samples-cycleN.jsonl
     (after PR3: mix 70% ONNX / 30% heuristic; before PR3: heuristic OK)
     -playouts 200, -full-only true, -size 9
  2. replay.append(cycleN → replay.jsonl); replay.trim(50000)
  3. python train_bootstrap.py --data replay.jsonl --resume training/state/best.pt
  4. export → models/gofer-9x9-candidate.onnx from training/state/best.pt
  5. sidecar on candidate.onnx
  6. arena 200 games: -black-eval heuristic -white-eval onnx -playouts 400
     -eval-timeout 2s -arena-enhanced none -seed $((42+N))
  7. PROMOTE if win_rate > manifest.best_win_rate + 0.02 OR win_rate >= WIN_TARGET (0.75):
       cp candidate → best.onnx; update best.pt; update manifest; log training-history/cycle-N.json
     ELSE:
       revert sidecar to best.onnx; log regression; do NOT overwrite best.*
  8. N++; repeat until WIN_TARGET or WEEK_DAYS deadline
```

Promotion margin **2pp** over historical best avoids 200-game noise (Wilson CI at 50% ≈ ±7pp). Log Wilson CI; decide on point estimate + margin.

---

## GAP REGISTER — implement all

| ID | Defect | Fix |
|----|--------|-----|
| G1 | Fresh `GoferBootstrapNet()` every cycle | `--resume` / `--init-from` in trainer (PR1) |
| G2 | `gofer-9x9-best.onnx` write-only | Load for resume + self-play sidecar (PR2, PR3) |
| G3 | Per-cycle JSONL only | `training/replay.py` FIFO buffer (PR2) |
| G4 | `selfplay.go:79` hardcoded heuristic | `-selfplay-eval mix` (PR3) |
| G5 | `best.pt` = last epoch | Val split, save on min val loss (PR1) |
| G6 | Self-play 100 pl vs arena 400 pl | Script uses `-playouts 200` (PR2) |
| G7 | 0.55 vs 0.75 vs none | `scripts/gating.env` (PR5) |
| G8 | Worse net always deployed | Monotonic promote in loop (PR2) |
| G9 | No Python tests | `training/test_*.py` + CI pytest (PR5) |
| G10 | Random ONNX if missing checkpoint | Fail in remote-arena-gate (PR5) |
| G11 | Sidecar no batching | Batch ORT in inference_server (PR4) |
| G12 | No manifest / crash resume | `manifest.json` (PR2) |
| G13 | `fetch` one JSON | `fetch-all` subcommand (PR4) |
| G14 | `ownership`/`policy_opp` unused | **Defer** — do not scope creep |

---

## PR1 — Trainer (G1, G5) — BLOCKS EVERYTHING

**File:** `training/train_bootstrap.py`

Add CLI:
- `--resume PATH` — load state_dict if exists
- `--init-from PATH` — one-time seed (cycle2)
- `--val-split 0.1` — default 0.1
- `--lr` — default 0.01 fresh, script passes 0.001 on resume
- `--epochs` — 25 fresh, loop passes 15 on resume
- `--patience 5` — early stop on val loss plateau

Logic:
- Fixed shuffle seed for train/val split
- Each epoch: train loss + val loss
- Save `best.pt` only when val improves
- Save `last.pt` every epoch
- Return path to best.pt

**File:** `training/test_train.py` (new)
- Tiny fixture JSONL (5–10 rows) in `training/testdata/`
- Train 5 epochs where val diverges; assert `best.pt` ≠ `last.pt` byte-wise or by epoch metadata
- Test `--resume` continues from prior weights (same loss on epoch 1 resume ≈ prior epoch 0)

**Commit message:**
```
fix(training): resume checkpoint and val-based best.pt (G1, G5)
```

Run before commit:
```bash
cd C:/Coding-Projects/GoEngine
pytest training/test_train.py -q
go test ./... -short
```

---

## PR2 — Replay + loop (G2, G3, G6, G8, G12) — BLOCKS SERVER RESTART

**File:** `training/replay.py` (new)
```python
def append_jsonl(src: Path, dst: Path) -> int  # stream lines, skip duplicate headers optional
def trim(path: Path, max_lines: int = 50000) -> int  # keep tail FIFO
def count_lines(path: Path) -> int
```

**File:** `training/manifest.py` (new, optional) or shell+json in script
```json
{
  "version": 3,
  "cycle": 3,
  "best_win_rate": 0.58,
  "best_cycle": 2,
  "replay_rows": 8100,
  "last_promoted_at": "2026-07-01T14:30:27Z",
  "seed": "cycle2"
}
```

**File:** `scripts/train-loop-v3.sh` (new — do not patch weekly in place; keep weekly for reference)
- Source `scripts/gating.env` if present
- Env vars: `WEEK_DAYS`, `WIN_TARGET`, `NEW_SELFPLAY_PER_CYCLE`, `SEED_FROM_CYCLE2`, `REPLAY_MAX`
- Init block when `SEED_FROM_CYCLE2=1`
- Per-cycle logging to `train-v3.log` and `.tectonix/reports/training-history/cycle-N.json`
- On promote: `cp models/gofer-9x9-candidate.onnx models/gofer-9x9-best.onnx && cp models/gofer-9x9-best.onnx models/gofer-9x9-bootstrap.onnx`
- On reject: restore sidecar to best.onnx
- Exit 0 on WIN_TARGET hit

**File:** `scripts/gating.env` (stub in PR2, full in PR5)
```bash
WIN_TARGET=0.75
PROMOTE_MIN=0.55
PROMOTE_MARGIN=0.02
```

**Commit message:**
```
feat(training): replay buffer and train-loop-v3 monotonic promote (G2,G3,G6,G8,G12)
```

Tag after this commit: `git tag ml-v3-loop-seed`

Smoke test locally:
```bash
SEED_FROM_CYCLE2=0 NEW_SELFPLAY_PER_CYCLE=2 WEEK_DAYS=0.001 bash scripts/train-loop-v3.sh
# (needs fixture data or skip if no go binary env)
```

**DO NOT restart Lightsail until PR1 + PR2 pass tests.**

---

## PR3 — Self-play ONNX (G4)

**Files:** `cmd/gofer/selfplay.go`, `cmd/gofer/cmdline.go`

Add to `SelfplayConfig`:
- `EvalMode string` — `heuristic|onnx|mix`
- `ONNXFraction float64` — default 0.7
- Wire `ONNXURL`, `EvalTimeout`, `BatchSize` from global flags

Replace `Heuristic{}` in `playSelfplayGameWithLog`:
- `heuristic`: both engines heuristic (current behavior)
- `onnx`: both engines via `parseEvaluator("onnx", ...)`
- `mix`: game index odd → onnx/onnx; even → heuristic/heuristic

CLI:
```
-selfplay-eval heuristic|onnx|mix   # default mix in train-loop-v3.sh
-selfplay-onnx-fraction 0.7
```
Reuse existing `-onnx-url`, `-eval-timeout`, `-batch-size`.

Update `train-loop-v3.sh` to pass `-selfplay-eval mix` and start sidecar on **best.onnx** before self-play.

**Tests:** `cmd/gofer/selfplay_test.go` or extend existing — mock sidecar via httptest, verify samples emitted with `-selfplay-eval onnx`.

**Commit:**
```
feat(selfplay): wire -eval onnx and mix mode (G4)
```

---

## PR4 — Sidecar batch + AWS ops (G11, G13)

**File:** `training/inference_server.py`
- Stack batch from request `spatial[]` / `globals[]` into `[B,8,9,9]` and `[B,4]`
- Single `session.run` per batch
- SIGHUP handler: reload `--model` path
- Optional: `CUDAExecutionProvider` if torch/cuda available (try/except, fall back CPU)
- Log request latency ms to stderr every N requests

**File:** `scripts/aws-run-arena.sh` — add subcommands (keep existing ones):

| Subcommand | Action |
|------------|--------|
| `stop-loop` | `kill week.pid`; `pkill inference_server`; `pkill train-loop-v3`; `pkill weekly-train-loop` |
| `start-v3` | `git pull`, `nohup train-loop-v3.sh`, write `week.pid` |
| `fetch-all` | scp `arena-cycle-*.json`, `training-history/*`, `train-v3.log`, `manifest.json` |
| `seed-status` | ssh cat manifest + ls -la best.pt best.onnx |
| `week-status` | tail train-v3.log (keep backward compat with week.log) |

**Commit:**
```
feat(ops): batched sidecar and aws-run-arena v3 commands (G11, G13)
```

---

## PR5 — Gating, CI, docs (G7, G9, G10)

**Files:**
- `scripts/gating.env` — canonical constants; source from all shell scripts
- `remote-arena-gate.sh` — if `SELFPLAY_GAMES>0` and no checkpoint after train, exit 1; if `ENFORCE_GATE=1` and win_rate < WIN_TARGET, exit 1
- `training/test_export.py` — export fixture checkpoint, ORT run, max abs diff vs PyTorch < 1e-4
- `training/test_replay.py` — append + trim FIFO
- `.github/workflows/ci.yml` — add `pip install pytest` + `pytest training/`
- `docs/decisions/0003-iterative-training-loop.md` — replay, resume, monotonic promote
- Update `docs/backlog-ml-integration.md` — ML-5.5 iterative loop
- Update `README.md`, `CHANGELOG.md`, `models/README.md`

**Commit:**
```
chore(ml): unify gating, training tests, ADR 0003 (G7,G9,G10)
```

---

## TECTONIX WORKFLOW (mandatory for this session)

Install if missing:
```bash
cargo install --git https://github.com/DevomB/Tectonix --force
export PATH="$HOME/.cargo/bin:$PATH"
tectonix --version
```

**At session start (repo root):**
```bash
cd C:/Coding-Projects/GoEngine
tectonix scan .
tectonix health . 2>/dev/null | jq '.root_causes'
tectonix test-gaps . 2>/dev/null | jq '{coverage_ratio, riskiest_untested: .riskiest_untested[:5]}'
tectonix session-start .
```

**Before editing risky files** (`selfplay.go`, `match.go`, `train_bootstrap.py`):
```bash
tectonix git-stats . 2>/dev/null | jq '{hotspot_count, top_hotspots: .top_hotspots[:5]}'
```

**After each PR commit:**
```bash
go test ./... -short
pytest training/ -q
tectonix rescan .
```

**At session end:**
```bash
tectonix session-end .
```
Report: quality_signal delta, weakest root_cause, whether modularity/acyclicity regressed.

Save reports optionally:
```bash
mkdir -p .tectonix/reports
tectonix health . 2>/dev/null > .tectonix/reports/health.json
```
Do not commit `session-baseline.json` unless user asks.

---

## AWS / LIGHTSAIL DEPLOY

| Resource | Value |
|----------|-------|
| Instance name | `gofer-v25-arena` |
| IP | `54.90.212.111` (verify with `aws lightsail get-instance`) |
| Bundle | `small_3_0` — 2 vCPU, 2 GB, **$12/mo** |
| Region | `us-east-1` |
| SSH key | `.tectonix/gofer-v25-run.pem` |
| Repo on box | `~/Gofer` |

**Budget cap $20/mo:** small_3_0 ($12) + snapshot (~$0.50) + optional ≤3 GPU-hr burst (~$7). **No 24/7 GPU.** No `medium_3_0` ($24) unless user raises budget.

**Phase 0 before deploy:**
```bash
# Stop broken loop
bash scripts/aws-run-arena.sh 54.90.212.111 stop-loop

# Preserve artifacts locally (do not git commit)
mkdir -p .tectonix/artifacts/cycle2-seed
scp -i .tectonix/gofer-v25-run.pem ubuntu@54.90.212.111:~/Gofer/models/gofer-9x9-best.onnx .tectonix/artifacts/cycle2-seed/
scp -i .tectonix/gofer-v25-run.pem ubuntu@54.90.212.111:~/Gofer/training/checkpoints/cycle2/best.pt .tectonix/artifacts/cycle2-seed/
scp -i .tectonix/gofer-v25-run.pem ubuntu@54.90.212.111:~/Gofer/training/data/samples-cycle*.jsonl .tectonix/artifacts/cycle2-seed/
```

**Deploy after PR1+PR2 merged and pushed:**
```bash
SEED_FROM_CYCLE2=1 WEEK_DAYS=14 NEW_SELFPLAY_PER_CYCLE=200 WIN_TARGET=0.75 \
  bash scripts/aws-run-arena.sh 54.90.212.111 start-v3

# Poll
bash scripts/aws-run-arena.sh 54.90.212.111 week-status
bash scripts/aws-run-arena.sh 54.90.212.111 fetch-all
```

**When done / budget save:**
```bash
bash scripts/aws-run-arena.sh 54.90.212.111 destroy
```

---

## COMMIT POLICY

- One commit per PR unit (5 commits minimum for v3)
- Commit messages: imperative, reference G* ids
- Tag `ml-v3-loop-seed` after PR2
- Push to `main` after each PR if user has been pushing; ask before force push
- Never commit secrets or `.tectonix/session-baseline.json`
- Run tests before every commit

---

## CODING RULES

- **Minimal diff** — fix the shared path once (trainer, loop script), not per-cycle hacks
- **Match existing style** — Go tab, Python type hints, same error handling patterns
- **Reuse** — `parseEvaluator`, `SidecarBackend`, `export_onnx.py`, `SampleDataset`
- **No new deps** unless pytest already implied; add `pytest` to `training/requirements.txt` dev or CI only
- **No scope creep** — G14 ownership heads, 19×19, SWA, 90% win target are out of scope
- **Lazy senior dev** — delete/replace `weekly-train-loop` usage in docs, don't maintain two loops forever

---

## ACCEPTANCE CHECKLIST (all must pass before declaring v3 done)

- [ ] `pytest training/` green
- [ ] `go test ./... -short` green
- [ ] `go test -tags=onnx_integration ./cmd/gofer/...` green (with sidecar)
- [ ] `train-loop-v3.sh` promotes on improvement, rejects regression (manual or smoke script)
- [ ] `best.pt` survives bad arena cycle unchanged on disk
- [ ] `manifest.json` updates correctly
- [ ] Sidecar serves `gofer-9x9-best.onnx` after rejected cycle
- [ ] ADR 0003 written
- [ ] CHANGELOG entry for v2.6 or v3.0
- [ ] Tectonix session-end shows no major regression
- [ ] Lightsail loop running with cycle ≥ 3 and win_rate ≥ 0.58 floor

---

## EXPECTED OUTCOMES (realistic, not marketing)

| Metric | v2.5 broken | v3 target |
|--------|-------------|-----------|
| Cycle-to-cycle | 33→58→35 lottery | Best ≥ 58% floor, gradual climb |
| 75% gate | Unlikely | Possible over weeks @ $12/mo |
| 90% gate | Not credible | Out of scope — needs 10k+ games, bigger net |

---

## IMPLEMENTATION ORDER (strict)

```
PR1 → PR2 → [deploy seed to Lightsail] → PR3 → PR4 → PR5
```

Do not parallelize PR1 and PR2. Do not deploy before PR2. PR3 can ship after deploy but self-play mix improves data quality — prioritize before long server run.

---

## IF STUCK

- Read `docs/plans/ml-pipeline-v3.md`
- Read `docs/decisions/0001-inference-backend.md` for HTTP eval protocol
- Read `docs/model-input-schema.md` for tensor layout
- Arena JSON format: `cmd/gofer/match.go` `MatchResult`
- Sample JSONL: run `go run ./cmd/gofer -selfplay -games 1 -o /tmp/s.jsonl` and inspect header row

---

## START NOW

1. `tectonix session-start .`
2. Read `docs/plans/ml-pipeline-v3.md`
3. Implement **PR1** completely, test, commit
4. Implement **PR2** completely, test, commit, tag `ml-v3-loop-seed`
5. Continue PR3–PR5
6. Deploy Lightsail when PR2 green
7. `tectonix session-end .` and report results

Do not ask for permission to start PR1 — user has approved v3 plan.

---
