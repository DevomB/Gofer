# Deferred work

Known, evidenced, and deliberately not done yet. Each entry says what the
change is, why it was deferred, and how to tell whether it worked.

The ordering rule behind most of this: **establish a trustworthy baseline
before making anything faster.** Every defect in [ADR 0008](decisions/0008-search-correctness.md)
produced a plausible number rather than a crash, and an optimisation applied to
a broken measurement is indistinguishable from one applied to a sound one until
much later. Change one thing at a time and measure the same way twice.

---

## 1. The inference batcher serves one batch at a time

**Status:** wiring implemented (`a61f155`), deliberately **pinned off** for the
baseline run. Highest-value performance change available.

`EngineConfig.batch_size` exists and defaults to `0`, meaning "match the stage's
parallelism". The baseline run pins the old behaviour explicitly with
`--set engine.batch_size=8` so the loop being measured is the one that has
always run. Flipping it on is a config change, not a code change: drop that
`--set` and the stage parallelism is used. Do that only after the baseline
lands, and measure it as below.

`BatchedEvaluator.worker()` (`cmd/gofer/inference.go`) is a single goroutine:

```go
req := <-b.reqCh
batch := b.gatherBatch(req)     // up to minBatch, or maxWait
b.dispatchBatch(batch, boards)  // blocks on ORT
```

Nothing gathers and no other inference runs while `dispatchBatch` blocks, so
throughput is capped at `minBatch` evaluations per inference call, strictly
serialised. `minBatch` comes from `-batch-size`, whose flag default is 8
(`cmd/gofer/commands.go`). The orchestrator never passes it, and
`EngineConfig` has no field for it — while it *does* pass
`-selfplay-parallel` and `-arena-parallel` set to `os.cpu_count()`.

So on a 32-core box, 32 games queue behind a dispatcher serving 8 at a time.

**Evidence.** Load average was 2.67 on 32 cores during self-play: one goroutine
computing, the rest blocked on response channels. CPU-bound games would show
~32. Measured cost of putting the net in the loop was 8.3x (self-play 34.7s
heuristic-only against 287.0s at `onnx_fraction 0.7`, same 200 games), and the
gate — same evaluator, more of it — was 81% of a ~28 minute cycle.

**The change (done).** `EngineConfig.batch_size`, passed by both
`selfplay_command` and `arena_command`. Nothing in `cmd/gofer` changed: the flag
already existed and `evaluators.go` already threaded `evalConfig.BatchSize` into
the in-process and sidecar constructors. Two tests pin the wiring, because the
failure mode is only slowness - nothing fails, so nobody looks.

**Safe to do.** `reqTimeout` comes from `-eval-timeout` (2s), not from the
`maxWait*4` default, so a slower large batch cannot silently fall back to the
heuristic. Under-filled batches still dispatch after `maxWait`.

**How to tell it worked.** One cycle before, one after, same config, comparing
self-play and gate seconds — the way the 8.3x was measured. Watch
`eval_fallbacks` in the gate reports: if it climbs, batches are exceeding the
2s timeout and the change went too far.

**Why deferred.** No clean multi-cycle baseline existed at the time. Making an
unvalidated loop faster buys a faster unvalidated loop.

**Second lever, only if the ceiling still binds afterwards:** multiple dispatch
goroutines would lift the single-flight limit itself. That is a real
concurrency change; batch size is the cheap 80%.

---

## 2. Three `-run` omissions in the Makefile, one of which ships

**Status:** evidenced, unassigned, not done.

`go test` defaults `-run` to `.*`, so `go test -bench=X` executes the entire
test suite before reaching the benchmark. Fixed in
`.github/workflows/ci.yml`; three instances remain:

| line | effect |
|---|---|
| `Makefile:26` | writes `default.pgo` — **the Dockerfile builds the shipped binary with it** (`-pgo=default.pgo` when present) |
| `Makefile:20` | `legalmoves-cpu.prof` |
| `Makefile:23` | `legalmoves-mem.prof` |

The profiling ones are straightforwardly wrong: their stated purpose is
analysing legal-move generation, and the profile is dominated by everything
else. Anyone who opened `legalmoves-cpu.prof` looking for a hotspot was reading
the test suite.

The PGO one is not necessarily harmful — the test suite does exercise real
engine paths — but it is not the profile the target claims to produce, and
nobody can say what is in it. Regenerating `default.pgo` after the fix makes
the production binary a genuinely different artifact.

**Fix:** insert `-run '^$'` (matches no test) in all three.

---

## 3. The loop takes no absolute measurement after cycle 1

**Status:** structural. Mitigated by hand, not fixed in the pipeline.

`Pipeline.stage_gate` runs the vs-heuristic arena only when
`self.state.champion is None` — cycle 1. Every later cycle goes to
`_sprt_gate`, challenger against champion. `publish.regression_check` compares
the champion against a generation `regression_lookback` back, which is less
local but still relative.

So after cycle 1 the loop only ever measures motion relative to itself. That is
a ladder where every rung is measured against the rung below it: it can report
steady promotions while going nowhere absolute, and nothing in the run would
say so.

**Mitigation in use:** take anchors post-hoc against saved generations, with
the flags the seed gate used so the numbers compare directly to generation 1's
recorded Elo:

```
bin/gofer -arena -games 40 -size 9 -komi 6.5 -playouts 400 \
  -black-eval heuristic -white-eval onnx -eval-backend inprocess \
  -model runs/<run>/models/gen-00NN.onnx \
  -arena-play-all -seed 4242 -json anchor-gen00NN.json
```

`-arena-play-all` matters: the eval names differ, so without it the in-match
reject stop fires and truncates the sample.

Run these **after** a run, not during — a 40-game 400-playout arena contends
with a live pipeline for the same cores and corrupts the per-stage timings.
Label them post-hoc so they are never read as live gate data.

**If the anchor curve is flat while the SPRT keeps promoting, the gate is
measuring drift rather than strength.** That is the result worth having either
way, and the loop cannot currently produce it by itself.

---

## 4. A GPU is only reachable through the sidecar backend

**Status:** known limit, relevant to any AMD or NVIDIA plan.

`pick_providers()` in `training/inference_server.py` selects CUDA, ROCm or
MIGraphX at runtime from whatever ONNX Runtime reports, falling back to CPU.
That path works on any accelerator whose ORT build is installed.

The **in-process** backend cannot use a GPU at all: every build in
`ORT_BUILDS` (`training/pipeline/procs.py`) and in `infra/cloud/bootstrap.sh`
is a CPU build (`onnxruntime-linux-x64`, not `-x64-gpu`). Both
`pipeline-cpu.toml` and `pipeline-gpu.toml` set `backend = "inprocess"`.

**So using a GPU requires `backend = "sidecar"`**, regardless of driver or
provider. Verify before trusting any GPU timing:

```
curl -s localhost:8080/health | jq .providers
```

`["CPUExecutionProvider"]` on a GPU box means the card is not being used.

Adding a GPU ORT build to `ORT_BUILDS` would make the in-process path usable on
NVIDIA; there is no ROCm build of the Go binding, so AMD stays sidecar-only.

---

## 5. The net-size ablation needs re-running

**Status:** recorded in [ADR 0005](decisions/0005-net-size-ablation.md), not done.

ADR 0005 compared four architectures on a replay snapshot frozen before the
ADR 0008 defects, so its policy targets came from a search that never left the
root, and policy dominates the loss. Every validation-loss number in it
measures how well each net fit a target with no search in it.

The decision survives because it was a decision to change nothing. The reason
does not. **If an architecture question comes up, re-run it on post-fix shards
first.**

---

## 6. Memory ceiling above 9x9

**Status:** recorded as a limitation, deliberately not fixed.

`buildShardArrays` (Go) and `load_rows` (Python) both materialise every column
in memory. At 9x9 this is nothing. At 19x19, 100k rows x 22 planes x 361 points
is ~800MB in one contiguous allocation before the policy columns.

The Go side now fails with a clear error above 2 GiB rather than being killed
by the OS. The Python side has no bound; the right fix there is a batched
reader in the learner, which is not worth a speculative rewrite until a 19x19
run is actually planned.
