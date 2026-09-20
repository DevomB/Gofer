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

**Status:** done (`2d12a58`). Kept for the reasoning, which generalises.

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

**Fixed** by inserting `-run '^$'` in all three, and by changing what
`pgo-profile` profiles. PGO wants a profile of what the binary does, which is
search; legal-move generation is a leaf inside it. The target now runs
`BenchmarkBestMove` and `BenchmarkSearchParallel` alongside `BenchmarkLegalMoves`.

The `default.pgo` found in the working tree was dated 29 June: `removeDeadGroups`
at 26% of samples, no MCTS anywhere in its top twelve, and samples attributed to
`Board.Neighbors`, deleted the same day it was found. So it profiled a test
suite, running on the pre-[ADR 0008](decisions/0008-search-correctness.md) engine
whose search never left the root, partly against symbols that no longer exist.
Deleted and regenerated. A clean checkout never had one; it is gitignored.

**Worth keeping from this:** a profile is an artifact with a provenance, and
nothing in the toolchain will tell you it describes a binary that no longer
exists. The regenerated one also measured something previously only suspected —
roughly 22% of search samples land in synchronisation (`procyieldAsm`,
`semasleep`, `semawakeup`, `preemptM`), which is the root-parallel contention in
[known-issues](known-issues.md).

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

---

## 7. `main` and this branch have diverged, and the website links to `main`

**Status:** structural, needs a decision rather than a fix.

`origin/main` still carries the twenty documents this branch reduced to fifteen,
including the four backlogs, both optimization files and the v3 plans. It has
ADRs 0001–0005; this branch has 0001–0008 plus this file. The engine on `main`
predates every fix in ADR 0008.

`web/index.html` links to `blob/main/...`. That is currently the only reason its
links resolve: it pointed at `docs/implementation-blueprint.md` for hours after
that file was deleted here. The links were changed to paths that exist in both
states, and the ADR count was removed from the prose because five is right on
one branch and eight on the other.

That is a patch over the real issue. Until the branch merges, the public site
describes a project state that is not on the branch it links to, and every new
document is unreachable from it.

**Decision needed, not a fix:** merge, or point the site at the branch and
accept churn at merge time. Whoever merges should re-check the site's links and
counts, which are measured in `54fcd93` and will drift again.

---

## 8. The fallback threshold is a guess

**Status:** shipped at 2%, never calibrated.

`maxEvalFallbackRate` in `cmd/gofer/eval_health.go` is 2%. Above it the arena
refuses to report and self-play refuses to write a shard. The number was chosen
so a handful of timeouts across a 600-game gate would not make gating flaky, not
from any measurement of what the fallback rate actually is in healthy operation.

**The data now exists to calibrate it.** Every gate report carries
`eval_fallback_rate` and every shard carries it in `ShardMeta`. The baseline run
at commit `0f93a7d` produced fifteen gate reports with a rate of exactly `0` — so
in a healthy in-process run the true rate is zero, not "small", and 2% may be
two orders of magnitude too generous.

**How to settle it:** read the distribution across a completed run, then set the
bar just above whatever healthy operation actually produces. Check the sidecar
backend separately before tightening: it crosses a process boundary and has a
real timeout, so its healthy rate is probably not zero.

---

## 9. The loop has not yet been shown to close

**Status:** the open question. Everything else is in service of it.

This is not a defect and not a task; it is what the current run is for, recorded
so the next person knows what the run was asking.

As of cycle 4 of the baseline run (`0f93a7d`, 32 vCPU, `pipeline-cpu.toml`,
`engine.batch_size=8`):

| cycle | outcome | games | note |
|---|---|---|---|
| 1 | seed promote → gen 1 | 40 vs heuristic | score 0.425, **−52 Elo**: the first net is weaker than the heuristic it learned from |
| 2 | **REJECT** | 320 | SPRT accepted H0 |
| 3 | **REJECT** | 160 | SPRT accepted H0 |
| 4 | in progress | — | first batch positive (0.600, LLR +0.60) |

Two consecutive rejections is not yet evidence of failure — 200 games and ~16k
rows per cycle is very little data, and gen 1 is itself weak. But it is the
thing to watch, and it is the reason the batch-size optimisation was deferred:
making an unvalidated loop faster buys a faster unvalidated loop.

**What would settle it:** the anchor curve in §3. If the vs-heuristic score
climbs toward and past 0.5, the loop closes and the net stops being a worse
evaluator than the hand-written one. If it is flat at cycle 20 while the SPRT
keeps promoting, the gate is measuring drift.

**Cycle cost at this configuration,** for anyone sizing a future run:

| stage | cycle 2 | share |
|---|---|---|
| self-play | 287s | 17% |
| train | 27s | 2% |
| gate | 1343s | 81% |

The gate dominates, and it is inference-bound for the same reason self-play is
(§1). Cycle 1 is not representative: it is heuristic-only, and at 229s it is the
cheapest cycle a run will ever have. Do not extrapolate from it — that mistake
produced a confident and wrong conclusion that a GPU would not help this
workload.
