# In-process ONNX Runtime inference (Phase 1)

## Status

Accepted — default since Step 5 (July 2026). Supersedes the sidecar-as-default posture in [0001-inference-backend.md](./0001-inference-backend.md) for production training and arena.

## Context

Gofer v2.5 ran ONNX inference through a Python HTTP sidecar (`training/inference_server.py`) behind `SidecarBackend`. That worked for bootstrapping but created measurable overhead and operational friction on the Lightsail training box:

1. **HTTP and JSON serialization** — every batched eval crossed a process boundary with encode/decode cost on both sides.
2. **Dual-sidecar thread oversubscription during arena** — champion and challenger each ran a separate Python ORT process; with `arena-parallel=2` the host ran four inference stacks (two Go processes × two sidecars) competing for two vCPUs.
3. **Effective batch size stuck at 1–2** — `BatchedEvaluator` in Go could aggregate positions, but the sidecar served requests one HTTP batch at a time with no cross-request batching; under parallel self-play/arena, batches rarely reached `minBatch=8`.

Memory and latency targets were written assuming a **4 GiB** Lightsail plan. Production validation (Step 3) revealed the live instance is **`t3.small` (~2 GiB RAM, 1910 MiB reported by `free -m`)** — not 4 GiB. In-process inference still fits (see Evidence), but prior headroom assumptions should be revisited separately.

## Options

### (a) In-process ONNX Runtime via CGO (`onnxruntime_go`) — **chosen**

Link Microsoft's ORT C API into the Go binary (`//go:build onnx`). `ORTBackend` implements the same `EvalBackend` interface as `SidecarBackend`; `BatchedEvaluator` unchanged.

**Why chosen:**

- Removes HTTP/JSON overhead; eval stays in-process.
- Arena loads two ONNX sessions inside one Go binary instead of two Python processes — fewer threads, less RSS duplication.
- Batches gathered in Go are passed directly to ORT without a second hop.
- Build tag keeps default `go test ./...` CGO-free; production build uses `make build-onnx`.
- Parity harness (`scripts/parity-onnx.sh`) proves bit-exact agreement with the Python reference path at ORT **1.26.0** (matches `onnxruntime_go` v1.31.0).

### (b) Hand-written Go forward pass — **ruled out**

Reimplement conv blocks, policy/value heads, and softmax in pure Go.

**Why ruled out:**

- High implementation and maintenance cost; every architecture/export change needs a manual port.
- Risk of subtle numeric drift vs PyTorch export with no single ORT reference to diff against.
- ORT already ships optimized CPU kernels; reinventing them buys little over (a) while costing a lot.

## Decision

- **Default `-eval-backend=inprocess`** in `cmd/gofer` and `EVAL_BACKEND=inprocess` in `scripts/train-loop-v3.sh`.
- **`SidecarBackend` and `training/inference_server.py` remain** in the tree, unused by default, for one more production cycle before deletion.
- Sidecar path still available via `-eval-backend=sidecar` or `EVAL_BACKEND=sidecar` for rollback.

## Evidence (Lightsail `t3.small`, linux/amd64, July 2026)

| Check | Sidecar baseline (cycle 23) | In-process validation |
|-------|----------------------------|------------------------|
| `PARALLEL` | 2 | 2 |
| Arena games | 200 | 45 (early-stop; 50 requested) |
| Win rate (challenger) | 0.505 | 0.489 |
| Wilson CI | [0.436, 0.574] | [0.350, 0.630] |
| Promote decision | REJECT | REJECT |
| Arena sec/game | 2.72 | 2.58 |
| Parity (500 positions) | — | **500/500 PASS**, bit-exact |
| Peak `gofer` RSS | not logged | **310 MB** |

Instance RAM at validation: **1910 MiB total**, ~1518 MiB available — no swap. Peak RSS ~16% of total; in-process is viable on the **actual 2 GiB box**, not the previously assumed 4 GiB plan.

## Consequences

- Production builds must use `CGO_ENABLED=1 go build -tags=onnx` (`make build-onnx`).
- ORT shared library **1.26.0** downloaded to `.tectonix/artifacts/` on Linux amd64 (no bundled `.so` in `onnxruntime_go`).
- Parity preflight on Lightsail needs Python **≥3.11** (`.venv311`) for `onnxruntime==1.26.0`; training-only in-process cycles skip Python ORT install.
- `ONNXRUNTIME_SHARED_LIBRARY_PATH` must point at `libonnxruntime.so.1.26.0` when not using the train-loop downloader.

## Revisit trigger

- RSS or arena latency regression vs sidecar on the same hardware after a full 200-game cycle.
- ORT version bump requiring re-run of parity harness and pinned `requirements.txt`.
- Migration to GPU inference (may revisit ORT EP config or deployment shape).
