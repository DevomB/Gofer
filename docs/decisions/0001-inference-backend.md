# Inference backend

## Context

Gofer v2 needs a batched evaluation path for MCTS without blocking playout loops. Real GPU inference arrives in v2.5.

## Options

| Option | Pros | Cons |
|--------|------|------|
| ONNX Runtime cgo | Single binary, low latency | CGO build complexity, GPU drivers |
| Python sidecar (gRPC/HTTP) | Fast iteration, PyTorch export | Process boundary, ops overhead |
| Mock batched queue | Unblocks search integration | No strength gain |

## Decision

**v2.0:** `BatchedEvaluator` with mock/heuristic fallback, `minBatch=8`, `maxWait=2ms`.

**v2.5:** Revisit with latency experiment (p50/p99 ms/move @ 1600 playouts 19×19). Prefer Python ONNX sidecar first for bootstrap, then cgo if SLO requires.

## Why

Search batching and backpressure are orthogonal to model quality. Ship the queue contract now; swap `EvalBackend` implementation later.

## Performance impact

Target SLO (reference hardware): GTP `genmove` p99 < 5s @ 1600 playouts @ 19×19.

## Revisit trigger

- v2.5 ONNX bootstrap
- p99 regression > 10% after backend swap
- Batch size sweep shows GPU underutilization
