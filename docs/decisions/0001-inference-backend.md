# Inference backend

## Context

Gofer v2 needs a batched evaluation path for MCTS without blocking playout loops. Real GPU inference ships in v2.5.

## Options

| Option | Pros | Cons |
|--------|------|------|
| ONNX Runtime cgo | Single binary, low latency | CGO build complexity, GPU drivers |
| Python sidecar (HTTP) | Fast iteration, PyTorch export | Process boundary, ops overhead |
| Mock batched queue | Unblocks search integration | No strength gain |

## Decision

**v2.0:** `BatchedEvaluator` with mock/heuristic fallback, `minBatch=8`, `maxWait=2ms`.

**v2.5:** Python ONNX Runtime HTTP sidecar (`training/inference_server.py`). Go client: `SidecarBackend` in `cmd/gofer/onnx_sidecar.go`, wrapped by `BatchedEvaluator` with `Heuristic{}` fallback.

CGO deferred unless sidecar p99 regresses >10% vs SLO on reference hardware.

## HTTP protocol

`POST /v1/eval` — JSON body:

```json
{
  "schema_version": 2,
  "batch_size": 8,
  "spatial": [[...]],
  "globals": [[...]]
}
```

- `spatial`: `batch_size × (planes × H × W)` float32, row-major NCHW per board (8 planes @ 9×9 for bootstrap net)
- `globals`: `batch_size × 4` float32 (komi/10, move_num norm, black-to-move, white-to-move)

Response:

```json
{
  "results": [{"value": 0.12, "policy": [82 floats]}]
}
```

- `value`: tanh-bounded, from current player perspective
- `policy`: softmax probabilities, length `H×W+1` (pass last)

`GET /health` — model path, schema version, input shapes.

## Batching and fallback

| Parameter | Default | Notes |
|-----------|---------|-------|
| `minBatch` | 8 | `-batch-size` flag |
| `maxWait` | 2ms | gather window |
| `reqTimeout` | 500ms | `-eval-timeout`; queue wait + HTTP round-trip |

On HTTP error, shape mismatch, or timeout: `Heuristic{}` result for that position (search continues; strength drops).

## Latency (reference hardware)

Measured on: 11th Gen Intel Core i7-1185G7 @ 3.00GHz, Windows 11, sidecar on localhost.

| Scenario | p50 ms/move | p99 ms/move | Notes |
|----------|-------------|-------------|-------|
| 9×9 @ 400 playouts, `-eval onnx` | 1485 | 2286 | sidecar localhost, i7-1185G7, n=50 |
| 19×19 @ 1600 playouts, `-eval heuristic` | ~12000 | ~12000 | single `-analyze` run, i7-1185G7 |
| 19×19 @ 1600 playouts, `-eval onnx` | deferred | deferred | bootstrap net is 9×9-only |

Target SLO: GTP `genmove` p99 < 5s @ 1600 playouts @ 19×19 with ONNX sidecar on reference hardware.

## Revisit trigger

- Sidecar p99 regression > 10% after optimization pass
- Batch size sweep shows GPU underutilization with local CUDA
- Single-binary deployment requirement (→ ONNX Runtime cgo spike)
