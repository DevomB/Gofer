# Failure Modes

Known limitations and blunder classes for Gofer v2.5.

## Rules / scoring

- Chinese `Score()` uses area flood without Benson dead-stone removal; seki may be mis-scored.
- `OwnershipLabel` uses area flood (not full Benson); training labels may differ from KataGo.
- Simple ko only unless Tromp-Taylor + superko wrapper is selected.

## Search

- Heuristic/uniform evaluators plateau far below pro strength without a trained net.
- Forced root playouts add latency; disable via `ForcedRootPlayouts=0` for latency-sensitive GTP.
- Batched mock eval can fall back to heuristic on queue timeout (strength unchanged).
- Root-parallel MCTS may not speed up at low playout counts (e.g. 200 on 9×9): lock contention and worker startup can match or exceed single-threaded time. Plan target ≥1.5× at 8 cores is **not met** at v2.0 microbench settings; revisit at 800+ playouts / 19×19.

## Inference (v2.0)

- No real ONNX yet; `-eval batched` exercises queue only.
- p99 under load not characterized on reference hardware until v2.5 ONNX experiment.

## Inference (v2.5 ONNX)

- `-eval onnx` requires sidecar (`make sidecar`); falls back to heuristic on timeout or HTTP error.
- Feature schema mismatch between Go and exported ONNX fails at sidecar startup or returns HTTP 400.
- Sidecar down: all positions evaluate as heuristic (silent strength drop unless stats inspected).
- Batch queue timeout under load increases fallback rate; tune `-batch-size` and `-eval-timeout`.
- Self-play policy export uses board-indexed visit distribution (`RootPolicyBoard`) for training alignment.

## Arena / statistics

- Short match counts (e.g. CI smoke 20 games) are not strength claims; use ≥200 games for gating.
- Draws are rare with area scoring but possible; Wilson CI includes all games in denominator.

## Operational

- `make pgo-profile` uses `BenchmarkLegalMoves` microbench; may mis-optimize search paths.
- `make bench-check` uses max-of-3 bench samples; on Windows, thermal/noise can exceed the 10% gate for search and I/O-heavy benches even when code is unchanged. CI (Linux) is the authoritative regression gate.
