# LegalMoves allocation reduction

## Context

`BenchmarkLegalMoves` on 19×19 was ~1158 allocs/op. Root causes: per-candidate board `Clone`, `map[int]struct{}` in group flood-fill, and `Neighbors()` allocating a slice per call.

## Options

1. Incremental union-find groups on `Board` (largest win, highest risk)
2. Reuse trial board + generation-based visit marks + non-allocating neighbor iteration (chosen)
3. Bitboards for legality (defer)

## Decision

Reuse trial board + generation-based visit marks + non-allocating neighbor iteration (`legalityScratch`, `forEachNeighbor`, `groupBuf`/`stackBuf`).

## Why

Measured 1158 → 7 allocs/op without incremental groups. Minimal change; tests unchanged.

## Performance impact

| Benchmark | Before | After |
|-----------|--------|-------|
| `BenchmarkLegalMoves` allocs/op | ~1158 | **7** |
| `BenchmarkLegalMoves` ns/op | ~210k | ~145k |

## Revisit trigger

- If `Play`/`wouldBeLegal` paths dominate profiles, add incremental groups
- If concurrent `LegalMoves` needs scratch pools, use `sync.Pool` on `legalityScratch`
