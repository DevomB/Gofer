# Known issues

Open problems and accepted limits, written down so they survive chat history.

## Chinese scoring: dead-stone / Benson pass-alive (open)

**Status:** Open ceiling, not a color-bias bug.

`chineseRules.Score` and `OwnershipLabel` use area flood-fill without Benson dead-stone
removal. Surrounded dead stones still on the board count for their owner, and seki can be
mis-scored. Tournament Chinese rules remove dead stones in a two-pass phase first, so our
training ownership labels can differ from KataGo's on the same position.

**Upgrade path:** Benson pass-alive marking before the territory flood.

## Arena stone-color vs role wins (superseded Sep 2026)

**Status:** The Jul 2026 entry here concluded that the White-heavy stone-color split was
expected from komi and that role gating was unaffected. That conclusion was wrong, and it
is worth keeping visible because it is how the underlying defects survived: a real symptom
was explained away as a known property of the game.

What was actually happening ([ADR 0008](decisions/0008-search-correctness.md)):

- The search never left the root on 9x9, so games were decided near-empty and komi took
  them. The 9-of-188 stone-color split was that, not first-move advantage.
- Role attribution was *not* balanced. Per-game seeds were linear in the game index while
  colours swapped on the same parity, so identical evaluators split 126-74 by role. The
  earlier "~50/50 challenger win rate" held only because colour alternation cancelled it
  in the particular runs that were looked at.
- Every strength number quoted in the old entry, including the 85% challenger rate for
  cycle 24, is void.

**Now:** with the search repaired and seeds mixed, identical evaluators at fair komi split
99-101 by colour over 200 games. Stone-color skew at tournament komi 6.5 is still expected
(Black wins about 29% at 50 playouts) because 6.5 is not fair for engines this weak; fair
komi is near 0.5 and `TestIdenticalEvalColorBalance` asserts balance there, while
`TestArenaIdenticalNetsNoSystematicRoleBias` asserts role balance at 6.5.

## Strength limits

- Heuristic and uniform evaluators plateau far below pro strength; only a trained net moves
  that ceiling.
- Simple ko only, unless the Tromp-Taylor superko wrapper is selected.
- Root-parallel MCTS does not reliably speed up at low playout counts (200 on 9x9): lock
  contention and worker startup can match single-threaded time. Revisit at 800+ playouts.

## Inference fallbacks are silent

`-eval onnx` falls back to the heuristic on sidecar timeout or HTTP error, and the batched
evaluator falls back on queue timeout. Strength drops with no error surfaced, so read the
eval stats rather than assuming the net was used. A feature-schema mismatch between Go and
the exported ONNX fails loudly instead, at sidecar startup or with HTTP 400.

## Measurement

- Short matches are not strength claims. The CI smoke arena runs 20 games; gate on 200.
- `make bench-check` compares max-of-3 samples. On Windows, thermal noise alone can exceed
  the 10% gate for search and I/O-heavy benches. Linux CI is the authoritative gate.
- `make pgo-profile` profiles `BenchmarkLegalMoves`, which is not the search hot path.

## Deferred by choice

Carried forward from the v2 backlogs; none of these are blocked, they are just not worth
the complexity yet.

| Idea | Note |
|---|---|
| LCB move selection at the root | KataGo uses it; our visit counts are small enough that it rarely differs |
| Global pooling inside the search net | Implemented in the learner (`gofer_train/nets.py`), not in the Go heuristic path |
| Dynamic score maximization | Post-paper KataGo; needs a score head we trust first |
| Open-addressing transposition table | Current `map`-backed TT has not shown up in profiles |
| Progressive net scaling | ADR 0005 found no win for 9x9, but on a pre-fix snapshot whose policy targets are void; re-run before relying on it |
| Policy surprise weighting, JSON analysis API | Post-paper, no current consumer |
| Index the root-child lookup in `mcts_policy.go` | `fillLegalPolicy` and `prunePolicy` match children against legal moves by scanning, so each is O(children x legal) and the second repeats the matching the first already did. That is ~13k comparisons per self-play move on 9x9 against a 16-95ms move, so about 0.02%; `policyIndex` already gives the key an index would use. Worth doing when a profile says so, or when 19x19 becomes real |
