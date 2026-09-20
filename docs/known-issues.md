# Known issues

Open problems documented here so they survive chat history.

## Chinese scoring: dead-stone / Benson pass-alive (open)

**Status:** Open ceiling, not a color-bias bug.

`chineseRules.Score` and `OwnershipLabel` use area flood-fill without Benson dead-stone removal. Surrounded dead stones still on the board count for their owner. Tournament Chinese rules often remove dead stones in a two-pass phase first.

**Upgrade path:** Benson pass-alive marking before territory flood (see `docs/failure-modes.md`).

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

## Production hardware assumption

Lightsail training box is **`t3.small` (~2 GiB RAM)**, not 4 GiB. See ADR 0004. A 2 GiB swapfile is configured on the instance.

## Champion ONNX archive (from cycle 25+)

On promotion, `train-loop-v3.sh` copies the current `gofer-9x9-best.onnx` to `models/archive/pre-promote-cycle-N.onnx` before overwriting. Cycle 24 and earlier promotions have no archived ONNX.
