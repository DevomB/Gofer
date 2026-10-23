# Known issues

Open problems documented here so they survive chat history.

## Chinese scoring: dead-stone / Benson pass-alive (open)

**Status:** Open ceiling, not a color-bias bug.

`chineseRules.Score` and `OwnershipLabel` use area flood-fill without Benson dead-stone removal. Surrounded dead stones still on the board count for their owner. Tournament Chinese rules often remove dead stones in a two-pass phase first.

**Upgrade path:** Benson pass-alive marking before territory flood (see `docs/failure-modes.md`).

## Arena stone-color vs role wins (resolved — not a scoring bug)

**Status:** Resolved Jul 2026.

**Finding:** Chinese area scoring in `cmd/gofer/chinese_rules.go` is symmetric (mirror tests, conservation invariant, indexing symmetry). Systematic **stone-color** skew at komi 6.5 on 9×9 is expected: first-move advantage plus komi favors White in equal-strength play.

**Arena gating uses role wins** (challenger vs baseline) with `SwapColors=true`. Equal-strength nets (`heuristic` vs `heuristic2`) show ~50/50 challenger win rate at komi 6.5; stone-color split stays White-heavy. Cycle 24's 85% challenger rate reflects model strength, not a scoring defect.

**Removed:** `normalizeArenaKomi` / `komi9x9Arena` arena-only komi remap (did not fix role gating; masked diagnosis).

**Unified komi:** `6.5` for self-play and arena (`DefaultSelfplayConfig`, CLI default).

## Production hardware assumption

Lightsail training box is **`t3.small` (~2 GiB RAM)**, not 4 GiB. See ADR 0004. A 2 GiB swapfile is configured on the instance.

## Champion ONNX archive (from cycle 25+)

On promotion, `train-loop-v3.sh` copies the current `gofer-9x9-best.onnx` to `models/archive/pre-promote-cycle-N.onnx` before overwriting. Cycle 24 and earlier promotions have no archived ONNX.
