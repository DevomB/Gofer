# Known issues

Open problems documented here so they survive chat history. Do not treat workarounds below as resolved root causes unless this file is updated.

## Arena komi vs self-play komi (scoring bias — unresolved)

**Status:** Open. Workaround in production; root cause not found.

**Symptoms:** With equal-strength ONNX nets under Chinese area scoring, arena outcomes skew heavily toward White at the default CLI komi (`6.5` on 9×9). Identical-net control matches showed systematic color imbalance before arena-specific mitigations.

**Workaround (intentional):**

| Mode | Komi | Where set |
|------|------|-----------|
| Self-play | `6.5` | `DefaultSelfplayConfig()` in `cmd/gofer/selfplay.go` |
| Arena (9×9) | `3.5` | `normalizeArenaKomi()` remaps CLI default `6.5` → `komi9x9Arena` in `cmd/gofer/match.go` |

Arena and self-play therefore run at **different komi by design** until the scoring path is understood. This is not a drift bug in the train loop.

**Not in scope (yet):** Debugging Tromp–Taylor vs Chinese scoring, first-move advantage, or ownership labels. Phase 2+ may revisit with controlled identical-net experiments and SGF replay.

**Related:** ADR [0004](./decisions/0004-in-process-onnx-inference.md) (hardware note); cycle 24 promotion showed strong White skew — see training logs / `arena-cycle-24.json`.

## Production hardware assumption

Lightsail training box is **`t3.small` (~2 GiB RAM)**, not 4 GiB. See ADR 0004 and `docs/decisions/0004-in-process-onnx-inference.md`. A 2 GiB swapfile is configured on the instance as a safety margin.
