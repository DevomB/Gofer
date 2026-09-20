"""Gate operating-characteristic table and curves for the paper.

Thin wrapper over training.pipeline.gate_oc, which is the model the pipeline
itself uses (`python -m training.pipeline plan-sprt`), so the paper cannot drift
from the shipped code. For a candidate that is truly `elo` stronger than the
champion we report, exactly:

  * P(accept), the probability the gate promotes it
  * E[games], the expected number of arena games

Gates compared:
  katago      200 games, promote on >= 100 wins (Wu 2020, App. E)
  gofer-v3    one 200-game arena with the Go arena's interim stops, then the
              win-rate + Wilson rule (the v3 loop; see ADR 0003)
  v4-default  the shipped defaults: SPRT(0, 35), alpha 0.05, beta 0.10,
              batches of 40, cap 600, v3 rule as fallback at the cap
  v4-actions  the free-runner budget: SPRT(0, 50), batches of 24, cap 240

Usage (repo root):  python paper/analysis/gate_oc.py
Writes paper/figures/gate_oc.csv and paper/figures/gate_table.tex.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.pipeline.config import GatingConfig, load_config  # noqa: E402
from training.pipeline.gate_oc import final_rule, fixed_gate_oc, sprt_gate_oc, v3_gate_oc  # noqa: E402

OUT = ROOT / "paper" / "figures"


def _cfg(path: str) -> GatingConfig:
    return load_config(ROOT / "configs" / path).gating


GATES = {
    "katago": lambda elo: fixed_gate_oc(200, elo, lambda w, n: w >= 100),
    "gofer-v3": v3_gate_oc,
    "v4-default": lambda elo: sprt_gate_oc(GatingConfig(), elo),
    "v4-actions": lambda elo: sprt_gate_oc(_cfg("pipeline-actions.toml"), elo),
}
TABLE_ELOS = [-100, -50, -20, 0, 20, 35, 50, 100, 200]
PLOT_ELOS = list(range(-150, 251, 10))
COLUMNS = ["elo", "katagoAcc", "katagoN", "vthreeAcc", "vthreeN", "vfourAcc", "vfourN", "actionsAcc", "actionsN"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {elo: {name: gate(float(elo)) for name, gate in GATES.items()}
               for elo in sorted(set(PLOT_ELOS) | set(TABLE_ELOS))}

    with (OUT / "gate_oc.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(COLUMNS)   # plain identifiers: pgfplots column names in section 06
        for elo in PLOT_ELOS:
            wr.writerow([elo] + [f"{v:.6f}" for name in GATES for v in results[elo][name]])

    single = fixed_gate_oc(200, 0.0, lambda w, n: final_rule(w, n, 0.55))
    lines = [
        r"\begin{tabular}{r rr rr rr rr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\katago{} 100/200} & \multicolumn{2}{c}{\gofer{} v3} & \multicolumn{2}{c}{v4 SPRT (default)} & \multicolumn{2}{c}{v4 SPRT (Actions)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"True $\Delta$Elo & $P_{\mathrm{acc}}$ & $E[n]$ & $P_{\mathrm{acc}}$ & $E[n]$ & $P_{\mathrm{acc}}$ & $E[n]$ & $P_{\mathrm{acc}}$ & $E[n]$ \\",
        r"\midrule",
    ]
    for e in TABLE_ELOS:
        cells = " & ".join(f"{a:.3f} & {g:.0f}" for a, g in results[e].values())
        lines.append(f"${e:+d}$ & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "gate_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for e in TABLE_ELOS:
        print(f"{e:+5d} " + "  ".join(f"{n}: acc={a:.3f} n={g:.0f}" for n, (a, g) in results[e].items()))
    print(f"\nreference: one 200-game test with the v3 rule and no interim looks "
          f"accepts {single.accept:.3f} at zero true gain")


if __name__ == "__main__":
    main()
