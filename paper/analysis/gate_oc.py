"""Exact operating characteristics of promotion gates (paper Secs. 6 and 8).

A candidate with true advantage `elo` over the champion wins each game with
p = 1 / (1 + 10^(-elo/400)); 9x9 with komi 6.5 cannot draw. For each gate we
compute exactly, by dynamic programming (no Monte Carlo):

  * P(accept)  probability the gate promotes the candidate
  * E[games]   expected number of arena games played

Gates:
  katago      200 games, promote if wins >= 100 (Wu 2020, App. E)
  gofer-v3    one `gofer -arena -games 200` call. The Go arena stops early
              (cmd/gofer/match.go: promotionGateDecided): after >= 20 games it
              rejects once 55% is unreachable, and after >= 100 games it accepts
              once p_hat >= 0.55 and Wilson low > 0.5. At 200 games the v3 loop
              applies p_hat >= 0.55 and Wilson low > 0.5 (training/cycle.py).
  v4-default  orchestrator SPRT(elo0=0, elo1=35, alpha=beta=0.05) over arena
              batches of 40, cap 400, v3 rule as fallback at the cap. Each batch
              is itself a Go arena call, so the same in-batch early reject applies.
  v4-actions  same, elo1=40, batches of 24, cap 192 (configs/pipeline-actions.toml)

SPRT decisions use the orchestrator's own code (training.pipeline.stats), and the
in-arena rule mirrors the Go implementation, so the numbers describe what runs.
Modeling assumption: games finish in order (the real arena plays games in
parallel and stops at the first decided state among completed games).

Usage (repo root):  python paper/analysis/gate_oc.py
Writes paper/figures/gate_oc.csv and paper/figures/gate_table.tex.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.pipeline import stats  # noqa: E402

OUT = ROOT / "paper" / "figures"
PROMOTE_WIN = 0.55          # cmd/gofer/gating.go PromoteMin, gating.env PROMOTE_WIN
MIN_GAMES_BEFORE_STOP = 20  # cmd/gofer/match.go
MIN_GAMES_BEFORE_PROMOTE = 100


def p_of(elo: float) -> float:
    return stats.elo_to_score(elo)


def v3_final(w: int, n: int) -> bool:
    lo, _ = stats.wilson(w, n)
    return n > 0 and w / n >= PROMOTE_WIN and lo > 0.5


def arena_early(w: int, played: int, max_games: int) -> str | None:
    """Mirror of Go promotionGateDecided: 'accept', 'reject' or None."""
    if played < MIN_GAMES_BEFORE_STOP or max_games < MIN_GAMES_BEFORE_STOP:
        return None
    lo, _ = stats.wilson(w, played)
    if played >= MIN_GAMES_BEFORE_PROMOTE and w / played >= PROMOTE_WIN and lo > 0.5:
        return "accept"
    if (w + max_games - played) / max_games < PROMOTE_WIN:
        return "reject"
    return None


@lru_cache(maxsize=None)
def arena_outcomes(g: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Distribution of one `gofer -arena -games g` call.

    Returns (dist[k, w], early_accept[k, w]): probability the call ends after k
    games with w challenger wins, and whether it ended by the in-arena accept.
    """
    dist = np.zeros((g + 1, g + 1))
    acc = np.zeros((g + 1, g + 1), dtype=bool)
    live = np.zeros(g + 1)
    live[0] = 1.0
    for k in range(1, g + 1):
        nxt = np.zeros(g + 1)
        nxt[1:] += live[:-1] * p
        nxt += live * (1 - p)
        live = nxt
        for w in range(k + 1):
            if live[w] == 0:
                continue
            verdict = arena_early(w, k, g) if k < g else "end"
            if verdict is None:
                continue
            dist[k, w] += live[w]
            acc[k, w] = verdict == "accept"
            live[w] = 0.0
    return dist, acc


def fixed_katago(elo: float) -> tuple[float, float]:
    from scipy.stats import binom

    return float(binom.sf(99, 200, p_of(elo))), 200.0


def gofer_v3(elo: float) -> tuple[float, float]:
    dist, acc = arena_outcomes(200, p_of(elo))
    p_accept = exp_games = 0.0
    for k, w in zip(*np.nonzero(dist)):
        m = dist[k, w]
        exp_games += m * k
        if acc[k, w] or (k == 200 and v3_final(int(w), int(k))):
            p_accept += m
    return p_accept, exp_games


@dataclass(frozen=True)
class Sprt:
    elo0: float
    elo1: float
    alpha: float
    beta: float
    batch: int
    max_games: int

    def __call__(self, elo: float) -> tuple[float, float]:
        p = p_of(elo)
        n = self.max_games
        live = np.zeros((n + 1, n + 1))   # live[played, wins]
        live[0, 0] = 1.0
        p_accept = exp_games = 0.0
        while live.sum() > 1e-12:
            nxt = np.zeros_like(live)
            for played in np.nonzero(live.sum(axis=1))[0]:
                g = min(self.batch, n - played)
                g -= g % 2
                row = np.zeros((n + 1, n + 1))
                row[played] = live[played]
                dist, _ = arena_outcomes(g, p)   # in-batch accept impossible: g < 100
                nxt += fftconvolve(row, dist)[: n + 1, : n + 1].clip(min=0)
            live = np.zeros_like(nxt)
            for played, w in zip(*np.nonzero(nxt > 1e-15)):
                m = nxt[played, w]
                verdict, _ = stats.sprt_decision(int(w), int(played - w), 0, elo0=self.elo0, elo1=self.elo1,
                                                 alpha=self.alpha, beta=self.beta)
                capped = played >= n or n - played < 2
                if verdict == stats.CONTINUE and not capped:
                    live[played, w] = m
                    continue
                exp_games += m * played
                if verdict == stats.ACCEPT or (verdict == stats.CONTINUE and v3_final(int(w), int(played))):
                    p_accept += m
        return p_accept, exp_games


GATES = {
    "katago": fixed_katago,
    "gofer-v3": gofer_v3,
    "v4-default": Sprt(0, 35, 0.05, 0.05, 40, 400),
    "v4-actions": Sprt(0, 40, 0.05, 0.05, 24, 192),
}
TABLE_ELOS = [-100, -50, -20, 0, 20, 35, 50, 100, 200]
PLOT_ELOS = list(range(-150, 251, 10))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, tuple[float, float]]] = {}
    for elo in sorted(set(PLOT_ELOS) | set(TABLE_ELOS)):
        results[elo] = {name: gate(elo) for name, gate in GATES.items()}

    with (OUT / "gate_oc.csv").open("w", newline="") as f:
        # Plain identifiers: pgfplots column names in paper/sections/06-orchestration.tex.
        cols = ["elo", "katagoAcc", "katagoN", "vthreeAcc", "vthreeN", "vfourAcc", "vfourN", "actionsAcc", "actionsN"]
        wr = csv.writer(f)
        wr.writerow(cols)
        for elo in PLOT_ELOS:
            wr.writerow([elo] + [f"{v:.6f}" for n in GATES for v in results[elo][n]])

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


if __name__ == "__main__":
    main()
