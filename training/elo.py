"""Bradley-Terry Elo ratings from pairwise match results."""

from __future__ import annotations

import math
from collections import defaultdict


def fit(results: list[dict], anchor: str, prior_games: float = 1.0, iters: int = 2000) -> dict[str, dict]:
    """Fit ratings from results [{"a", "b", "wins", "losses", "draws"}].

    Draws count as half a win each way. `prior_games` adds that many virtual
    drawn games between every pair that met, which keeps perfect scores finite.
    Returns {player: {"elo": r, "stderr": s, "games": n}} with `anchor` at 0.
    """
    wins: dict[tuple[str, str], float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    players: set[str] = set()
    for r in results:
        a, b = r["a"], r["b"]
        players |= {a, b}
        n = r["wins"] + r["losses"] + r["draws"]
        wa = r["wins"] + 0.5 * r["draws"]
        wins[(a, b)] += wa + prior_games / 2
        wins[(b, a)] += n - wa + prior_games / 2
        games[(a, b)] += n + prior_games
        games[(b, a)] += n + prior_games
    if anchor not in players:
        anchor = sorted(players)[0] if players else anchor
    gamma = {p: 1.0 for p in players}
    for _ in range(iters):
        delta = 0.0
        for p in players:
            w = sum(wins[(p, q)] for q in players if (p, q) in wins)
            denom = sum(games[(p, q)] / (gamma[p] + gamma[q]) for q in players if (p, q) in games)
            if denom > 0 and w > 0:
                new = w / denom
                delta = max(delta, abs(math.log(new / gamma[p])))
                gamma[p] = new
        if delta < 1e-9:
            break
    ref = gamma.get(anchor, 1.0)
    out = {}
    for p in players:
        # Fisher information of log-strength gives an approximate standard error.
        info = sum(games[(p, q)] * gamma[p] * gamma[q] / (gamma[p] + gamma[q]) ** 2 for q in players if (p, q) in games)
        se = 400 / math.log(10) / math.sqrt(info) if info > 0 else float("inf")
        n = sum(games[(p, q)] for q in players if (p, q) in games) - prior_games * sum(1 for q in players if (p, q) in games)
        out[p] = {"elo": 400 * math.log10(gamma[p] / ref), "stderr": 0.0 if p == anchor else se, "games": int(round(n))}
    return out
