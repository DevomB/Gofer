"""Exact operating characteristics of a promotion gate.

Answers, for a candidate that is truly `elo` stronger than the champion:
how often does the gate promote it, and how many arena games does that cost?

Both are computed exactly, by dynamic programming over (games played, wins) --
no simulation. The model covers what the pipeline actually runs:

  * each SPRT step is one ``gofer -arena -arena-play-all`` call, i.e. a batch of
    a fixed size (the engine's own in-match stop is disabled for the gate, so
    only one stopping rule acts on the stream; `ArenaRule` still models that
    stop, which the v3 gate used and which measurement runs must avoid);
  * the orchestrator's SPRT decision after every batch (training.pipeline.stats);
  * the win-rate + Wilson fallback applied at `max_games`.

On 9x9 with komi 6.5 a game cannot be drawn, so games are Bernoulli trials.
numpy only: scipy is not a dependency of this project.

Used by `python -m training.pipeline plan-sprt` and by paper/analysis/gate_oc.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple

import numpy as np

from training.pipeline import stats
from training.pipeline.config import GatingConfig


class GateOC(NamedTuple):
    accept: float   # P(the gate promotes)
    games: float    # E[arena games played]


@dataclass(frozen=True)
class ArenaRule:
    """The Go arena's in-call stopping rule (cmd/gofer/match.go)."""

    promote_win: float = 0.55        # gating.go PromoteMin
    min_games_before_stop: int = 20
    min_games_before_promote: int = 100

    def verdict(self, wins: int, played: int, max_games: int) -> str | None:
        if played < self.min_games_before_stop or max_games < self.min_games_before_stop:
            return None
        if played >= self.min_games_before_promote and final_rule(wins, played, self.promote_win):
            return "accept"
        if (wins + max_games - played) / max_games < self.promote_win:
            return "reject"
        return None


def final_rule(wins: int, played: int, promote_win: float) -> bool:
    """The v3 promotion rule, also the v4 fallback at the cap."""
    if played <= 0:
        return False
    lo, _ = stats.wilson(wins, played)
    return wins / played >= promote_win and lo > 0.5


def binom_pmf(n: int, p: float) -> np.ndarray:
    """P(k successes in n trials) for k = 0..n, computed in log space."""
    k = np.arange(n + 1)
    if p <= 0.0:
        out = np.zeros(n + 1)
        out[0] = 1.0
        return out
    if p >= 1.0:
        out = np.zeros(n + 1)
        out[n] = 1.0
        return out
    log_c = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    return np.exp(log_c + k * math.log(p) + (n - k) * math.log1p(-p))


@lru_cache(maxsize=256)
def _arena_outcomes(games: int, p: float, rule: ArenaRule) -> tuple[np.ndarray, np.ndarray]:
    """One arena call: P(ends after k games with w wins) and whether it ended by accept.

    Returned arrays are indexed [k, w]; rows for k < games are the early stops.
    """
    dist = np.zeros((games + 1, games + 1))
    accepted = np.zeros((games + 1, games + 1), dtype=bool)
    live = np.zeros(games + 1)
    live[0] = 1.0
    for k in range(1, games + 1):
        nxt = np.zeros(games + 1)
        nxt[1:] += live[:-1] * p
        nxt += live * (1.0 - p)
        live = nxt
        for w in np.nonzero(live)[0]:
            verdict = rule.verdict(int(w), k, games) if k < games else "end"
            if verdict is None:
                continue
            dist[k, w] += live[w]
            accepted[k, w] = verdict == "accept"
            live[w] = 0.0
    return dist, accepted


def fixed_gate_oc(games: int, elo: float, accept: Callable[[int, int], bool]) -> GateOC:
    """A fixed-length test of `games` games with no early stopping."""
    pmf = binom_pmf(games, stats.elo_to_score(elo))
    mask = np.array([accept(w, games) for w in range(games + 1)])
    return GateOC(float(pmf[mask].sum()), float(games))


def v3_gate_oc(elo: float, *, max_games: int = 200, rule: ArenaRule = ArenaRule()) -> GateOC:
    """The v3 gate: one arena call with interim stops, final rule at the end."""
    dist, accepted = _arena_outcomes(max_games, stats.elo_to_score(elo), rule)
    p_accept = games = 0.0
    for k, w in zip(*np.nonzero(dist)):
        mass = dist[k, w]
        games += mass * k
        if accepted[k, w] or (k == max_games and final_rule(int(w), int(k), rule.promote_win)):
            p_accept += mass
    return GateOC(p_accept, games)


def sprt_gate_oc(cfg: GatingConfig, elo: float, *, rule: ArenaRule | None = None) -> GateOC:
    """The v4 gate: SPRT over arena batches, with the v3 rule as fallback at the cap.

    Batches run with ``-arena-play-all``, so each one is the fixed size the test
    assumes; `rule` models the engine's in-match stop for the legacy behaviour,
    where a second stopping rule ran inside every batch.
    """
    p = stats.elo_to_score(elo)
    cap = cfg.max_games
    live = np.zeros((cap + 1, cap + 1))   # live[played, wins], undecided mass
    live[0, 0] = 1.0
    p_accept = games = 0.0
    while live.sum() > 1e-12:
        nxt = np.zeros_like(live)
        for played in np.nonzero(live.sum(axis=1))[0]:
            batch = min(cfg.batch_games, cap - played)
            batch -= batch % 2                       # colours alternate
            if batch <= 0:
                continue
            if rule is None:
                dist = np.zeros((batch + 1, batch + 1))
                dist[batch] = binom_pmf(batch, p)   # the batch always plays out
            else:
                dist, _ = _arena_outcomes(batch, p, rule)
            row = live[played]
            for k in range(batch + 1):
                if not dist[k].any():
                    continue
                spread = np.convolve(row, dist[k])[: cap + 1]
                nxt[played + k, : cap + 1] += spread[: cap + 1]
        live = np.zeros_like(nxt)
        for played, w in zip(*np.nonzero(nxt > 1e-15)):
            mass = nxt[played, w]
            verdict, _ = stats.sprt_decision(int(w), int(played - w), 0, elo0=cfg.elo0, elo1=cfg.elo1,
                                             alpha=cfg.alpha, beta=cfg.beta)
            if verdict == stats.CONTINUE and played < cap and cap - played >= 2:
                live[played, w] = mass
                continue
            games += mass * played
            if verdict == stats.ACCEPT or (verdict == stats.CONTINUE and final_rule(int(w), int(played), cfg.promote_win)):
                p_accept += mass
    return GateOC(p_accept, games)
