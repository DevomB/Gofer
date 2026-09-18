"""Gating statistics: SPRT on Elo, Wilson intervals, Elo estimates.

Fixed 200-game arenas spend most of their games on candidates that are clearly
better or clearly worse. A sequential probability ratio test (as used by Leela
Zero / Fishtest) stops as soon as the evidence is decisive, typically cutting
gating cost 2-4x while keeping the configured error rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ACCEPT = "accept"
REJECT = "reject"
CONTINUE = "continue"


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """(lower, upper) log-likelihood-ratio bounds: <=lower rejects H1, >=upper accepts."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def sprt_llr(wins: int, losses: int, draws: int, elo0: float, elo1: float) -> float:
    """Bernoulli LLR of H1 (elo1) vs H0 (elo0); draws count as half a win and half a loss."""
    w = wins + 0.5 * draws
    l = losses + 0.5 * draws
    p0, p1 = elo_to_score(elo0), elo_to_score(elo1)
    return w * math.log(p1 / p0) + l * math.log((1.0 - p1) / (1.0 - p0))


def sprt_decision(wins: int, losses: int, draws: int, *, elo0: float, elo1: float, alpha: float, beta: float) -> tuple[str, float]:
    llr = sprt_llr(wins, losses, draws, elo0, elo1)
    lower, upper = sprt_bounds(alpha, beta)
    if llr >= upper:
        return ACCEPT, llr
    if llr <= lower:
        return REJECT, llr
    return CONTINUE, llr


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


@dataclass
class MatchTally:
    wins: int = 0      # candidate wins
    losses: int = 0    # candidate losses
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.5

    def add(self, other: "MatchTally") -> "MatchTally":
        return MatchTally(self.wins + other.wins, self.losses + other.losses, self.draws + other.draws)

    def elo(self) -> tuple[float, float, float]:
        """(estimate, low95, high95) Elo of candidate over opponent.

        Scores are kept half a game away from 0 and 1 so a shutout maps to a
        finite gap that grows with the sample (10-0 ~ +500, 40-0 ~ +750)
        instead of an arbitrary clamp.
        """
        if not self.games:
            return 0.0, 0.0, 0.0
        edge = 0.5 / self.games
        lo, hi = wilson(self.wins + 0.5 * self.draws, self.games)
        clamp = lambda s: min(max(s, edge), 1 - edge)  # noqa: E731
        return score_to_elo(clamp(self.score)), score_to_elo(clamp(lo)), score_to_elo(clamp(hi))

    def to_dict(self) -> dict:
        elo, lo, hi = self.elo()
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "score": self.score,
            "elo": elo,
            "elo_low": lo,
            "elo_high": hi,
        }


def tally_from_arena(report: dict) -> MatchTally:
    """Candidate = challenger (white-eval side of the gofer arena report)."""
    return MatchTally(
        wins=int(report.get("wins_challenger", 0)),
        losses=int(report.get("wins_baseline", 0)),
        draws=int(report.get("draws", 0)),
    )


def expected_games(elo0: float, elo1: float, alpha: float, beta: float, true_elo: float) -> float:
    """Wald's approximation of the average SPRT sample size at a true Elo gap (for planning)."""
    p0, p1, p = elo_to_score(elo0), elo_to_score(elo1), elo_to_score(true_elo)
    a, b = sprt_bounds(alpha, beta)
    step_w, step_l = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    drift = p * step_w + (1 - p) * step_l
    if abs(drift) < 1e-9:
        return -a * b / (step_w * step_w * p * (1 - p))
    # Probability of ending at the upper bound under the true p (Wald's OC).
    h = _oc_exponent(p, p0, p1)
    if abs(h) < 1e-9:
        pa = -a / (b - a)
    else:
        pa = (1 - math.exp(h * a)) / (math.exp(h * b) - math.exp(h * a))
    return (pa * b + (1 - pa) * a) / drift


def _oc_exponent(p: float, p0: float, p1: float) -> float:
    """Solve p*(p1/p0)^h + (1-p)*((1-p1)/(1-p0))^h = 1 for h != 0 by bisection."""
    rw, rl = p1 / p0, (1 - p1) / (1 - p0)

    def f(h: float) -> float:
        return p * rw**h + (1 - p) * rl**h - 1

    lo, hi = (-50.0, -1e-6) if p * math.log(rw) + (1 - p) * math.log(rl) > 0 else (1e-6, 50.0)
    for _ in range(200):
        mid = (lo + hi) / 2
        if (f(lo) < 0) == (f(mid) < 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
