"""Tests for the exact gate operating-characteristic model."""

from __future__ import annotations

import math
import random

import pytest

from training.pipeline import stats
from training.pipeline.config import GatingConfig
from training.pipeline.gate_oc import (
    ArenaRule,
    binom_pmf,
    final_rule,
    fixed_gate_oc,
    sprt_gate_oc,
    v3_gate_oc,
)


def test_binom_pmf_matches_closed_form():
    pmf = binom_pmf(10, 0.3)
    assert math.isclose(pmf.sum(), 1.0, rel_tol=1e-12)
    assert math.isclose(pmf[3], math.comb(10, 3) * 0.3**3 * 0.7**7, rel_tol=1e-12)
    assert binom_pmf(5, 0.0)[0] == 1.0 and binom_pmf(5, 1.0)[5] == 1.0


def test_fixed_gate_is_binomial_tail():
    # KataGo's gate at zero true gain: P(at least 100 wins in 200 fair games).
    oc = fixed_gate_oc(200, 0.0, lambda w, n: w >= 100)
    assert 0.52 < oc.accept < 0.54 and oc.games == 200


def test_v3_gate_inflates_false_promotions():
    """The arena's per-game re-checks promote ~8% of zero-gain candidates."""
    zero = v3_gate_oc(0.0)
    assert 0.07 < zero.accept < 0.09
    assert zero.games < 200  # interim stops end most gates early
    # A single test at the same bound, without interim looks, is far stricter.
    single = fixed_gate_oc(200, 0.0, lambda w, n: final_rule(w, n, 0.55))
    assert single.accept < zero.accept / 2


def test_sprt_controls_false_promotions_and_is_monotone():
    cfg = GatingConfig(batch_games=40, max_games=400, elo0=0, elo1=35, alpha=0.05, beta=0.05)
    curve = [sprt_gate_oc(cfg, elo) for elo in (-100, -50, 0, 35, 50, 100)]
    accepts = [c.accept for c in curve]
    assert accepts == sorted(accepts)                 # stronger candidates promoted more often
    assert accepts[2] < 0.05                          # false promotion at no real gain
    assert accepts[2] < v3_gate_oc(0.0).accept        # better than v3
    assert accepts[-1] > 0.95                         # clear gains are promoted
    assert all(0 <= c.games <= cfg.max_games for c in curve)


def test_clear_results_stop_early():
    cfg = GatingConfig(batch_games=40, max_games=400, elo0=0, elo1=35)
    assert sprt_gate_oc(cfg, -200).games < 120        # hopeless candidate rejected fast
    assert sprt_gate_oc(cfg, 200).games < 120         # dominant candidate accepted fast
    assert sprt_gate_oc(cfg, 0).games > 200           # near the indifference zone it pays more


def _simulate(cfg: GatingConfig, elo: float, trials: int, seed: int) -> tuple[float, float]:
    """Independent per-game Monte Carlo of the same procedure, for cross-checking."""
    rng = random.Random(seed)
    rule = ArenaRule(promote_win=cfg.promote_win)
    p = stats.elo_to_score(elo)
    accepts = total = 0
    for _ in range(trials):
        wins = played = 0
        verdict = stats.CONTINUE
        while played < cfg.max_games:
            batch = min(cfg.batch_games, cfg.max_games - played)
            batch -= batch % 2
            if batch <= 0:
                break
            bw = 0
            for k in range(1, batch + 1):
                bw += rng.random() < p
                if k < batch and rule.verdict(bw, k, batch) is not None:
                    break
            wins += bw
            played += k
            verdict, _ = stats.sprt_decision(wins, played - wins, 0, elo0=cfg.elo0, elo1=cfg.elo1,
                                             alpha=cfg.alpha, beta=cfg.beta)
            if verdict != stats.CONTINUE:
                break
        total += played
        if verdict == stats.ACCEPT or (verdict == stats.CONTINUE and final_rule(wins, played, cfg.promote_win)):
            accepts += 1
    return accepts / trials, total / trials


@pytest.mark.parametrize("elo", [0.0, 60.0])
def test_exact_matches_simulation(elo):
    cfg = GatingConfig(batch_games=20, max_games=60, elo0=0, elo1=35, alpha=0.05, beta=0.05)
    exact = sprt_gate_oc(cfg, elo)
    acc, games = _simulate(cfg, elo, trials=4000, seed=11)
    # 4000 trials: 3 sigma is at most 0.024 on a probability.
    assert abs(exact.accept - acc) < 0.03
    assert abs(exact.games - games) < 3.0
