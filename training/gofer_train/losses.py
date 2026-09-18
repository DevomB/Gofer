"""Multi-head loss with per-row masks (KataGo-style playout-cap handling)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import Batch
from .nets import SCORE_SCALE, NetOutputs


@dataclass
class LossWeights:
    policy: float = 1.0
    value: float = 1.5
    ownership: float = 0.15
    score: float = 0.05
    policy_opp: float = 0.15
    distill: float = 0.0  # >0 with a teacher net


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row cross-entropy against a probability target."""
    return -(target * F.log_softmax(logits.float(), dim=1)).sum(dim=1)


def compute_loss(
    out: NetOutputs,
    batch: Batch,
    w: LossWeights,
    teacher: NetOutputs | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Returns (total, parts). Parts are detached scalars for logging.

    * policy: only rows searched at full playout cap (fast-cap visit counts are
      too noisy to imitate) and with a non-empty target.
    * policy_opp: full-cap rows whose opponent reply is known.
    * value / ownership: every row — the game outcome is equally valid for
      fast-cap moves, which is where most of the data comes from.
    * score: rows with a finite margin.
    """
    parts: dict[str, torch.Tensor] = {}
    has_pol = batch.full_search & (batch.policy.sum(1) > 0)
    ce = soft_ce(out.policy_logits, batch.policy)
    parts["policy"] = _masked_mean(ce, has_pol)
    parts["value"] = F.mse_loss(out.value.float(), batch.value)
    parts["ownership"] = F.mse_loss(out.ownership.float(), batch.ownership)
    total = w.policy * parts["policy"] + w.value * parts["value"] + w.ownership * parts["ownership"]

    if out.policy_opp_logits is not None and w.policy_opp > 0:
        has_opp = batch.full_search & (batch.policy_opp.sum(1) > 0)
        parts["policy_opp"] = _masked_mean(soft_ce(out.policy_opp_logits, batch.policy_opp), has_opp)
        total = total + w.policy_opp * parts["policy_opp"]

    if out.score is not None and w.score > 0:
        finite = torch.isfinite(batch.score)
        target = torch.where(finite, batch.score, torch.zeros_like(batch.score)) / SCORE_SCALE
        huber = F.smooth_l1_loss(out.score.float(), target, reduction="none")
        parts["score"] = _masked_mean(huber, finite)
        total = total + w.score * parts["score"]

    if teacher is not None and w.distill > 0:
        t_probs = F.softmax(teacher.policy_logits.float(), dim=1)
        parts["distill"] = soft_ce(out.policy_logits, t_probs).mean() + F.mse_loss(
            out.value.float(), teacher.value.float()
        )
        total = total + w.distill * parts["distill"]

    with torch.no_grad():
        pred = out.policy_logits.argmax(1)
        tgt = batch.policy.argmax(1)
        parts["policy_acc"] = _masked_mean((pred == tgt).float(), has_pol)
        parts["value_sign_acc"] = ((out.value.float() * batch.value) > 0).float()[batch.value != 0].mean() \
            if bool((batch.value != 0).any()) else torch.zeros((), device=batch.value.device)
    parts["total"] = total
    return total, {k: v.detach() for k, v in parts.items()}
