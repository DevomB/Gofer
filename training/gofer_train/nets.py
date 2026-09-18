"""Network architectures for the Gofer learner.

Two families live behind one interface (``build_net`` / ``NetOutputs``):

* ``legacy`` — the exact ``GoferBootstrapNet`` layout that produced the current
  champion (flatten -> FC heads). Kept bit-compatible so ``--resume`` from
  ``training/state/best.pt`` keeps working.
* ``gpool`` — a KataGo-style residual trunk: some blocks carry a global-pooling
  bias (mean + max of a channel subset feeds back as a per-channel bias), and
  every head is fully convolutional + pooled. No layer depends on H*W, so the
  parameter count is independent of board size and a single net can train on
  mixed 9x9 / 13x13 / 19x19 data. It also adds a score-margin head and an
  opponent-reply policy head (both auxiliary; stripped from inference ONNX).

Every net's ``forward`` returns ``NetOutputs``; the legacy module keeps its
original 3-tuple ``forward`` for existing callers and exposes ``forward_all``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_PLANES = 8
GLOBALS = 4
# Score head predicts margin / SCORE_SCALE so targets sit roughly in [-3, 3] on 9x9.
SCORE_SCALE = 20.0


class NetOutputs(NamedTuple):
    policy_logits: torch.Tensor  # [N, S*S+1]
    value: torch.Tensor  # [N] tanh, side-to-move
    ownership: torch.Tensor  # [N, S*S] tanh, side-to-move
    score: torch.Tensor | None = None  # [N] margin / SCORE_SCALE
    policy_opp_logits: torch.Tensor | None = None  # [N, S*S+1] opponent's reply


@dataclass
class ArchConfig:
    """Serializable architecture description stored inside every checkpoint."""

    kind: str = "legacy"  # legacy | gpool
    blocks: int = 6
    channels: int = 64
    board_size: int = 9
    # gpool only
    gpool_every: int = 3  # every Nth block is a global-pooling block (0 = none)
    gpool_channels: int = 16
    head_channels: int = 32
    value_hidden: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.blocks}x{self.channels}"


# ---------------------------------------------------------------- legacy ----


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class LegacyNet(nn.Module):
    """State-dict compatible with the original ``GoferBootstrapNet``."""

    def __init__(self, board_size: int = 9, channels: int = 64, blocks: int = 6) -> None:
        super().__init__()
        self.board_size = board_size
        self.policy_size = board_size * board_size + 1
        flat = channels * board_size * board_size
        self.stem = nn.Sequential(
            nn.Conv2d(FEATURE_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.global_fc = nn.Sequential(nn.Linear(GLOBALS, channels), nn.ReLU(inplace=True))
        self.policy_fc = nn.Linear(flat + channels, self.policy_size)
        self.value_fc = nn.Sequential(
            nn.Linear(flat + channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        self.ownership_conv = nn.Conv2d(channels, 1, 1)

    def forward_all(self, spatial: torch.Tensor, global_in: torch.Tensor) -> NetOutputs:
        x = self.blocks(self.stem(spatial))
        g = self.global_fc(global_in)
        flat = torch.cat([x.reshape(x.size(0), -1), g], dim=1)
        policy_logits = self.policy_fc(flat)
        value = torch.tanh(self.value_fc(flat).squeeze(-1))
        own = torch.tanh(self.ownership_conv(x)).reshape(x.size(0), -1)
        return NetOutputs(policy_logits, value, own)

    def forward(
        self, spatial: torch.Tensor, global_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward_all(spatial, global_in)
        return out.policy_logits, out.value, out.ownership


# ----------------------------------------------------------------- gpool ----


def _pool_stats(x: torch.Tensor) -> torch.Tensor:
    """[N, C, H, W] -> [N, 2C] (mean, max). ONNX-friendly (no adaptive pools)."""
    mean = x.mean(dim=(2, 3))
    mx = x.amax(dim=(2, 3))
    return torch.cat([mean, mx], dim=1)


class GlobalPoolBias(nn.Module):
    """Pools ``gc`` channels to a bias added to the remaining ``c - gc`` channels."""

    def __init__(self, channels: int, gc: int) -> None:
        super().__init__()
        self.gc = gc
        self.bn = nn.BatchNorm2d(gc)
        self.fc = nn.Linear(2 * gc, channels - gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        regular, pooled = x[:, : -self.gc], x[:, -self.gc :]
        g = _pool_stats(F.relu(self.bn(pooled)))
        return regular + self.fc(g)[:, :, None, None]


class GPoolResBlock(nn.Module):
    """Pre-activation residual block, optionally with a global-pooling bias."""

    def __init__(self, channels: int, gpool_channels: int = 0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gpool = GlobalPoolBias(channels, gpool_channels) if gpool_channels else None
        mid = channels - gpool_channels if gpool_channels else channels
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, channels, 3, padding=1, bias=False)
        # Zero-init the last conv so each block starts as identity: stabilizes
        # early training (the 6x32 seed-11 divergence in ADR 0005).
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        if self.gpool is not None:
            out = self.gpool(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out


class GPoolNet(nn.Module):
    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        c, hc = cfg.channels, cfg.head_channels
        self.board_size = cfg.board_size
        self.stem = nn.Conv2d(FEATURE_PLANES, c, 3, padding=1, bias=False)
        self.global_in = nn.Linear(GLOBALS, c)
        blocks = []
        for i in range(cfg.blocks):
            use_gp = cfg.gpool_every > 0 and (i + 1) % cfg.gpool_every == 0
            blocks.append(GPoolResBlock(c, cfg.gpool_channels if use_gp else 0))
        self.blocks = nn.ModuleList(blocks)
        self.trunk_bn = nn.BatchNorm2d(c)

        # Policy: 1x1 conv -> pooled bias -> per-point logits (2 maps: self, opponent reply).
        self.p_conv = nn.Conv2d(c, hc, 1, bias=False)
        self.g_conv = nn.Conv2d(c, hc, 1, bias=False)
        self.g_bn = nn.BatchNorm2d(hc)
        self.p_bias = nn.Linear(2 * hc, hc)
        self.p_bn = nn.BatchNorm2d(hc)
        self.p_out = nn.Conv2d(hc, 2, 1)
        self.pass_fc = nn.Linear(2 * hc, 2)

        # Value / score: 1x1 conv -> pool -> MLP.
        self.v_conv = nn.Conv2d(c, hc, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(hc)
        self.v_fc1 = nn.Linear(2 * hc, cfg.value_hidden)
        self.v_fc2 = nn.Linear(cfg.value_hidden, 2)  # value, score
        self.own_conv = nn.Conv2d(hc, 1, 1)

    def forward_all(self, spatial: torch.Tensor, global_in: torch.Tensor) -> NetOutputs:
        n = spatial.size(0)
        x = self.stem(spatial) + self.global_in(global_in)[:, :, None, None]
        for blk in self.blocks:
            x = blk(x)
        x = F.relu(self.trunk_bn(x))

        g = _pool_stats(F.relu(self.g_bn(self.g_conv(x))))
        p = self.p_conv(x) + self.p_bias(g)[:, :, None, None]
        p = F.relu(self.p_bn(p))
        maps = self.p_out(p).reshape(n, 2, -1)
        passes = self.pass_fc(g)
        policy = torch.cat([maps[:, 0], passes[:, 0:1]], dim=1)
        policy_opp = torch.cat([maps[:, 1], passes[:, 1:2]], dim=1)

        v = F.relu(self.v_bn(self.v_conv(x)))
        vs = self.v_fc2(F.relu(self.v_fc1(_pool_stats(v))))
        value = torch.tanh(vs[:, 0])
        score = vs[:, 1]
        own = torch.tanh(self.own_conv(v)).reshape(n, -1)
        return NetOutputs(policy, value, own, score, policy_opp)

    def forward(
        self, spatial: torch.Tensor, global_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward_all(spatial, global_in)
        return out.policy_logits, out.value, out.ownership


# --------------------------------------------------------------- factory ----

PRESETS: dict[str, ArchConfig] = {
    "legacy-6x64": ArchConfig("legacy", 6, 64),
    "legacy-4x48": ArchConfig("legacy", 4, 48),
    "gpool-6x64": ArchConfig("gpool", 6, 64, gpool_every=3, gpool_channels=16, head_channels=32),
    "gpool-10x96": ArchConfig("gpool", 10, 96, gpool_every=3, gpool_channels=24, head_channels=32),
    "gpool-15x128": ArchConfig(
        "gpool", 15, 128, gpool_every=3, gpool_channels=32, head_channels=48, value_hidden=96
    ),
    "gpool-tiny": ArchConfig("gpool", 2, 16, gpool_every=2, gpool_channels=4, head_channels=8, value_hidden=16),
}


def build_net(cfg: ArchConfig) -> nn.Module:
    if cfg.kind == "legacy":
        return LegacyNet(cfg.board_size, cfg.channels, cfg.blocks)
    if cfg.kind == "gpool":
        return GPoolNet(cfg)
    raise ValueError(f"unknown arch kind {cfg.kind!r}")


def resolve_arch(spec: str | ArchConfig | None, **overrides: Any) -> ArchConfig:
    """Preset name ("gpool-6x64"), "kind-BxC", or an ArchConfig; overrides win."""
    if isinstance(spec, ArchConfig):
        cfg = ArchConfig.from_dict(spec.to_dict())
    elif not spec:
        cfg = ArchConfig()
    elif spec in PRESETS:
        cfg = ArchConfig.from_dict(PRESETS[spec].to_dict())
    else:
        kind, _, shape = spec.partition("-")
        b, _, c = shape.partition("x")
        if kind not in ("legacy", "gpool") or not b.isdigit() or not c.isdigit():
            raise ValueError(f"bad arch spec {spec!r}; use a preset {sorted(PRESETS)} or kind-BxC")
        cfg = ArchConfig(kind, int(b), int(c))
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def infer_arch(state: dict[str, torch.Tensor], board_size: int = 9) -> ArchConfig:
    """Recover an ArchConfig from a bare state_dict (for pre-metadata checkpoints)."""
    if "policy_fc.weight" in state:
        channels = state["stem.0.weight"].shape[0]
        blocks = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
        return ArchConfig("legacy", blocks, channels, board_size)
    if "p_out.weight" in state:
        channels = state["stem.weight"].shape[0]
        idx = {int(k.split(".")[1]) for k in state if k.startswith("blocks.")}
        gp = [i for i in idx if f"blocks.{i}.gpool.fc.weight" in state]
        gpc = state[f"blocks.{gp[0]}.gpool.bn.weight"].shape[0] if gp else 0
        every = (gp[0] + 1) if gp else 0
        return ArchConfig(
            "gpool",
            len(idx),
            channels,
            board_size,
            gpool_every=every,
            gpool_channels=gpc or 16,
            head_channels=state["p_conv.weight"].shape[0],
            value_hidden=state["v_fc1.weight"].shape[0],
        )
    raise ValueError("unrecognized state_dict layout")


# Heads the engine never reads; older checkpoints may predate them.
AUX_PREFIXES = ("ownership_conv.",)


def load_weights(net: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    """Strict load, except auxiliary heads may be missing (they keep their init).

    Returns the missing auxiliary keys. Any other mismatch raises.
    """
    result = net.load_state_dict(state, strict=False)
    bad_missing = [k for k in result.missing_keys if not k.startswith(AUX_PREFIXES)]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch: missing={bad_missing} unexpected={result.unexpected_keys}"
        )
    return list(result.missing_keys)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def forward_all(net: nn.Module, spatial: torch.Tensor, global_in: torch.Tensor) -> NetOutputs:
    fn = getattr(net, "forward_all", None)
    if fn is not None:
        return fn(spatial, global_in)
    logits, value, own = net(spatial, global_in)
    return NetOutputs(logits, value, own)


__all__ = [
    "ArchConfig",
    "GPoolNet",
    "LegacyNet",
    "NetOutputs",
    "PRESETS",
    "SCORE_SCALE",
    "build_net",
    "count_params",
    "forward_all",
    "infer_arch",
    "load_weights",
    "resolve_arch",
]
