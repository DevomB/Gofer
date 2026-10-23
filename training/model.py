"""Residual policy/value network with global pooling (KataGo-style heads)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.features import NUM_GLOBALS, NUM_PLANES


@dataclass
class ModelConfig:
    board_size: int = 9
    blocks: int = 6
    channels: int = 64
    head_channels: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


def global_pool(x: torch.Tensor) -> torch.Tensor:
    """Mean and max over the board: [N, C, H, W] -> [N, 2C]."""
    return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class GlobalPoolBlock(nn.Module):
    """Residual block whose second conv is biased by pooled board context,
    so local decisions can see whole-board state (ko threats, who is ahead)."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pool_fc = nn.Linear(2 * c, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = y + self.pool_fc(global_pool(y))[:, :, None, None]
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class GoferNet(nn.Module):
    """Inputs: spatial [N, NUM_PLANES, S, S], global [N, NUM_GLOBALS].
    Outputs: policy logits [N, S*S+1], value [N] in [-1, 1] for the side to
    move, ownership [N, S*S] in [-1, 1] for the side to move."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c, h = cfg.channels, cfg.head_channels
        self.stem = nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False)
        self.stem_global = nn.Linear(NUM_GLOBALS, c)
        self.stem_bn = nn.BatchNorm2d(c)
        blocks: list[nn.Module] = []
        for i in range(cfg.blocks):
            # every third block (starting with the second) gets global pooling
            blocks.append(GlobalPoolBlock(c) if i % 3 == 1 else ResBlock(c))
        self.trunk = nn.Sequential(*blocks)

        self.policy_conv = nn.Conv2d(c, h, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(h)
        self.policy_out = nn.Conv2d(h, 1, 1)
        self.pass_fc = nn.Linear(2 * h, 1)

        self.value_conv = nn.Conv2d(c, h, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(h)
        self.value_fc1 = nn.Linear(2 * h, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.ownership_out = nn.Conv2d(h, 1, 1)

    def forward(self, spatial: torch.Tensor, glob: torch.Tensor):
        x = self.stem(spatial) + self.stem_global(glob)[:, :, None, None]
        x = F.relu(self.stem_bn(x))
        x = self.trunk(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        board_logits = self.policy_out(p).flatten(1)
        pass_logit = self.pass_fc(global_pool(p))
        policy = torch.cat([board_logits, pass_logit], dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)))
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(global_pool(v))))).squeeze(1)
        ownership = torch.tanh(self.ownership_out(v)).flatten(1)
        return policy, value, ownership


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
