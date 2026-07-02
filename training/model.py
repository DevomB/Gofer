"""Bootstrap 9x9 ResNet-small for Gofer v2.5."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_PLANES = 8
GLOBALS = 4
BOARD_SIZE = 9
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1


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


class GoferBootstrapNet(nn.Module):
    # Wider/deeper than the 4x32 bootstrap: 6x64 gives a much higher strength
    # ceiling on 9x9 while staying cheap enough to train per cycle.
    def __init__(self, board_size: int = BOARD_SIZE, channels: int = 64, blocks: int = 6) -> None:
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
        self.global_fc = nn.Sequential(
            nn.Linear(GLOBALS, channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(flat + channels, self.policy_size)
        self.value_fc = nn.Sequential(
            nn.Linear(flat + channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        # Ownership auxiliary head (KataGo-style): predicts per-point final owner
        # in [-1,1]. Shares the trunk, so it sharpens features the value/policy
        # heads reuse; the Go engine ignores this output at inference time.
        self.ownership_conv = nn.Conv2d(channels, 1, 1)

    def forward(
        self, spatial_input: torch.Tensor, global_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(spatial_input)
        x = self.blocks(x)
        g = self.global_fc(global_input)
        flat = torch.cat([x.reshape(x.size(0), -1), g], dim=1)
        policy_logits = self.policy_fc(flat)
        value = torch.tanh(self.value_fc(flat).squeeze(-1))
        ownership = torch.tanh(self.ownership_conv(x)).reshape(x.size(0), self.board_size * self.board_size)
        return policy_logits, value, ownership
