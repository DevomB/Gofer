"""Bootstrap 9x9 ResNet-small for Gofer v2.5.

Kept as the stable import point for existing callers; the implementations (and
the newer ``gpool`` architecture) live in ``gofer_train.nets``.
"""

from __future__ import annotations

from gofer_train.nets import FEATURE_PLANES, GLOBALS, LegacyNet, ResBlock

BOARD_SIZE = 9
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1


class GoferBootstrapNet(LegacyNet):
    """Legacy flatten+FC net (6x64 default). State dicts are unchanged."""


__all__ = ["BOARD_SIZE", "FEATURE_PLANES", "GLOBALS", "POLICY_SIZE", "GoferBootstrapNet", "ResBlock"]
