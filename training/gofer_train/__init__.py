"""Gofer learner: data, nets, trainer and ONNX export for the self-play loop.

Importable both as ``gofer_train`` (PYTHONPATH=training, the scripts' style)
and ``training.gofer_train``.
"""

from .nets import PRESETS, ArchConfig, build_net, infer_arch, resolve_arch
from .shards import Rows, load_rows, read_shard, write_shard
from .trainer import TrainConfig, Trainer, load_net, train

__all__ = [
    "PRESETS",
    "ArchConfig",
    "Rows",
    "TrainConfig",
    "Trainer",
    "build_net",
    "infer_arch",
    "load_net",
    "load_rows",
    "read_shard",
    "resolve_arch",
    "train",
    "write_shard",
]
