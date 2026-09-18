"""Pytest path: allow `from replay import` style in training tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Raw SGF corpora / run outputs: nothing to collect, and some directories are
# unreadable on Windows (PermissionError aborts collection of the whole tree).
collect_ignore_glob = ["experiments/*", "data/*", "checkpoints/*", "state/*"]
