"""Training loop manifest for ML pipeline v3."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 3


def default_manifest(*, seed: str = "fresh", best_win_rate: float = 0.0, best_cycle: int = 0) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "cycle": best_cycle,
        "best_win_rate": best_win_rate,
        "best_cycle": best_cycle,
        "replay_rows": 0,
        "last_promoted_at": None,
        "seed": seed,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_manifest()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != MANIFEST_VERSION:
        data["version"] = MANIFEST_VERSION
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
