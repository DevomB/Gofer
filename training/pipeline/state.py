"""Run state (manifest v4): champion lineage, stage checkpoints, Elo ladder.

Everything a crashed run needs to resume lives in <run_dir>/state.json, written
atomically. Each cycle records completed stages so a restart resumes at the
first unfinished stage instead of redoing self-play or training.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_VERSION = 4
STAGES = ("selfplay", "train", "export", "gate", "promote", "publish")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Generation:
    generation: int
    cycle: int
    onnx: str
    pt: str
    elo: float
    promoted_at: str
    gate: dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleProgress:
    cycle: int
    done: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunState:
    version: int = STATE_VERSION
    cycle: int = 0                                  # last fully completed cycle
    lifetime_rows: int = 0                          # every self-play row ever produced
    in_progress: CycleProgress | None = None
    generations: list[Generation] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @property
    def champion(self) -> Generation | None:
        return self.generations[-1] if self.generations else None

    def begin_cycle(self, cycle: int) -> CycleProgress:
        if self.in_progress is None or self.in_progress.cycle != cycle:
            self.in_progress = CycleProgress(cycle)
        return self.in_progress

    def mark(self, stage: str, **data: Any) -> None:
        assert self.in_progress is not None, "mark() outside a cycle"
        if stage not in self.in_progress.done:
            self.in_progress.done.append(stage)
        self.in_progress.data.update(data)

    def is_done(self, stage: str) -> bool:
        return self.in_progress is not None and stage in self.in_progress.done

    def finish_cycle(self) -> None:
        assert self.in_progress is not None
        self.cycle = self.in_progress.cycle
        self.in_progress = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        if data.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported run state version {data.get('version')} (want {STATE_VERSION})")
        ip = data.get("in_progress")
        return cls(
            version=STATE_VERSION,
            cycle=int(data.get("cycle", 0)),
            lifetime_rows=int(data.get("lifetime_rows", 0)),
            in_progress=CycleProgress(**ip) if ip else None,
            generations=[Generation(**g) for g in data.get("generations", [])],
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )


def load_state(path: Path) -> RunState:
    if not path.exists():
        return RunState()
    return RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_state(path: Path, state: RunState) -> None:
    state.updated_at = utc_now()
    write_json_atomic(path, state.to_dict())


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a scanner/indexer briefly holds the target open
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def append_event(path: Path, **event: Any) -> None:
    """One JSON object per line; the report and cost estimator read these."""
    path.parent.mkdir(parents=True, exist_ok=True)
    event.setdefault("ts", utc_now())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
