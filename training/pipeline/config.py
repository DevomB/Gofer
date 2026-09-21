"""Typed pipeline configuration loaded from TOML (configs/pipeline*.toml).

Every knob of the self-play -> train -> export -> gate loop lives here so a run
is fully described by one file (snapshotted into the run dir at start). CLI
overrides use dotted keys: ``--set gating.max_games=80``.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    name: str = "default"
    root: str = "runs"                  # run dir = <root>/<name>
    max_cycles: int = 0                 # 0 = until deadline
    deadline_hours: float = 0.0         # 0 = no deadline
    seed: int = 42
    overlap_selfplay: bool = True       # generate cycle N+1 data while cycle N trains/gates
    init_checkpoint: str = ""           # optional warm-start .pt for the first training run
    keep_train_dirs: int = 0            # >0: delete train/cycle-* dirs older than the newest N (champions are copied out)
    keep_candidates: int = 0            # >0: delete non-champion candidate ONNX older than the newest N


@dataclass
class EngineConfig:
    gofer_bin: str = ""                 # "" = bin/gofer(.exe)
    build: bool = True                  # go build the engine once at start
    backend: str = "inprocess"          # inprocess (ORT in Go, -tags=onnx) | sidecar (HTTP)
    ort_lib: str = ""                   # ONNXRUNTIME_SHARED_LIBRARY_PATH override
    python: str = ""                    # "" = current interpreter
    sidecar_base_port: int = 8080
    eval_timeout: str = "2s"
    # Evaluations per inference call; 0 matches stage parallelism.
    batch_size: int = 0


@dataclass
class SelfplayConfig:
    games_per_cycle: int = 200
    bootstrap_games: int = 0            # heuristic games before the first net (0 = games_per_cycle)
    board_size: int = 9
    komi: float = 6.5
    full_playouts: int = 200
    fast_playouts: int = 50
    cap_randomize_p: float = 0.20
    temp_moves: int = 16
    eval: str = "mix"                   # once a champion exists: onnx | mix
    onnx_fraction: float = 0.7
    parallel: int = 0                   # 0 = os.cpu_count()


@dataclass
class ReplayConfig:
    # Sliding window (KataGo-style): it grows sublinearly with total data so early
    # heuristic-quality rows age out quickly but the window never collapses.
    min_window_rows: int = 20_000
    max_window_rows: int = 500_000
    window_alpha: float = 0.75
    window_beta: float = 0.4
    window_decay: float = 0.0           # recency weighting, passed to the trainer
    min_rows_to_train: int = 1_000      # accumulate data before the first training run
    archive_factor: float = 2.0         # shards older than factor * window move to selfplay/archive


@dataclass
class TrainConfig:
    script: str = "training/train_bootstrap.py"
    export_script: str = "training/export_onnx.py"
    epochs_fresh: int = 25
    epochs_resume: int = 15
    lr_fresh: float = 0.01
    lr_resume: float = 0.001
    patience: int = 5
    # "champion" restarts from the latest promotion; "latest" retains rejected work.
    warm_start: str = "champion"
    # Extra trainer flags passed through verbatim: {batch-size = 256, amp = true}
    # -> --batch-size 256 --amp. Keeps the orchestrator decoupled from trainer flags.
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatingConfig:
    mode: str = "normal"                # normal | hold (arena runs, never promotes)
    playouts: int = 400
    batch_games: int = 40               # arena games per SPRT step (even: colors alternate)
    max_games: int = 600
    bootstrap_games: int = 40           # sanity arena vs heuristic for the first net
    # The initial network must pass the same sequential gate against the heuristic.
    opening_moves: int = 8
    parallel: int = 0
    # SPRT hypotheses: H0 is elo0 better; H1 is elo1 better.
    elo0: float = 0.0
    elo1: float = 35.0
    alpha: float = 0.05
    beta: float = 0.10
    # Fallback when SPRT is inconclusive at max_games (legacy v3 rule).
    promote_win: float = 0.55
    # Periodically evaluate against the fixed heuristic anchor; 0 disables it.
    anchor_every: int = 0
    anchor_games: int = 40


@dataclass
class PublishConfig:
    """Where each new champion ("best") goes. Old bests are never deleted or
    overwritten: every generation gets its own registry file and, optionally,
    its own GitHub Release."""
    enabled: bool = True
    registry: str = "models/champions"  # every published best + index.json (never pruned)
    alias: str = ""                     # stable path overwritten with the current best (e.g. models/gofer-9x9-best.onnx)
    github_release: bool = False        # gh release per best (outward-facing: opt in)
    github_repo: str = ""               # owner/name; "" = the repo gh resolves from cwd
    release_prefix: str = "gofer-9x9-"   # files/tags: <prefix><run>-gen<NNNN>
    # Guard against a "false best": before publishing, the new champion also
    # plays the champion from `regression_lookback` generations back. Scoring
    # below regression_min_score holds it back (status "held"; alias and
    # release untouched).
    regression_games: int = 0           # 0 disables (even: colors alternate)
    regression_lookback: int = 2
    regression_min_score: float = 0.5
    regression_action: str = "hold"     # hold: keep training from it | demote: also revert the champion


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.run.root) / self.run.name

    def python(self) -> str:
        return self.engine.python or sys.executable

    def gofer_bin(self) -> Path:
        if self.engine.gofer_bin:
            return Path(self.engine.gofer_bin)
        return Path("bin") / ("gofer.exe" if os.name == "nt" else "gofer")

    def selfplay_parallel(self) -> int:
        return self.selfplay.parallel or os.cpu_count() or 4

    def gating_parallel(self) -> int:
        return self.gating.parallel or os.cpu_count() or 4

    def validate(self) -> None:
        errors = []
        if self.engine.backend not in ("inprocess", "sidecar"):
            errors.append(f"engine.backend must be inprocess|sidecar, got {self.engine.backend!r}")
        if self.gating.mode not in ("normal", "hold"):
            errors.append(f"gating.mode must be normal|hold, got {self.gating.mode!r}")
        if self.train.warm_start not in ("champion", "latest"):
            errors.append(f"train.warm_start must be champion|latest, got {self.train.warm_start!r}")
        if self.gating.batch_games <= 0 or self.gating.batch_games % 2:
            errors.append("gating.batch_games must be a positive even number (colors alternate)")
        if self.gating.bootstrap_games <= 0 or self.gating.bootstrap_games % 2:
            errors.append("gating.bootstrap_games must be a positive even number (colors alternate)")
        if self.selfplay.games_per_cycle <= 0:
            errors.append("selfplay.games_per_cycle must be positive")
        if self.selfplay.bootstrap_games < 0:
            errors.append("selfplay.bootstrap_games must be >= 0 (0 = use games_per_cycle)")
        if self.gating.max_games < self.gating.batch_games:
            errors.append("gating.max_games must be >= gating.batch_games")
        if not 0 < self.gating.alpha < 0.5 or not 0 < self.gating.beta < 0.5:
            errors.append("gating.alpha/beta must be in (0, 0.5)")
        if self.gating.elo1 <= self.gating.elo0:
            errors.append("gating.elo1 must exceed gating.elo0")
        if self.selfplay.eval not in ("onnx", "mix"):
            errors.append("selfplay.eval must be onnx|mix")
        if self.replay.min_window_rows > self.replay.max_window_rows:
            errors.append("replay.min_window_rows must be <= replay.max_window_rows")
        if self.publish.regression_games < 0 or self.publish.regression_games % 2:
            errors.append("publish.regression_games must be 0 or a positive even number")
        if self.publish.regression_action not in ("hold", "demote"):
            errors.append("publish.regression_action must be hold|demote")
        if self.publish.regression_lookback < 1:
            errors.append("publish.regression_lookback must be >= 1")
        if errors:
            raise ValueError("invalid pipeline config:\n  " + "\n  ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _section_types() -> dict[str, type]:
    return {
        "run": RunConfig,
        "engine": EngineConfig,
        "selfplay": SelfplayConfig,
        "replay": ReplayConfig,
        "train": TrainConfig,
        "gating": GatingConfig,
        "publish": PublishConfig,
    }


def from_dict(data: dict[str, Any]) -> PipelineConfig:
    types = _section_types()
    unknown = set(data) - set(types)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    sections = {}
    for name, cls in types.items():
        raw = data.get(name, {})
        known = {f.name for f in dataclasses.fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown keys in [{name}]: {sorted(bad)}")
        sections[name] = cls(**raw)
    cfg = PipelineConfig(**sections)
    cfg.validate()
    return cfg


def _parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` (TOML-typed value) overrides in place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be section.key=value: {item!r}")
        dotted, value = item.split("=", 1)
        parts = dotted.strip().split(".")
        if len(parts) < 2:
            raise ValueError(f"override key needs a section: {dotted!r}")
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value.strip())
    return data


def load_config(path: Path | None, overrides: list[str] | None = None) -> PipelineConfig:
    data: dict[str, Any] = {}
    if path is not None:
        with path.open("rb") as f:
            data = tomllib.load(f)
    apply_overrides(data, overrides or [])
    return from_dict(data)


def dump_toml(cfg: PipelineConfig) -> str:
    """Serialize to TOML (flat sections, inline tables for dicts)."""
    lines: list[str] = []
    for section, values in cfg.to_dict().items():
        lines.append(f"[{section}]")
        for key, val in values.items():
            lines.append(f"{key} = {_toml_value(val)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return repr(val)
    if isinstance(val, dict):
        inner = ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in val.items())
        return "{" + inner + "}"
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in val) + "]"
    escaped = str(val).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_key(key: str) -> str:
    return key if key.replace("-", "").replace("_", "").isalnum() else f'"{key}"'
