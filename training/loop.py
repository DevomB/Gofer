"""AlphaZero-style training loop: self-play -> train -> export -> evaluate.

Every phase writes its outputs before state.json records it as done, so the
loop can be stopped at any point and resumed with the same command.

    python -m training.loop --run runs/9x9 --gofer bin/gofer --hours 6
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from training import elo
from training.data import load_shard, load_window
from training.export import export
from training.model import GoferNet, ModelConfig, param_count
from training.train import TrainConfig, cosine_lr, device, evaluate, load_checkpoint, make_optimizer, save_checkpoint, train_steps


@dataclass
class LoopConfig:
    board_size: int = 9
    komi: float = 7.5
    blocks: int = 6
    channels: int = 64
    # self-play
    games_per_gen: int = 600
    full_visits: int = 160
    fast_visits: int = 32
    full_prob: float = 0.25
    temp_moves: int = 10
    resign: float = -0.9
    no_resign_prob: float = 0.1
    # training
    window_min: int = 60_000  # replay window grows from here...
    window_max: int = 400_000  # ...to here
    window_growth: int = 25_000  # rows added to the window per generation
    sample_reuse: float = 8.0  # training samples drawn per new self-play row
    total_gens: int = 30  # horizon for the learning-rate schedule
    train: TrainConfig = field(default_factory=TrainConfig)
    # evaluation
    eval_games: int = 40
    eval_visits: int = 100
    anchors: list[str] = field(default_factory=lambda: ["gnugo,level=1", "gnugo,level=6"])
    anchor_every: int = 2
    anchor_games: int = 40

    @classmethod
    def from_dict(cls, d: dict) -> "LoopConfig":
        d = dict(d)
        d["train"] = TrainConfig(**d.get("train", {}))
        return cls(**d)


class Run:
    def __init__(self, root: Path, cfg: LoopConfig, gofer: Path, threads: int) -> None:
        self.root, self.cfg, self.gofer, self.threads = root, cfg, gofer, threads
        for d in ("models", "ckpt", "data", "eval", "sgf"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.state_path = root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"gen": 0, "step": 0, "done": {}}
        self.env = dict(os.environ)
        if "ONNXRUNTIME_SHARED_LIBRARY_PATH" not in self.env:
            self.env["ONNXRUNTIME_SHARED_LIBRARY_PATH"] = find_ort_library()
        if Path("/usr/games").is_dir():  # Debian/Ubuntu install gnugo there
            self.env["PATH"] = self.env.get("PATH", "") + ":/usr/games"

    # ---- bookkeeping -------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.root / "log.txt", "a") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)

    def mark(self, gen: int, phase: str, info: dict | None = None) -> None:
        self.state["done"].setdefault(str(gen), {})[phase] = info or True
        self.save_state()

    def is_done(self, gen: int, phase: str) -> bool:
        return phase in self.state["done"].get(str(gen), {})

    def metric(self, record: dict) -> None:
        with open(self.root / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def model_path(self, gen: int) -> Path:
        return self.root / "models" / f"gen-{gen:04d}.onnx"

    def ckpt_path(self, gen: int) -> Path:
        return self.root / "ckpt" / f"gen-{gen:04d}.pt"

    def gofer_run(self, args: list[str]) -> None:
        cmd = [str(self.gofer), *args]
        self.log("$ " + " ".join(cmd))
        subprocess.run(cmd, check=True, env=self.env)

    # ---- phases ------------------------------------------------------
    def init(self) -> None:
        if self.ckpt_path(0).exists():
            return
        torch.manual_seed(0)
        mc = ModelConfig(board_size=self.cfg.board_size, blocks=self.cfg.blocks, channels=self.cfg.channels)
        model = GoferNet(mc).to(device())
        opt = make_optimizer(model, self.cfg.train)
        save_checkpoint(self.ckpt_path(0), model, opt, 0)
        export(model, self.model_path(0))
        self.log(f"initialized {mc.blocks}x{mc.channels} network ({param_count(model):,} parameters)")

    def selfplay(self, gen: int) -> None:
        if self.is_done(gen, "selfplay"):
            return
        c = self.cfg
        out = self.root / "data" / f"gen-{gen:04d}"
        if out.exists():
            shutil.rmtree(out)  # partial output from an interrupted run
        stats = self.root / "eval" / f"selfplay-{gen:04d}.json"
        self.gofer_run([
            "selfplay", "-model", str(self.model_path(gen)), "-size", str(c.board_size), "-komi", str(c.komi),
            "-games", str(c.games_per_gen), "-full-visits", str(c.full_visits), "-fast-visits", str(c.fast_visits),
            "-full-prob", str(c.full_prob), "-temp-moves", str(c.temp_moves), "-resign", str(c.resign),
            "-no-resign-prob", str(c.no_resign_prob), "-seed", str(1000 + gen), "-out", str(out),
            "-prefix", f"gen-{gen:04d}", "-sgf", str(self.root / "sgf" / f"gen-{gen:04d}"), "-stats", str(stats),
        ])
        info = json.loads(stats.read_text())
        self.metric({"gen": gen, "phase": "selfplay", **{k: v for k, v in info.items() if k != "shards"}})
        self.mark(gen, "selfplay", {"rows": info["rows"], "seconds": info["seconds"]})

    def shards(self, upto: int) -> list[Path]:
        out: list[Path] = []
        for g in range(upto + 1):
            out += sorted(Path(p) for p in glob.glob(str(self.root / "data" / f"gen-{g:04d}" / "*.npz")))
        return out

    def train(self, gen: int) -> None:
        if self.is_done(gen, "train"):
            return
        c = self.cfg
        dev = device()
        model, opt, step = load_checkpoint(self.ckpt_path(gen), c.train, dev)
        new_shards = self.shards(gen)[len(self.shards(gen - 1)) if gen > 0 else 0 :]
        new_rows = load_window(new_shards, 10**9) if new_shards else None
        val = evaluate(model, new_rows) if new_rows is not None else {}
        window = min(c.window_max, c.window_min + gen * c.window_growth)
        rows = load_window(self.shards(gen), window)
        steps = max(1, int(c.sample_reuse * len(new_rows) / c.train.batch_size)) if new_rows is not None else 0
        lr = cosine_lr(c.train.lr, gen, c.total_gens)
        self.log(f"train gen {gen}: window {len(rows):,} rows, {steps} steps, lr {lr:.2e}; "
                 f"previous net on new data: policy {val.get('policy', 0):.3f} (target entropy {val.get('policy_entropy', 0):.3f}) value {val.get('value', 0):.3f}")
        metrics, step = train_steps(model, opt, rows, steps, c.train, lr, step, seed=gen, log=self.log)
        save_checkpoint(self.ckpt_path(gen + 1), model, opt, step)
        export(model, self.model_path(gen + 1))
        self.state["step"] = step
        self.metric({"gen": gen, "phase": "train", "window": len(rows), "lr": lr, **metrics,
                     **{f"val_{k}": v for k, v in val.items()}})
        self.mark(gen, "train", {"steps": steps})

    def match(self, a: str, b: str, games: int, visits: int, tag: str) -> dict:
        out = self.root / "eval" / f"{tag}.json"
        if not out.exists():
            self.gofer_run([
                "arena", "-a", a, "-b", b, "-games", str(games), "-visits", str(visits), "-size", str(self.cfg.board_size),
                "-komi", str(self.cfg.komi), "-parallel", str(2 * self.threads), "-json", str(out), "-seed", str(len(tag)),
            ])
        return json.loads(out.read_text())

    def evaluate(self, gen: int) -> None:
        if self.is_done(gen, "eval"):
            return
        c = self.cfg
        new, old = gen + 1, gen
        name = lambda g: f"gen-{g:04d}"
        results = []
        r = self.match(f"nn:{self.model_path(new)}", f"nn:{self.model_path(old)}", c.eval_games, c.eval_visits, f"{name(new)}-vs-{name(old)}")
        results.append({"a": name(new), "b": name(old), "wins": r["wins"], "losses": r["losses"], "draws": r["draws"]})
        self.log(f"eval: {name(new)} vs {name(old)}: {r['wins']}-{r['losses']}-{r['draws']} elo {r['elo']:+.0f} [{r['elo_ci_low']:+.0f}, {r['elo_ci_high']:+.0f}]")
        if new % c.anchor_every == 0 or new == 1:
            for anchor in c.anchors:
                label = anchor_label(anchor)
                r = self.match(f"nn:{self.model_path(new)}", anchor, c.anchor_games, c.eval_visits, f"{name(new)}-vs-{label}")
                results.append({"a": name(new), "b": label, "wins": r["wins"], "losses": r["losses"], "draws": r["draws"]})
                self.log(f"eval: {name(new)} vs {label}: {r['wins']}-{r['losses']}-{r['draws']} score {r['score']:.2f}")
        self.state.setdefault("results", []).extend(results)
        ratings = elo.fit(self.state["results"], anchor="gen-0000")
        (self.root / "eval" / "ratings.json").write_text(json.dumps(ratings, indent=2, sort_keys=True))
        self.metric({"gen": new, "phase": "eval", "elo": ratings.get(name(new), {}).get("elo"), "results": results})
        self.log("ratings: " + ", ".join(f"{p} {v['elo']:+.0f}±{v['stderr']:.0f}" for p, v in sorted(ratings.items(), key=lambda kv: -kv[1]["elo"])[:8]))
        self.mark(gen, "eval")

    def run(self, hours: float, max_gens: int) -> None:
        deadline = time.time() + hours * 3600
        self.init()
        while self.state["gen"] < max_gens:
            gen = self.state["gen"]
            for phase in (self.selfplay, self.train, self.evaluate):
                if time.time() > deadline:
                    self.log("time budget reached; resume with the same command")
                    return
                phase(gen)
            self.state["gen"] = gen + 1
            self.save_state()
        self.log(f"finished {max_gens} generations")


def anchor_label(spec: str) -> str:
    """'gnugo,level=6' -> 'gnugo-l6' (options other than level are dropped)."""
    kind, *opts = spec.split(",")
    level = next((o.split("=", 1)[1] for o in opts if o.startswith("level=")), None)
    label = f"{kind}-l{level}" if level else kind
    return re.sub(r"[^A-Za-z0-9.-]+", "-", label)


def find_ort_library() -> str:
    import onnxruntime

    libs = glob.glob(str(Path(onnxruntime.__file__).parent / "capi" / "libonnxruntime.so*"))
    libs = [l for l in libs if "providers" not in l]
    if not libs:
        raise SystemExit("set ONNXRUNTIME_SHARED_LIBRARY_PATH to libonnxruntime.so")
    return sorted(libs)[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run directory (created or resumed)")
    ap.add_argument("--gofer", type=Path, default=Path("bin/gofer"), help="gofer binary built with -tags onnx")
    ap.add_argument("--hours", type=float, default=1e9, help="stop after this much wall time")
    ap.add_argument("--gens", type=int, default=10**6, help="stop after this many generations")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a LoopConfig field (new runs only)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    cfg_path = args.run / "config.json"
    if cfg_path.exists():
        cfg = LoopConfig.from_dict(json.loads(cfg_path.read_text()))
    else:
        d = dataclasses.asdict(LoopConfig())
        for kv in args.set:
            k, v = kv.split("=", 1)
            target = d["train"] if k.startswith("train.") else d
            k = k.removeprefix("train.")
            if k not in target:
                raise SystemExit(f"unknown config key {k}")
            cur = target[k]
            target[k] = json.loads(v) if not isinstance(cur, str) else v
        cfg = LoopConfig.from_dict(d)
        args.run.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    torch.set_num_threads(args.threads)
    Run(args.run, cfg, args.gofer.resolve(), args.threads).run(args.hours, args.gens)


if __name__ == "__main__":
    main()
