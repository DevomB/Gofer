"""AlphaZero-style training loop: self-play -> train -> export -> evaluate -> champion.

Every phase writes its outputs before state.json records it as done, so the
loop can be stopped at any point and resumed with the same command.

    python -m training.loop --run runs/9x9 --gofer bin/gofer --hours 6 --release-repo DevomB/Gofer

With --release-repo, every new champion is published as a GitHub Release
(see training/release.py).
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
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from training import elo, release
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
    init_from: str = ""  # checkpoint (.pt) whose network the run starts from; empty: random weights
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
    # champions: on anchor generations, a network rated above the champion plays
    # it directly and takes over only if the match's SPRT accepts H1
    champion_games: int = 200  # game limit (SPRT usually stops sooner); 0 disables champions
    champion_visits: int = 200
    champion_sprt: str = "0,50"  # elo0,elo1: false promotion <= 5.3% when not stronger
    champion_min_elo: float = 1500.0  # early generations (gen 0 = 0) never become champion

    @classmethod
    def from_dict(cls, d: dict) -> "LoopConfig":
        d = dict(d)
        d["train"] = TrainConfig(**d.get("train", {}))
        return cls(**d)


class Run:
    def __init__(self, root: Path, cfg: LoopConfig, gofer: Path, threads: int, release_repo: str = "") -> None:
        self.root, self.cfg, self.gofer, self.threads = root, cfg, gofer, threads
        self.release_repo = release_repo
        for d in ("models", "ckpt", "data", "eval", "sgf"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self.state_path = root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"gen": 0, "step": 0, "done": {}}
        self.env = dict(os.environ)
        if "ONNXRUNTIME_SHARED_LIBRARY_PATH" not in self.env:
            self.env["ONNXRUNTIME_SHARED_LIBRARY_PATH"] = find_ort_library()
        self.env["PATH"] = tool_path()

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
        """Create the initial network: random weights, or the network of the
        checkpoint named by init_from. The checkpoint and its ONNX export are
        checked separately, so a run stopped between the two still starts."""
        c = self.cfg
        if not self.ckpt_path(0).exists():
            if c.init_from:
                model = load_network(Path(c.init_from), c).to(device())
                origin = f"started from the network of {c.init_from}: {c.blocks}x{c.channels}"
            else:
                torch.manual_seed(0)
                model = GoferNet(ModelConfig(board_size=c.board_size, blocks=c.blocks, channels=c.channels)).to(device())
                origin = f"initialized {c.blocks}x{c.channels} network"
            save_checkpoint(self.ckpt_path(0), model, make_optimizer(model, c.train), 0)
            self.log(f"{origin} ({param_count(model):,} parameters)")
        if not self.model_path(0).exists():
            model, _, _ = load_checkpoint(self.ckpt_path(0), c.train, device())
            export(model, self.model_path(0))

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

    def match(self, a: str, b: str, games: int, visits: int, tag: str, extra: list[str] | None = None) -> dict:
        out = self.root / "eval" / f"{tag}.json"
        if not out.exists():
            self.gofer_run([
                "arena", "-a", a, "-b", b, "-games", str(games), "-visits", str(visits), "-size", str(self.cfg.board_size),
                "-komi", str(self.cfg.komi), "-parallel", str(2 * self.threads), "-json", str(out), "-seed", str(match_seed(tag)),
                *(extra or []),
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

    def champion(self, gen: int) -> None:
        if self.is_done(gen, "champion"):
            return
        c, new = self.cfg, gen + 1
        if c.champion_games > 0 and new % c.anchor_every == 0:
            ratings = json.loads((self.root / "eval" / "ratings.json").read_text())
            name = lambda g: f"gen-{g:04d}"
            mine = ratings[name(new)]["elo"]
            champ = self.state.get("champion")
            theirs = ratings.get(name(champ["gen"]), {}).get("elo", champ["elo"]) if champ else None
            if mine >= c.champion_min_elo and (champ is None or mine > theirs):
                if champ is None:
                    self.promote(new, ratings, None)
                else:
                    r = self.match(f"nn:{self.model_path(new)}", f"nn:{self.model_path(champ['gen'])}", c.champion_games,
                                   c.champion_visits, f"{name(new)}-vs-champion-{name(champ['gen'])}", ["-sprt", c.champion_sprt])
                    self.state.setdefault("results", []).append(
                        {"a": name(new), "b": name(champ["gen"]), "wins": r["wins"], "losses": r["losses"], "draws": r["draws"]})
                    self.log(f"champion match: {name(new)} vs {name(champ['gen'])}: {r['wins']}-{r['losses']}-{r['draws']} "
                             f"elo {r['elo']:+.0f} [{r['elo_ci_low']:+.0f}, {r['elo_ci_high']:+.0f}] sprt {r.get('sprt') or 'undecided'}")
                    if release.won(r):
                        self.promote(new, ratings, {**r, "opponent": name(champ["gen"]), "visits": c.champion_visits})
        self.publish()
        self.mark(gen, "champion")

    def promote(self, gen: int, ratings: dict, match: dict | None) -> None:
        """Make generation gen the champion: keep a copy of the network and its
        metadata under champion/, and queue a release."""
        c, name = self.cfg, f"gen-{gen:04d}"
        d = self.root / "champion" / name
        d.mkdir(parents=True, exist_ok=True)
        asset = d / release.asset_name(c.board_size)
        shutil.copyfile(self.model_path(gen), asset)
        selfplay = [v["selfplay"] for g, v in self.state["done"].items() if int(g) < gen and isinstance(v.get("selfplay"), dict)]
        keep = ("wins", "losses", "draws", "elo", "elo_ci_low", "elo_ci_high", "sprt", "sprt_pairs",
                "sprt_sweeps_a", "sprt_sweeps_b", "opponent", "visits")
        info = {
            "run": self.root.name, "gen": gen, "tag": release.release_tag(self.root, gen),
            "elo": ratings[name]["elo"], "stderr": ratings[name]["stderr"],
            "match": {k: match.get(k) for k in keep} if match else None,
            "anchors": [r for r in self.state.get("results", []) if r["a"] == name and not r["b"].startswith("gen-")],
            "board_size": c.board_size, "komi": c.komi, "blocks": c.blocks, "channels": c.channels,
            "init_from": c.init_from,
            "games": c.games_per_gen * len(selfplay), "positions": sum(s["rows"] for s in selfplay),
            "sha256": release.sha256(asset),
        }
        (d / "champion.json").write_text(json.dumps(info, indent=2))
        latest = self.root / "champion" / release.asset_name(c.board_size)
        shutil.copyfile(asset, latest.with_suffix(".tmp"))
        latest.with_suffix(".tmp").replace(latest)
        self.state["champion"] = {"gen": gen, "elo": info["elo"], "tag": info["tag"]}
        self.state.setdefault("releases", []).append({"gen": gen, "tag": info["tag"], "published": False})
        self.save_state()
        self.log(f"champion: {name} (elo {info['elo']:+.0f}) -> {latest}")

    def publish(self) -> None:
        if release.publish_pending(self.root, self.state, self.release_repo, self.log):
            self.save_state()

    def run(self, hours: float, max_gens: int) -> None:
        deadline = time.time() + hours * 3600
        self.init()
        self.publish()  # retry anything a previous session could not publish
        while self.state["gen"] < max_gens:
            gen = self.state["gen"]
            for phase in (self.selfplay, self.train, self.evaluate, self.champion):
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


def load_network(path: Path, cfg: LoopConfig) -> GoferNet:
    """The network of checkpoint path, for a run with settings cfg. No weight
    depends on the board size (the network is convolutions and board-wide
    pooling), so a network trained on one size can start a run on another.
    The optimizer state is left behind: the new run has its own learning-rate
    schedule."""
    if not path.is_file():
        raise SystemExit(f"cannot start: init_from names {path}, which is not a file")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = ModelConfig(**{**ck["model_config"], "board_size": cfg.board_size})
    if (mc.blocks, mc.channels) != (cfg.blocks, cfg.channels):
        raise SystemExit(f"cannot start: {path} holds a {mc.blocks}x{mc.channels} network and this run is set to "
                         f"{cfg.blocks}x{cfg.channels}. Add --set blocks={mc.blocks} --set channels={mc.channels}.")
    model = GoferNet(mc)
    model.load_state_dict(ck["model"])
    return model


def match_seed(tag: str) -> int:
    """Arena seed for the match named tag. Each match gets its own seed, so each
    draws its own random openings; the same match always gets the same seed."""
    return zlib.crc32(tag.encode())


def tool_path() -> str:
    """PATH for the engine and the programs it starts; Debian and Ubuntu install gnugo in /usr/games."""
    path = os.environ.get("PATH", "")
    return f"{path}:/usr/games" if Path("/usr/games").is_dir() else path


def anchor_problems(anchors: list[str], path: str) -> list[str]:
    """Anchors that cannot play on this machine, one message each: an external
    program that is not on path, or a network file that is missing. Specs are
    those of `gofer arena` (see cmd/gofer/engine.go)."""
    problems = []
    for spec in anchors:
        head, *opts = spec.split(",")
        kind, _, arg = head.partition(":")
        if kind == "gnugo":
            program = next((o.split("=", 1)[1] for o in opts if o.startswith("bin=")), "gnugo")
            if shutil.which(program, path=path) is None:
                problems.append(f"{spec}: GNU Go ({program}) is not installed")
        elif kind == "gtp":
            program = (spec.removeprefix("gtp:").split() or [""])[0]
            if shutil.which(program, path=path) is None:
                problems.append(f"{spec}: program {program!r} not found")
        elif kind == "nn" and not Path(arg).exists():
            problems.append(f"{spec}: network file {arg} not found")
    return problems


def check_anchors(anchors: list[str], run: Path) -> None:
    """Stop before any work is done if an anchor cannot play: the first anchor
    match comes only after a full generation of self-play and training."""
    problems = anchor_problems(anchors, tool_path())
    if not problems:
        return
    lines = ["cannot start: these anchors cannot play on this machine", *(f"  {p}" for p in problems)]
    if any("GNU Go" in p for p in problems):
        lines.append("Install GNU Go: sudo dnf install gnugo (Fedora), sudo apt-get install gnugo (Debian, Ubuntu).")
    lines.append(f"""To use other anchors: --set 'anchors=["rollout,visits=800"]' on a new run, or edit "anchors" in {run / 'config.json'}.""")
    raise SystemExit("\n".join(lines))


def apply_overrides(d: dict, overrides: list[str]) -> list[str]:
    """Apply KEY=VALUE settings to a LoopConfig dict in place. Returns the keys
    whose value changed."""
    changed = []
    for kv in overrides:
        k, eq, v = kv.partition("=")
        if not eq:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        target = d["train"] if k.startswith("train.") else d
        name = k.removeprefix("train.")
        if name not in target:
            raise SystemExit(f"unknown config key {k}")
        try:
            new = v if isinstance(target[name], str) else json.loads(v)
        except json.JSONDecodeError:
            raise SystemExit(f"--set {k}: {v!r} is not a valid value (current: {json.dumps(target[name])})") from None
        if new != target[name]:
            changed.append(k)
        target[name] = new
    return changed


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
    ap.add_argument("--release-repo", default=os.environ.get("GOFER_RELEASE_REPO", ""), metavar="OWNER/REPO",
                    help="publish each new champion as a GitHub Release with gh (default: $GOFER_RELEASE_REPO)")
    args = ap.parse_args()
    cfg_path = args.run / "config.json"
    resuming = cfg_path.exists()
    d = dataclasses.asdict(LoopConfig.from_dict(json.loads(cfg_path.read_text())) if resuming else LoopConfig())
    changed = apply_overrides(d, args.set)
    if resuming and changed:
        # the same command resumes a run, so --set may repeat the run's settings, but not change them
        raise SystemExit(f"{cfg_path} already fixes this run's settings, and --set would change {', '.join(changed)}. "
                         "Edit that file to change a running run, or start a new one with another --run directory.")
    cfg = LoopConfig.from_dict(d)
    check_anchors(cfg.anchors, args.run)
    if cfg.init_from and not (args.run / "ckpt" / "gen-0000.pt").exists():
        load_network(Path(cfg.init_from), cfg)  # refuse a missing or mismatched checkpoint before anything is written
    if not resuming:
        args.run.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    torch.set_num_threads(args.threads)
    Run(args.run, cfg, args.gofer.resolve(), args.threads, args.release_repo).run(args.hours, args.gens)


if __name__ == "__main__":
    main()
