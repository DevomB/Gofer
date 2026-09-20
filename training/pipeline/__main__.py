"""CLI: python -m training.pipeline <command>.

  run       run the loop (resumable): --config configs/pipeline.toml [--cycles N] [--set k=v ...]
  status    print champion / cycle / progress for a run
  report    write a self-contained HTML dashboard for a run
  estimate  project wall-clock time and cloud cost from measured stage timings
  plan-sprt expected arena games per gate for a given SPRT setting
  publish   list published champions (--retry re-attempts failed GitHub releases)
  rollback  point "best" back at an earlier generation (--to N [--champion])
  config    print the fully-resolved config (defaults + file + overrides) as TOML
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from training.pipeline import stats
from training.pipeline.config import dump_toml, load_config
from training.pipeline.procs import SubprocessExecutor
from training.pipeline.runner import Pipeline, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None, help="pipeline TOML (default: built-in defaults)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config value (TOML syntax), repeatable")


def _sigterm_to_interrupt() -> None:
    """docker stop / spot preemption / CI timeouts send SIGTERM: stop like Ctrl-C so
    background self-play and sidecars are terminated. State is already checkpointed
    per stage, so rerunning the same command resumes."""
    def handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, handler)


def cmd_run(args: argparse.Namespace) -> int:
    _sigterm_to_interrupt()
    cfg = load_config(args.config, args.overrides)
    pipe = Pipeline(cfg, SubprocessExecutor(REPO_ROOT), REPO_ROOT)
    pipe.log(f"run '{cfg.run.name}' in {pipe.dir} backend={cfg.engine.backend}")
    pipe.prepare(build=False if args.no_build else None)
    n = pipe.run(max_cycles=args.cycles)
    pipe.log(f"completed {n} cycle(s); champion: {summarize(pipe.state_path)['champion']}")
    return 0


def _run_dir(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config, args.overrides)
    d = cfg.run_dir
    return d if d.is_absolute() else REPO_ROOT / d


def cmd_status(args: argparse.Namespace) -> int:
    d = _run_dir(args)
    if not (d / "state.json").exists():
        print(f"no run state at {d}")
        return 1
    s = summarize(d / "state.json")
    if args.json:
        print(json.dumps(s, indent=2))
        return 0
    champ = s["champion"]
    print(f"run dir        {d}")
    print(f"cycles done    {s['cycle']}")
    print(f"lifetime rows  {s['lifetime_rows']:,}")
    print(f"generations    {s['generations']}")
    if champ:
        print(f"champion       gen {champ['generation']} (cycle {champ['cycle']}, ladder Elo {champ['elo']:+.0f}) {champ['onnx']}")
    ip = s["in_progress"]
    if ip:
        print(f"in progress    cycle {ip['cycle']} done={ip['done']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from training.pipeline.report import write_report

    d = _run_dir(args)
    out = args.out or d / "report.html"
    write_report(d, out)
    print(f"wrote {out}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from training.pipeline.report import estimate

    d = _run_dir(args)
    est = estimate(d, cycles=args.cycles, hourly_usd=args.hourly_usd)
    print(json.dumps(est, indent=2) if args.json else est["text"])
    return 0


def cmd_plan_sprt(args: argparse.Namespace) -> int:
    """Exact operating characteristics of the configured gate (no simulation)."""
    from training.pipeline.gate_oc import sprt_gate_oc, v3_gate_oc

    cfg = load_config(args.config, args.overrides).gating
    lo, hi = stats.sprt_bounds(cfg.alpha, cfg.beta)
    print(f"SPRT elo0={cfg.elo0:g} elo1={cfg.elo1:g} alpha={cfg.alpha:g} beta={cfg.beta:g} "
          f"LLR bounds [{lo:+.2f}, {hi:+.2f}] batch={cfg.batch_games} cap={cfg.max_games} "
          f"fallback: score>={cfg.promote_win:g} and Wilson low>0.5")
    print("computing exact operating characteristics...", flush=True)
    elos = [float(e) for e in args.elos] if args.elos else [-100, -50, -20, 0, 20, 35, 50, 100, 200]
    print(f"\n{'true Elo':>9} {'promoted':>9} {'E[games]':>9}" + (f" {'v3 promoted':>12} {'v3 E[games]':>12}" if args.compare else ""))
    rows = []
    for elo in elos:
        oc = sprt_gate_oc(cfg, elo)
        row = f"{elo:>9g} {oc.accept:>9.3f} {oc.games:>9.0f}"
        if args.compare:
            v3 = v3_gate_oc(elo)
            row += f" {v3.accept:>12.3f} {v3.games:>12.0f}"
        rows.append((elo, oc))
        print(row, flush=True)
    zero = next((oc for e, oc in rows if e == 0), None)
    if zero:
        print(f"\nfalse promotion at no real gain: {zero.accept:.1%} (costing {zero.games:.0f} games)")
    detect = next((e for e, oc in rows if e > 0 and oc.accept >= 0.8), None)
    print("smallest listed gain promoted at least 80% of the time: "
          + (f"{detect:+.0f} Elo" if detect is not None else "none in this range -- raise max_games or lower elo1"))
    return 0


def _pipeline(args: argparse.Namespace) -> Pipeline:
    cfg = load_config(args.config, args.overrides)
    return Pipeline(cfg, SubprocessExecutor(REPO_ROOT), REPO_ROOT)


def cmd_publish(args: argparse.Namespace) -> int:
    from training.pipeline.publish import load_index, retry_releases

    pipe = _pipeline(args)
    if args.retry:
        done = retry_releases(pipe)
        print(f"released: {done or 'nothing pending'}")
    index = load_index(pipe._abs(Path(pipe.cfg.publish.registry)))
    print(f"best: {index.get('best')}  previous_best: {index.get('previous_best')}")
    for e in index.get("entries", []):
        print(f"  {e['key']:<16} {e['status']:<10} elo {e.get('ladder_elo', 0):+6.0f}  {e.get('release_url') or e.get('release_error', '')}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    from training.pipeline.publish import rollback

    res = rollback(_pipeline(args), args.to, champion=args.champion)
    print(json.dumps(res, indent=2))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    print(dump_toml(load_config(args.config, args.overrides)))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the training loop")
    _add_config_args(r)
    r.add_argument("--cycles", type=int, default=None, help="stop after N cycles (overrides run.max_cycles)")
    r.add_argument("--no-build", action="store_true", help="skip go build (binary must exist)")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="show run state")
    _add_config_args(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    rep = sub.add_parser("report", help="write HTML dashboard")
    _add_config_args(rep)
    rep.add_argument("--out", type=Path, default=None)
    rep.set_defaults(fn=cmd_report)

    e = sub.add_parser("estimate", help="project time/cost from measured timings")
    _add_config_args(e)
    e.add_argument("--cycles", type=int, default=100)
    e.add_argument("--hourly-usd", type=float, default=0.0, help="instance price per hour (0 = time only)")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_estimate)

    ps = sub.add_parser("plan-sprt", help="exact promotion rate and gate length for the configured SPRT")
    _add_config_args(ps)
    ps.add_argument("--compare", action="store_true", help="also show the v3 gate (200 games, Wilson, interim stops)")
    ps.add_argument("--elos", nargs="*", type=float, help="true Elo gaps to evaluate (default: -100..200)")
    ps.set_defaults(fn=cmd_plan_sprt)

    pub = sub.add_parser("publish", help="list published champions; --retry failed GitHub releases")
    _add_config_args(pub)
    pub.add_argument("--retry", action="store_true")
    pub.set_defaults(fn=cmd_publish)

    rb = sub.add_parser("rollback", help="point 'best' back at an earlier published generation")
    _add_config_args(rb)
    rb.add_argument("--to", type=int, required=True, help="generation number")
    rb.add_argument("--champion", action="store_true", help="also resume self-play/training from that generation")
    rb.set_defaults(fn=cmd_rollback)

    c = sub.add_parser("config", help="print resolved config")
    _add_config_args(c)
    c.set_defaults(fn=cmd_config)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
