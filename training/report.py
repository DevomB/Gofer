"""Summarize a training run: Elo by generation, losses, anchor results.

    python -m training.report runs/9x9 --out docs/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path):
    state = json.loads((run / "state.json").read_text())
    ratings = json.loads((run / "eval" / "ratings.json").read_text())
    metrics = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return state, ratings, metrics


def anchor_table(state: dict) -> dict[str, dict[str, tuple[int, int, int]]]:
    """{generation: {anchor: (wins, losses, draws)}} for non-generation opponents."""
    out: dict[str, dict[str, tuple[int, int, int]]] = {}
    for r in state.get("results", []):
        if not r["b"].startswith("gen-"):
            out.setdefault(r["a"], {})[r["b"]] = (r["wins"], r["losses"], r["draws"])
    return out


def plot(ratings: dict, metrics: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    gens = sorted((int(p.split("-")[1]), v) for p, v in ratings.items() if p.startswith("gen-"))
    anchors = {p: v for p, v in ratings.items() if not p.startswith("gen-")}
    train = [m for m in metrics if m.get("phase") == "train"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    x = [g for g, _ in gens]
    y = [v["elo"] for _, v in gens]
    e = [v["stderr"] for _, v in gens]
    ax1.errorbar(x, y, yerr=e, marker="o", ms=4, capsize=2, color="#3b6fd8", label="Gofer generation")
    ranked = sorted(anchors.items(), key=lambda kv: kv[1]["elo"])
    for i, (name, v) in enumerate(ranked):
        ax1.axhline(v["elo"], ls="--", lw=1, color="#888")
        # put a label under its line when the next anchor sits close above it
        crowded = i + 1 < len(ranked) and ranked[i + 1][1]["elo"] - v["elo"] < 150
        ax1.text(0.2, v["elo"], name, va="top" if crowded else "bottom", ha="left", fontsize=8, color="#555")
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax1.set_xlabel("generation")
    ax1.set_ylabel("Elo (gen 0 = 0)")
    ax1.set_title("Strength by generation")
    ax1.grid(alpha=0.3)
    if train:
        g = [m["gen"] for m in train]
        ax2.plot(g, [m.get("val_policy", float("nan")) for m in train], marker="o", ms=3, label="policy loss (new data, before training)")
        ax2.plot(g, [m.get("val_value", float("nan")) for m in train], marker="o", ms=3, label="value loss (new data, before training)")
        ax2.set_xlabel("generation")
        ax2.set_title("Held-out losses")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def markdown(state: dict, ratings: dict, metrics: list[dict]) -> str:
    sp = {m["gen"]: m for m in metrics if m.get("phase") == "selfplay"}
    tr = {m["gen"]: m for m in metrics if m.get("phase") == "train"}
    anchors = anchor_table(state)
    lines = [
        "| Gen | Elo | Self-play games/s | Avg moves | Rows | Policy loss* | Value loss* | Anchors (W-L) |",
        "|----:|----:|------------------:|----------:|-----:|-------------:|------------:|:--------------|",
    ]
    for gen in sorted({int(p.split("-")[1]) for p in ratings if p.startswith("gen-")}):
        name = f"gen-{gen:04d}"
        r = ratings[name]
        s = sp.get(gen, {})
        t = tr.get(gen, {})
        a = ", ".join(f"{k} {w}-{l}" for k, (w, l, _) in sorted(anchors.get(name, {}).items()))
        moves = s.get("moves", 0) / max(s.get("games", 1), 1)
        lines.append(
            f"| {gen} | {r['elo']:+.0f} ± {r['stderr']:.0f} | {s.get('games_per_second', 0):.2f} | {moves:.0f} | "
            f"{s.get('rows', 0):,} | {t.get('val_policy', float('nan')):.3f} | {t.get('val_value', float('nan')):.3f} | {a} |"
        )
    others = {p: v for p, v in ratings.items() if not p.startswith("gen-")}
    if others:
        lines.append("")
        lines.append("Anchors: " + ", ".join(f"{p} {v['elo']:+.0f} ± {v['stderr']:.0f}" for p, v in sorted(others.items())))
    lines.append("")
    lines.append("\\* loss of that generation's network on the self-play data it generated, measured before training on it.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="directory for elo.png and summary.md")
    args = ap.parse_args()
    state, ratings, metrics = load(args.run)
    md = markdown(state, ratings, metrics)
    print(md)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.md").write_text(md + "\n")
        plot(ratings, metrics, args.out / "elo.png")


if __name__ == "__main__":
    main()
