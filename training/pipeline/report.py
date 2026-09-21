"""Self-contained HTML dashboard + time/cost projection for a pipeline run.

Reads only <run_dir>/state.json, history/*.json and events.jsonl, so it works
on a laptop against a run dir rsync'd back from a rented GPU box.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from training.pipeline.state import STAGES, load_state, read_events

W, H = 720, 260
PAD_L, PAD_R, PAD_T, PAD_B = 52, 16, 16, 34
STAGE_VARS = {s: f"--series-{i + 1}" for i, s in enumerate(STAGES)}


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    hist = []
    for p in sorted((run_dir / "history").glob("cycle-*.json")):
        hist.append(json.loads(p.read_text(encoding="utf-8")))
    return hist


def stage_seconds(record: dict[str, Any]) -> dict[str, float]:
    return {s: float(record.get(f"{s}_seconds", 0.0) or 0.0) for s in STAGES}


# ------------------------------------------------------------------ estimate

def estimate(run_dir: Path, *, cycles: int, hourly_usd: float = 0.0, recent: int = 5) -> dict[str, Any]:
    """Project wall-clock and cost from the mean of the most recent trained cycles."""
    hist = [h for h in load_history(run_dir) if not h.get("skipped")]
    trained = [h for h in hist if h.get("train_seconds")] or hist
    sample = trained[-recent:]
    if not sample:
        return {"text": f"no completed cycles in {run_dir} yet; run at least one cycle first", "cycles_measured": 0}
    per_stage = {s: sum(stage_seconds(h)[s] for h in sample) / len(sample) for s in STAGES}
    per_cycle = sum(per_stage.values())
    hours = per_cycle * cycles / 3600
    out = {
        "cycles_measured": len(sample),
        "seconds_per_cycle": round(per_cycle, 1),
        "per_stage_seconds": {k: round(v, 1) for k, v in per_stage.items()},
        "cycles": cycles,
        "hours": round(hours, 2),
        "usd": round(hours * hourly_usd, 2) if hourly_usd else None,
    }
    lines = [f"measured over last {len(sample)} cycle(s): {per_cycle / 60:.1f} min/cycle"]
    for s, v in per_stage.items():
        share = v / per_cycle * 100 if per_cycle else 0
        lines.append(f"  {s:<9} {v / 60:7.1f} min  ({share:4.1f}%)")
    lines.append(f"{cycles} cycles -> {hours:.1f} h" + (f" -> ${hours * hourly_usd:,.2f} at ${hourly_usd}/h" if hourly_usd else ""))
    out["text"] = "\n".join(lines)
    return out


# -------------------------------------------------------------------- charts

def _scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) * (b - a) / (hi - lo)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi == lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** len(str(int(abs(raw)))) / 10 if raw >= 1 else 0.1
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = (lo // step) * step
    ticks, t = [], start
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            ticks.append(round(t, 6))
        t += step
    return ticks


def _axes(xs: list[int], y_lo: float, y_hi: float, y_fmt, x_label: str) -> str:
    parts = []
    for t in _nice_ticks(y_lo, y_hi):
        y = _scale(t, y_lo, y_hi, H - PAD_B, PAD_T)
        parts.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end">{y_fmt(t)}</text>')
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{H - PAD_B}" y2="{H - PAD_B}"/>')
    step = max(1, len(xs) // 10)
    x_lo, x_hi = (min(xs), max(xs)) if xs else (0, 1)
    for x in xs[::step]:
        px = _scale(x, x_lo - 0.5, x_hi + 0.5, PAD_L, W - PAD_R)
        parts.append(f'<text class="tick" x="{px:.1f}" y="{H - PAD_B + 16}" text-anchor="middle">{x}</text>')
    parts.append(f'<text class="tick" x="{(PAD_L + W - PAD_R) / 2}" y="{H - 4}" text-anchor="middle">{x_label}</text>')
    return "".join(parts)


def _svg(body: str, label: str) -> str:
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(label)}">{body}</svg>'


def elo_chart(gens: list[dict[str, Any]]) -> str:
    if not gens:
        return '<p class="empty">No champion yet.</p>'
    xs = [g["generation"] for g in gens]
    ys = [g["elo"] for g in gens]
    lo, hi = min(ys + [0]), max(ys + [0])
    pad = max(10.0, (hi - lo) * 0.1)
    lo, hi = lo - pad, hi + pad
    pts = [(_scale(x, min(xs) - 0.5, max(xs) + 0.5, PAD_L, W - PAD_R), _scale(y, lo, hi, H - PAD_B, PAD_T)) for x, y in zip(xs, ys)]
    body = _axes(xs, lo, hi, lambda t: f"{t:+.0f}", "generation")
    body += '<polyline class="line s1" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>'
    for (px, py), g in zip(pts, gens):
        tip = f"Gen {g['generation']} (cycle {g['cycle']})|Ladder Elo {g['elo']:+.0f}"
        body += f'<circle class="dot s1" cx="{px:.1f}" cy="{py:.1f}" r="4"/>'
        body += f'<circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="12" data-tip="{html.escape(tip)}"/>'
    return _svg(body, "Ladder Elo by champion generation")


def anchor_chart(hist: list[dict[str, Any]]) -> str:
    """Absolute strength: every gate that measured the candidate against the
    heuristic rather than against the champion.

    The ladder chart above cannot answer whether the loop is going anywhere,
    because every point on it is measured against the point before it. This one
    is measured against something that never moves. **Zero is parity with the
    hand-written evaluator** -- the line the nets have to cross for any of this
    to be worth running.
    """
    rows = [(h["cycle"], h["gate"]["vs_heuristic_elo"]) for h in hist
            if isinstance(h.get("gate"), dict) and h["gate"].get("vs_heuristic_elo") is not None]
    if not rows:
        return '<p class="empty">No absolute measurement yet: set <code>gating.anchor_every</code>.</p>'
    xs = [c for c, _ in rows]
    ys = [e for _, e in rows]
    lo, hi = min(ys + [0.0]), max(ys + [0.0])
    pad = max(10.0, (hi - lo) * 0.1)
    lo, hi = lo - pad, hi + pad
    body = _axes(xs, lo, hi, lambda t: f"{t:+.0f}", "cycle")
    zero = _scale(0.0, lo, hi, H - PAD_B, PAD_T)
    body += (f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{zero:.1f}" y2="{zero:.1f}" stroke-dasharray="5 4"/>'
             f'<text class="tick" x="{W - PAD_R}" y="{zero - 6:.1f}" text-anchor="end">heuristic</text>')
    pts = [(_scale(x, min(xs) - 0.5, max(xs) + 0.5, PAD_L, W - PAD_R), _scale(y, lo, hi, H - PAD_B, PAD_T))
           for x, y in zip(xs, ys)]
    if len(pts) > 1:
        body += '<polyline class="line s2" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>'
    for (px, py), (cycle, elo) in zip(pts, rows):
        tip = f"Cycle {cycle}|{elo:+.0f} Elo vs heuristic"
        body += f'<circle class="dot s2" cx="{px:.1f}" cy="{py:.1f}" r="4"/>'
        body += f'<circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="12" data-tip="{html.escape(tip)}"/>'
    return _svg(body, "Elo vs heuristic by cycle")


def gate_chart(hist: list[dict[str, Any]]) -> str:
    rows = [(h["cycle"], h["gate"]) for h in hist if isinstance(h.get("gate"), dict) and h["gate"].get("kind") == "sprt"]
    if not rows:
        return '<p class="empty">No head-to-head gates yet (the first network seeds the lineage).</p>'
    xs = [c for c, _ in rows]
    lo, hi = 0.0, 1.0
    body = _axes(xs, lo, hi, lambda t: f"{t:.0%}", "cycle")
    y50 = _scale(0.5, lo, hi, H - PAD_B, PAD_T)
    body += f'<line class="ref" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y50:.1f}" y2="{y50:.1f}"/>'
    body += f'<text class="tick" x="{W - PAD_R}" y="{y50 - 5:.1f}" text-anchor="end">50% = equal strength</text>'
    for c, gt in rows:
        px = _scale(c, min(xs) - 0.5, max(xs) + 0.5, PAD_L, W - PAD_R)
        y = _scale(gt.get("score", 0.5), lo, hi, H - PAD_B, PAD_T)
        yl = _scale(gt.get("wilson_low", 0.0), lo, hi, H - PAD_B, PAD_T)
        yh = _scale(gt.get("wilson_high", 1.0), lo, hi, H - PAD_B, PAD_T)
        promoted = bool(gt.get("promote"))
        body += f'<line class="whisker" x1="{px:.1f}" x2="{px:.1f}" y1="{yl:.1f}" y2="{yh:.1f}"/>'
        body += f'<circle class="dot s1 {"filled" if promoted else "hollow"}" cx="{px:.1f}" cy="{y:.1f}" r="5"/>'
        tip = (f"Cycle {c}: {'promoted' if promoted else 'rejected'}|Score {gt.get('score', 0):.1%} over {gt.get('games', 0)} games"
               f"|95% CI {gt.get('wilson_low', 0):.1%} to {gt.get('wilson_high', 1):.1%}|{gt.get('verdict', '')}, LLR {gt.get('llr', 0):+.2f}")
        body += f'<rect class="hit" x="{px - 10:.1f}" y="{PAD_T}" width="20" height="{H - PAD_B - PAD_T}" data-tip="{html.escape(tip)}"/>'
    return _svg(body, "Candidate score against champion per cycle")


def stage_chart(hist: list[dict[str, Any]]) -> str:
    if not hist:
        return '<p class="empty">No completed cycles yet.</p>'
    xs = [h["cycle"] for h in hist]
    mins = [{s: v / 60 for s, v in stage_seconds(h).items()} for h in hist]
    hi = max(sum(m.values()) for m in mins) or 1.0
    body = _axes(xs, 0.0, hi * 1.05, lambda t: f"{t:g}m", "cycle")
    slot = (W - PAD_L - PAD_R) / (len(xs) + 1)
    bw = max(4.0, min(28.0, slot * 0.6))
    base_y = H - PAD_B
    for x, m in zip(xs, mins):
        px = _scale(x, min(xs) - 0.5, max(xs) + 0.5, PAD_L, W - PAD_R)
        acc = 0.0
        segs = [(s, v) for s, v in m.items() if v > 0]
        for i, (s, v) in enumerate(segs):
            y0 = _scale(acc, 0.0, hi * 1.05, base_y, PAD_T)
            y1 = _scale(acc + v, 0.0, hi * 1.05, base_y, PAD_T)
            h_px = max(0.0, y0 - y1 - (2 if i else 0))  # 2px surface gap between stacked segments
            top = i == len(segs) - 1
            rx = ' rx="4"' if top and h_px > 8 else ""
            body += f'<rect class="bar" style="fill:var({STAGE_VARS[s]})" x="{px - bw / 2:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{h_px:.1f}"{rx}/>'
            acc += v
        tip = f"Cycle {x}: {sum(m.values()):.1f} min|" + "|".join(f"{s} {v:.1f} min" for s, v in m.items() if v > 0)
        body += f'<rect class="hit" x="{px - slot / 2:.1f}" y="{PAD_T}" width="{slot:.1f}" height="{base_y - PAD_T}" data-tip="{html.escape(tip)}"/>'
    return _svg(body, "Wall-clock minutes per cycle by stage")


def _legend_stages() -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var({STAGE_VARS[s]})"></i>{s}</span>' for s in STAGES) + "</div>"


def _table(hist: list[dict[str, Any]]) -> str:
    head = "<tr><th>Cycle</th><th>Rows</th><th>Window</th><th>Gate</th><th>Score</th><th>Games</th><th>Result</th><th>Minutes</th></tr>"
    body = []
    for h in reversed(hist):
        gt = h.get("gate") if isinstance(h.get("gate"), dict) else {}
        result = "skipped" if h.get("skipped") else ("promoted → gen %s" % h.get("generation") if h.get("promoted") else "rejected")
        score = gt.get("score") if gt.get("kind") == "sprt" else (gt.get("vs_heuristic") or {}).get("score")
        body.append(
            f"<tr><td>{h['cycle']}</td><td>{h.get('selfplay_rows', 0):,}</td><td>{h.get('window_rows', 0):,}</td>"
            f"<td>{html.escape(str(gt.get('kind', '-')))}</td><td>{'' if score is None else f'{score:.1%}'}</td>"
            f"<td>{gt.get('games', (gt.get('vs_heuristic') or {}).get('games', ''))}</td><td>{html.escape(result)}</td>"
            f"<td>{sum(stage_seconds(h).values()) / 60:.1f}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;--series-6:#008300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--text-primary);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:800px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 2px}
.sub{color:var(--text-secondary);margin:0 0 20px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.tile,.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile b{display:block;font-size:24px;font-weight:600}.tile span{color:var(--text-secondary);font-size:12px}
.card{margin-bottom:16px}.card p.note{color:var(--text-secondary);margin:0 0 8px;font-size:13px}svg{width:100%;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}.ref{stroke:var(--muted);stroke-dasharray:4 4}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.line{fill:none;stroke-width:2}.s1{stroke:var(--series-1)}
.dot.s1{fill:var(--series-1);stroke:var(--surface-1);stroke-width:2}.dot.hollow{fill:var(--surface-1);stroke:var(--series-1);stroke-width:2}
.whisker{stroke:var(--text-secondary);stroke-width:1.5}.hit{fill:transparent;cursor:crosshair}.hit:hover{fill:var(--text-primary);fill-opacity:.04}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--text-secondary);font-size:12px;margin:0 0 6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.legend i.hollow{border:2px solid var(--series-1);border-radius:50%;background:var(--surface-1)}.legend i.filled{background:var(--series-1);border-radius:50%}
.empty{color:var(--muted);margin:8px 0}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--text-secondary);font-weight:500}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);border-radius:8px;
padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;max-width:280px}#tip b{display:block;margin-bottom:2px}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
 el.addEventListener('mousemove',e=>{const [h,...r]=el.dataset.tip.split('|');tip.innerHTML='';const b=document.createElement('b');b.textContent=h;tip.append(b);
  r.forEach(t=>{const d=document.createElement('div');d.textContent=t;tip.append(d)});tip.style.display='block';
  const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px'});
 el.addEventListener('mouseleave',()=>tip.style.display='none');});
"""


def render(run_dir: Path) -> str:
    st = load_state(run_dir / "state.json")
    hist = load_history(run_dir)
    gens = [vars(g) for g in st.generations]
    champ = st.champion
    gated = [h for h in hist if isinstance(h.get("gate"), dict) and h["gate"].get("kind") == "sprt"]
    promoted = sum(1 for h in gated if h.get("promoted"))
    avg_games = sum(h["gate"].get("games", 0) for h in gated) / len(gated) if gated else 0
    hours = sum(sum(stage_seconds(h).values()) for h in hist) / 3600
    tiles = [
        (f"gen {champ.generation}" if champ else "—", "champion"),
        (f"{champ.elo:+.0f}" if champ else "—", "ladder Elo vs gen 1"),
        (str(st.cycle), "cycles completed"),
        (f"{st.lifetime_rows:,}", "self-play rows"),
        (f"{promoted}/{len(gated)}" if gated else "—", "gates passed"),
        (f"{avg_games:.0f}" if gated else "—", "avg games per gate"),
        (f"{hours:.1f} h", "wall-clock"),
    ]
    tiles_html = "".join(f'<div class="tile"><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>' for v, k in tiles)
    events = read_events(run_dir / "events.jsonl")
    last = events[-1]["ts"] if events else st.updated_at
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gofer Training Run</title><style>{CSS}</style></head><body><main>
<h1>Gofer training run: {html.escape(run_dir.name)}</h1><p class="sub">Last update {html.escape(last)}</p>
<div class="tiles">{tiles_html}</div>
<div class="card"><h2>Champion strength</h2><p class="note">Each promotion adds the Elo it measured against the previous champion.</p>{elo_chart(gens)}</div>
<div class="card"><h2>Strength vs the heuristic</h2><p class="note">Measured against a fixed opponent, so unlike the ladder above this can fall. Flat here while the ladder climbs means the gate is measuring drift.</p>{anchor_chart(hist)}</div>
<div class="card"><h2>Gate results</h2><p class="note">Candidate score against the current champion, with 95% Wilson interval. SPRT stops each gate as soon as the evidence is decisive.</p>
<div class="legend"><span><i class="filled"></i>promoted</span><span><i class="hollow"></i>rejected</span></div>{gate_chart(hist)}</div>
<div class="card"><h2>Time per cycle</h2><p class="note">Self-play shows only the time spent waiting when it overlaps the previous cycle.</p>{_legend_stages()}{stage_chart(hist)}</div>
<div class="card"><h2>Cycles</h2><div class="tablewrap">{_table(hist)}</div></div>
</main><div id="tip" role="tooltip"></div><script>{JS}</script></body></html>"""


def write_report(run_dir: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(run_dir), encoding="utf-8")
