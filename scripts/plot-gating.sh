#!/usr/bin/env bash
# Plot cumulative baseline wins from a gofer arena JSON report.
# Usage: ./scripts/plot-gating.sh [.tectonix/reports/arena-9x9-baseline.json] [out.png]
set -euo pipefail

JSON="${1:-.tectonix/reports/arena-9x9-baseline.json}"
OUT="${2:-.tectonix/reports/gating-cumulative.png}"
CSV="${OUT%.png}.csv"

if [[ ! -f "$JSON" ]]; then
  echo "missing arena JSON: $JSON" >&2
  exit 1
fi

python3 - "$JSON" "$CSV" <<'PY'
import json, sys
path, csv_path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
games = data.get("games") or []
baseline = data.get("baseline_eval", "")
cum = 0
rows = ["game,cumulative_baseline_wins,win_rate"]
for g in games:
    won = False
    if g.get("black_eval") == baseline and g.get("black_wins"):
        won = True
    if g.get("white_eval") == baseline and g.get("white_wins"):
        won = True
    if won:
        cum += 1
    n = g.get("game", len(rows))
    rows.append(f"{n},{cum},{cum/n:.6f}")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
PY

if command -v gnuplot >/dev/null 2>&1; then
  gnuplot -e "set terminal png size 800,480; set output '$OUT'; set xlabel 'game'; set ylabel 'cumulative baseline wins'; set title 'Arena gating cumulative'; plot '$CSV' using 1:2 with lines lw 2"
  echo "wrote $OUT"
else
  echo "gnuplot not found; wrote CSV only: $CSV"
fi
