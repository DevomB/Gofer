#!/usr/bin/env bash
# Run standard arena scenarios and emit per-cycle metrics.
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$ROOT"

GOFER_BIN="${GOFER_BIN:-bin/gofer}"
ARENA_GAMES="${ARENA_GAMES:-200}"
ARENA_PLAYOUTS="${ARENA_PLAYOUTS:-400}"
ARENA_OPENING_MOVES="${ARENA_OPENING_MOVES:-8}"
PARALLEL="${PARALLEL:-$(nproc 2>/dev/null || echo 8)}"
REPORT_DIR=".tectonix/reports/arena-bench"
PROMOTE_WIN="${PROMOTE_WIN:-0.55}"

mkdir -p "$REPORT_DIR" models bin
export PATH="/usr/local/go/bin:${PATH:-}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

go build -o "$GOFER_BIN" ./cmd/gofer

CHAMP_PORT=8080
CHALLENGER_PORT=8081
SIDECAR_PIDS=()

start_sidecar() {
  local model="$1" port="$2"
  python training/inference_server.py --model "$model" --port "$port" &
  SIDECAR_PIDS+=($!)
  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then return 0; fi
    sleep 1
  done
  echo "sidecar on port ${port} failed" >&2
  return 1
}

stop_sidecars() {
  pkill -f 'inference_server.py' 2>/dev/null || true
  for pid in "${SIDECAR_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  SIDECAR_PIDS=()
}
trap stop_sidecars EXIT

if [[ ! -f models/gofer-9x9-bootstrap.onnx ]]; then
  python training/export_onnx.py --out models/gofer-9x9-bootstrap.onnx
fi
cp -f models/gofer-9x9-bootstrap.onnx models/gofer-9x9-candidate.onnx
cp -f models/gofer-9x9-bootstrap.onnx models/gofer-9x9-best.onnx

# name|black|white|onnx_url|onnx_url_2|seed|notes
SCENARIOS=(
  "heuristic-onnx|heuristic|onnx|http://127.0.0.1:${CHAMP_PORT}||42|bootstrap sanity"
  "uniform-heuristic|uniform|heuristic||||7|lopsided reject"
  "heuristic-uniform|heuristic|uniform||||11|lopsided reject alt"
  "identical-onnx|onnx|onnx2|http://127.0.0.1:${CHAMP_PORT}|http://127.0.0.1:${CHALLENGER_PORT}|99|equal net-vs-net"
  "heuristic-onnx-repeat|heuristic|onnx|http://127.0.0.1:${CHAMP_PORT}||123|bootstrap repeat seed"
)

SUMMARY="${REPORT_DIR}/summary.jsonl"
: >"$SUMMARY"

echo "arena bench: games=$ARENA_GAMES playouts=$ARENA_PLAYOUTS parallel=$PARALLEL"
echo "cycle_id|game_count|win_rate|wilson_low|wilson_high|promoted|arena_sec|path|notes"

for spec in "${SCENARIOS[@]}"; do
  IFS='|' read -r cid black white url1 url2 seed notes <<<"$spec"
  json="${REPORT_DIR}/arena-${cid}.json"
  stop_sidecars

  needs_sidecar=0
  [[ "$black" == onnx* || "$white" == onnx* ]] && needs_sidecar=1
  if [[ "$needs_sidecar" == "1" ]]; then
    start_sidecar models/gofer-9x9-best.onnx "$CHAMP_PORT"
    if [[ "$black" == onnx2 || "$white" == onnx2 ]]; then
      start_sidecar models/gofer-9x9-candidate.onnx "$CHALLENGER_PORT"
    fi
  fi

  arena_args=(
    -arena -games "$ARENA_GAMES" -size 9 -playouts "$ARENA_PLAYOUTS"
    -black-eval "$black" -white-eval "$white"
    -arena-parallel "$PARALLEL" -arena-opening-moves "$ARENA_OPENING_MOVES"
    -eval-timeout 2s -arena-enhanced none -seed "$seed" -json "$json"
  )
  [[ -n "$url1" ]] && arena_args+=(-onnx-url "$url1")
  [[ -n "$url2" ]] && arena_args+=(-onnx-url-2 "$url2")

  t0=$(date +%s.%N)
  "./$GOFER_BIN" "${arena_args[@]}" 2>"${REPORT_DIR}/${cid}.log" | tee -a "${REPORT_DIR}/${cid}.log"
  t1=$(date +%s.%N)
  arena_sec=$(python3 -c "print(round(float('$t1')-float('$t0'), 1))")

  python3 - "$json" "$cid" "$arena_sec" "$notes" "$SUMMARY" "$PROMOTE_WIN" <<'PY'
import json, sys
from pathlib import Path

report_path, cid, arena_sec, notes, summary_path, promote_win = sys.argv[1:7]
d = json.loads(Path(report_path).read_text())
gc = int(d.get("game_count", 0))
rate = float(d.get("win_rate_challenger", 0))
lo = float(d.get("wilson_ci_low", 0))
hi = float(d.get("wilson_ci_high", 0))
go_promoted = bool(d.get("promoted", False))
py_promote = rate >= float(promote_win) and lo > 0.5
path = "lopsided" if gc < 200 else "close"
row = {
    "cycle_id": cid,
    "game_count": gc,
    "win_rate_challenger": rate,
    "wilson_ci_low": lo,
    "wilson_ci_high": hi,
    "go_promoted": go_promoted,
    "py_promote": py_promote,
    "decision": "promote" if py_promote else "reject",
    "arena_sec": float(arena_sec),
    "path": path,
    "notes": notes,
}
print(
    f"{cid}|{gc}|{rate:.3f}|{lo:.3f}|{hi:.3f}|"
    f"{'promote' if py_promote else 'reject'}|{arena_sec}|{path}|{notes}"
)
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")
if go_promoted != py_promote:
    print(f"WARN {cid}: go promoted={go_promoted} py promote={py_promote}", file=sys.stderr)
PY
done

echo ""
echo "Wrote $SUMMARY"
