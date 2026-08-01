#!/usr/bin/env bash
# Equal-strength ONNX arena probe (same weights, two sidecars).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CHAMP=8080
CHAL=8081
REPORT_DIR=".tectonix/reports/arena-bench"
json="${REPORT_DIR}/identical-onnx.json"
mkdir -p "$REPORT_DIR"

pkill -f inference_server.py 2>/dev/null || true
sleep 1
python training/inference_server.py --model models/gofer-9x9-best.onnx --port "$CHAMP" &
python training/inference_server.py --model models/gofer-9x9-candidate.onnx --port "$CHAL" &
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${CHAMP}/health" >/dev/null && curl -sf "http://127.0.0.1:${CHAL}/health" >/dev/null && break
  sleep 1
done

t0=$(date +%s.%N)
bin/gofer -arena -games 200 -size 9 -playouts 400 -black-eval onnx -white-eval onnx2 \
  -onnx-url "http://127.0.0.1:${CHAMP}" -onnx-url-2 "http://127.0.0.1:${CHAL}" \
  -arena-parallel 8 -arena-opening-moves 8 -eval-timeout 2s -arena-enhanced none \
  -seed 99 -json "$json"
t1=$(date +%s.%N)
pkill -f inference_server.py 2>/dev/null || true

python3 - "$json" "$t0" "$t1" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
sec = round(float(sys.argv[3]) - float(sys.argv[2]), 1)
n = d["game_count"]
bw, ww, dr = d["wins_black"], d["wins_white"], d["draws"]
rate, lo, hi = d["win_rate_challenger"], d["wilson_ci_low"], d["wilson_ci_high"]
dec = "promote" if rate >= 0.55 and lo > 0.5 else "reject"
onnx_w = onnx2_w = 0
for g in d["games"]:
    if g["black_wins"]:
        if g["black_eval"] == "onnx":
            onnx_w += 1
        else:
            onnx2_w += 1
    elif g["white_wins"]:
        if g["white_eval"] == "onnx":
            onnx_w += 1
        else:
            onnx2_w += 1
print(
    f"identical-onnx|n={n}|black={bw}({bw/n:.3f})|white={ww}({ww/n:.3f})|draws={dr}|"
    f"onnx_w={onnx_w}|onnx2_w={onnx2_w}|challenger={rate:.3f}|wilson=[{lo:.3f},{hi:.3f}]|{dec}|sec={sec}"
)
PY
