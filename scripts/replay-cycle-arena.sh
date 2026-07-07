#!/usr/bin/env bash
# Replay arena with unified komi (validation). Usage: bash scripts/replay-cycle-arena.sh [games] [seed]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/go/bin:${PATH:-}"

GAMES="${1:-50}"
SEED="${2:-66}"
OUT="/tmp/arena-replay.json"

pkill -f inference_server.py 2>/dev/null || true
sleep 1
if [[ -f .venv311/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv311/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python training/inference_server.py --model models/gofer-9x9-best.onnx --port 8080 &
python training/inference_server.py --model models/gofer-9x9-candidate.onnx --port 8081 &
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:8080/health" >/dev/null && curl -sf "http://127.0.0.1:8081/health" >/dev/null && break
  sleep 1
done

go build -o bin/gofer ./cmd/gofer
./bin/gofer -eval-backend sidecar -arena -games "$GAMES" -size 9 -komi 6.5 -playouts 200 \
  -black-eval onnx -white-eval onnx2 \
  -onnx-url "http://127.0.0.1:8080" -onnx-url-2 "http://127.0.0.1:8081" \
  -arena-parallel 2 -arena-opening-moves 8 -seed "$SEED" -arena-enhanced none \
  -json "$OUT"

pkill -f inference_server.py 2>/dev/null || true

python3 - "$OUT" <<'PY'
import json, sys
new = json.load(open(sys.argv[1]))
old_path = ".tectonix/reports/arena-cycle-24.json"
try:
    old = json.load(open(old_path))
    print(f"original_c24 stone b={old['wins_black']} w={old['wins_white']} wr_ch={old['win_rate_challenger']:.3f} n={old.get('game_count')}")
except FileNotFoundError:
    print("no arena-cycle-24.json")
print(f"replay_k65 stone b={new['wins_black']} w={new['wins_white']} wr_ch={new['win_rate_challenger']:.3f} n={new.get('game_count')}")
PY
