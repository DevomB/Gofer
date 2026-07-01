#!/usr/bin/env bash
# Run on Lightsail: loop train → arena until win gate or deadline.
# Usage: WEEK_DAYS=7 bash scripts/weekly-train-loop.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WEEK_DAYS="${WEEK_DAYS:-7}"
WIN_TARGET="${WIN_TARGET:-0.75}"
ARENA_GAMES="${ARENA_GAMES:-200}"
START_EPOCHS="${TRAIN_EPOCHS:-25}"
SELFPLAY_BASE="${SELFPLAY_GAMES:-100}"
SELFPLAY_STEP="${SELFPLAY_STEP:-100}"
SELFPLAY_MAX="${SELFPLAY_MAX:-500}"

DEADLINE="$(date -d "+${WEEK_DAYS} days" +%s)"
mkdir -p .tectonix/reports/training-history training/data

export PATH="/usr/local/go/bin:${PATH:-}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

cycle=1
selfplay_n="$SELFPLAY_BASE"
best_rate=0

log() { echo "[$(date -Is)] $*"; }

start_sidecar() {
  pkill -f 'inference_server.py' 2>/dev/null || true
  python training/inference_server.py --model models/gofer-9x9-bootstrap.onnx --port 8080 &
  SIDECAR_PID=$!
  sleep 2
  curl -sf http://127.0.0.1:8080/health >/dev/null
}

stop_sidecar() {
  kill "${SIDECAR_PID:-}" 2>/dev/null || true
}

while [[ "$(date +%s)" -lt "$DEADLINE" ]]; do
  log "cycle $cycle selfplay_games=$selfplay_n"
  go run ./cmd/gofer -selfplay -games "$selfplay_n" -size 9 -playouts 100 \
    -eval heuristic -o "training/data/samples-cycle${cycle}.jsonl" -seed "$((42 + cycle))"

  python training/train_bootstrap.py \
    --data "training/data/samples-cycle${cycle}.jsonl" \
    --epochs "$START_EPOCHS" \
    --out-dir "training/checkpoints/cycle${cycle}"

  python training/export_onnx.py \
    --checkpoint "training/checkpoints/cycle${cycle}/best.pt" \
    --out models/gofer-9x9-bootstrap.onnx

  report=".tectonix/reports/arena-cycle-${cycle}.json"
  start_sidecar
  go run ./cmd/gofer -arena -games "$ARENA_GAMES" -size 9 -playouts 400 \
    -black-eval heuristic -white-eval onnx \
    -eval-timeout 2s -seed "$((42 + cycle))" -arena-enhanced none \
    -json "$report" | tee ".tectonix/reports/arena-cycle-${cycle}.log"
  stop_sidecar

  rate="$(python3 -c "import json; d=json.load(open('$report')); print(d.get('win_rate_challenger',0))")"
  log "cycle $cycle win_rate_challenger=$rate target=$WIN_TARGET"
  cp "$report" .tectonix/reports/arena-latest.json

  if python3 -c "import sys; sys.exit(0 if float('$rate') >= float('$WIN_TARGET') else 1)"; then
    cp "$report" .tectonix/reports/arena-9x9-onnx-v25.json
    log "gate passed; stopping loop"
    exit 0
  fi

  if python3 -c "import sys; sys.exit(0 if float('$rate') > float('$best_rate') else 1)" 2>/dev/null; then
    best_rate="$rate"
    cp models/gofer-9x9-bootstrap.onnx "models/gofer-9x9-best.onnx"
  fi

  selfplay_n=$((selfplay_n + SELFPLAY_STEP))
  if [[ "$selfplay_n" -gt "$SELFPLAY_MAX" ]]; then
    selfplay_n="$SELFPLAY_BASE"
  fi
  cycle=$((cycle + 1))
done

log "deadline reached after ${WEEK_DAYS} days; best win_rate=$best_rate"
cp .tectonix/reports/arena-latest.json .tectonix/reports/arena-9x9-onnx-v25.json 2>/dev/null || true

if [[ "${LIGHTSAIL_AUTO_DESTROY:-0}" == "1" ]]; then
  log "LIGHTSAIL_AUTO_DESTROY=1 — delete instance manually or via aws-run-arena.sh destroy"
fi
