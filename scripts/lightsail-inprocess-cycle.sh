#!/usr/bin/env bash
# One training cycle on Lightsail with in-process ORT (no Python sidecar).
# Run on the instance after pulling code that includes Step 3 wiring.
#
# From laptop (SSH + run remotely):
#   bash scripts/aws-run-arena.sh IP inprocess-cycle
#
# On the instance directly:
#   bash scripts/lightsail-inprocess-cycle.sh
#
# Env overrides (all optional):
#   SKIP_PARITY=1          skip scripts/parity-onnx.sh preflight
#   NEW_SELFPLAY_PER_CYCLE=50   ARENA_GAMES=50  PARALLEL=2
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$ROOT"

export EVAL_BACKEND=inprocess
export MAX_CYCLES=1
export WEEK_DAYS="${WEEK_DAYS:-14}"
export PATH="/usr/local/go/bin:${PATH:-}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Production train-loop uses PARALLEL=2 on the Lightsail box; override via env if needed.
export PARALLEL="${PARALLEL:-2}"
export NEW_SELFPLAY_PER_CYCLE="${NEW_SELFPLAY_PER_CYCLE:-50}"
export ARENA_GAMES="${ARENA_GAMES:-50}"

RSS_LOG=".tectonix/reports/inprocess-rss.log"
mkdir -p .tectonix/reports

log() { echo "[$(date -Is)] $*" | tee -a "$RSS_LOG"; }

sample_rss() {
  local pid
  pid="$(pgrep -n gofer 2>/dev/null || true)"
  if [[ -n "$pid" ]]; then
    ps -o pid=,rss=,vsz= -p "$pid" 2>/dev/null | awk -v t="$(date -Is)" '{printf "%s pid=%s rss_kb=%s vsz_kb=%s\n", t, $1, $2, $3}' >>"$RSS_LOG"
  fi
}

(
  while true; do
    sample_rss
    sleep 10
  done
) &
RSS_PID=$!
trap 'kill "$RSS_PID" 2>/dev/null || true' EXIT

log "=== in-process cycle start EVAL_BACKEND=$EVAL_BACKEND MAX_CYCLES=$MAX_CYCLES ==="
log "go: $(go version 2>/dev/null || echo missing)"
log "python: $(python3 --version 2>/dev/null || echo missing)"

# Sidecar path needs Python ORT in .venv; in-process only needs PyTorch for training.
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ "${SKIP_PARITY:-0}" != "1" ]]; then
  log "parity preflight (500 positions)"
  if [[ -f .venv311/bin/activate ]]; then
    # ORT 1.26.0 requires Python >=3.11; system .venv on Lightsail is 3.10.
    # shellcheck disable=SC1091
    source .venv311/bin/activate
  fi
  GOFER_PARITY_LIMIT=500 bash scripts/parity-onnx.sh 2>&1 | tee -a "$RSS_LOG"
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
fi

log "train-loop-v3 one cycle selfplay=$NEW_SELFPLAY_PER_CYCLE arena=$ARENA_GAMES parallel=$PARALLEL"
bash scripts/train-loop-v3.sh 2>&1 | tee -a train-inprocess-cycle.log

sample_rss
log "=== done; rss log: $RSS_LOG train log: train-inprocess-cycle.log ==="
log "arena summary:"
grep -E 'PROMOTE|REJECT|win_rate_challenger|arena.*games|MAX_CYCLES' train-inprocess-cycle.log 2>/dev/null | tail -10 || true
