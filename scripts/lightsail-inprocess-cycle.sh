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
#   NEW_SELFPLAY_PER_CYCLE=200  ARENA_GAMES=200  PARALLEL=8
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$ROOT"

export EVAL_BACKEND=inprocess
export PATH="/usr/local/go/bin:${PATH:-}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

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

log "=== in-process cycle start EVAL_BACKEND=$EVAL_BACKEND ==="
log "go: $(go version 2>/dev/null || echo missing)"
log "python: $(python3 --version 2>/dev/null || echo missing)"

pip install -q -r training/requirements.txt

if [[ "${SKIP_PARITY:-0}" != "1" ]]; then
  log "parity preflight (500 positions)"
  GOFER_PARITY_LIMIT=500 bash scripts/parity-onnx.sh 2>&1 | tee -a "$RSS_LOG"
fi

# WEEK_DAYS=0: train-loop-v3 deadline is now; runs one cycle then exits.
export WEEK_DAYS=0
export NEW_SELFPLAY_PER_CYCLE="${NEW_SELFPLAY_PER_CYCLE:-200}"
export ARENA_GAMES="${ARENA_GAMES:-200}"
export PARALLEL="${PARALLEL:-8}"

log "train-loop-v3 one cycle selfplay=$NEW_SELFPLAY_PER_CYCLE arena=$ARENA_GAMES parallel=$PARALLEL"
bash scripts/train-loop-v3.sh 2>&1 | tee -a train-inprocess-cycle.log

sample_rss
log "=== done; rss log: $RSS_LOG train log: train-inprocess-cycle.log ==="
log "arena summary:"
grep -E 'PROMOTE|REJECT|win_rate_challenger|arena.*games' train-inprocess-cycle.log 2>/dev/null | tail -10 || true
