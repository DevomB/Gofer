#!/usr/bin/env bash
# Smoke pipeline: fetch SGF → convert → validate → short pretrain.
# Does not touch training/state/best.pt or Lightsail.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="training/experiments/warmstart/data"
RAW_DIR="training/experiments/warmstart/raw"
OUT="${DATA_DIR}/supervised-v1-smoke.jsonl"
CKPT_DIR="training/experiments/warmstart/checkpoints/smoke"
GOFER="bin/gofer"
MAX_ROWS="${MAX_ROWS:-100000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-3}"

mkdir -p "$DATA_DIR" "$CKPT_DIR" bin

log() { echo "[warmstart-smoke] $*"; }

if [[ ! -x "$GOFER" ]]; then
  log "building gofer"
  go build -o "$GOFER" ./cmd/gofer
fi

bash scripts/fetch-warmstart-smoke-data.sh

log "converting SGF (max ${MAX_ROWS} rows, equal per-source cap)"
AEB_9X9="${RAW_DIR}/aeb/games/other_sizes/9x9"
if [[ ! -d "$AEB_9X9" ]]; then
  log "aeb 9x9 dir missing — run fetch-warmstart-smoke-data.sh first"
  exit 1
fi
"$GOFER" -convert-sgf -size 9 -convert-epsilon 0.10 -convert-max-rows "$MAX_ROWS" \
  -o "$OUT" \
  "$RAW_DIR/cgos" "$RAW_DIR/minigo" "$AEB_9X9"

log "validating JSONL"
python training/experiments/warmstart/validate_smoke.py "$OUT"
"$GOFER" -verify-jsonl -o "$OUT"

log "short pretrain (${PRETRAIN_EPOCHS} epochs)"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python training/train_bootstrap.py \
  --data "$OUT" \
  --epochs "$PRETRAIN_EPOCHS" \
  --lr 0.01 \
  --out-dir "$CKPT_DIR" \
  --val-split 0.1

log "smoke complete: $OUT"
