#!/usr/bin/env bash
# ML pipeline v3: persistent replay, resume training, monotonic arena promote.
# Usage: SEED_FROM_CYCLE2=1 WEEK_DAYS=14 bash scripts/train-loop-v3.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
[[ -f scripts/gating.env ]] && source scripts/gating.env

WEEK_DAYS="${WEEK_DAYS:-14}"
WIN_TARGET="${WIN_TARGET:-0.75}"
PROMOTE_MARGIN="${PROMOTE_MARGIN:-0.02}"
ARENA_GAMES="${ARENA_GAMES:-200}"
NEW_SELFPLAY_PER_CYCLE="${NEW_SELFPLAY_PER_CYCLE:-200}"
SELFPLAY_PLAYOUTS="${SELFPLAY_PLAYOUTS:-200}"
ARENA_PLAYOUTS="${ARENA_PLAYOUTS:-400}"
REPLAY_MAX="${REPLAY_MAX:-50000}"
SEED_FROM_CYCLE2="${SEED_FROM_CYCLE2:-0}"
TRAIN_EPOCHS_FRESH="${TRAIN_EPOCHS_FRESH:-25}"
TRAIN_EPOCHS_RESUME="${TRAIN_EPOCHS_RESUME:-15}"
TRAIN_LR_FRESH="${TRAIN_LR_FRESH:-0.01}"
TRAIN_LR_RESUME="${TRAIN_LR_RESUME:-0.001}"

STATE_DIR="training/state"
DATA_DIR="training/data"
MANIFEST="${STATE_DIR}/manifest.json"
REPLAY="${DATA_DIR}/replay.jsonl"
BEST_PT="${STATE_DIR}/best.pt"
CANDIDATE_ONNX="models/gofer-9x9-candidate.onnx"
BEST_ONNX="models/gofer-9x9-best.onnx"
BOOTSTRAP_ONNX="models/gofer-9x9-bootstrap.onnx"
LOG_FILE="train-v3.log"
HISTORY_DIR=".tectonix/reports/training-history"

DEADLINE="$(date -d "+${WEEK_DAYS} days" +%s 2>/dev/null || python3 -c "import time; print(int(time.time()+86400*float('${WEEK_DAYS}')))")"

mkdir -p "$STATE_DIR" "$DATA_DIR" models "$HISTORY_DIR" .tectonix/reports
export PATH="/usr/local/go/bin:${PATH:-}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

log() { echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"; }

start_sidecar() {
  local model="$1"
  pkill -f 'inference_server.py' 2>/dev/null || true
  python training/inference_server.py --model "$model" --port 8080 &
  SIDECAR_PID=$!
  sleep 2
  curl -sf http://127.0.0.1:8080/health >/dev/null
}

stop_sidecar() {
  kill "${SIDECAR_PID:-}" 2>/dev/null || true
}

init_from_cycle2() {
  log "INIT: seeding state from cycle-2 artifacts"
  cp training/checkpoints/cycle2/best.pt "$BEST_PT"
  : > "$REPLAY"
  for f in training/data/samples-cycle1.jsonl training/data/samples-cycle2.jsonl; do
    if [[ -f "$f" ]]; then
      python -c "from pathlib import Path; from training.replay import append_jsonl; append_jsonl(Path('$f'), Path('$REPLAY'))"
    fi
  done
  python - <<'PY'
from pathlib import Path
from training.manifest import default_manifest, save_manifest
from training.replay import count_lines

m = default_manifest(seed="cycle2", best_win_rate=0.58, best_cycle=2)
m["cycle"] = 2
m["replay_rows"] = count_lines(Path("training/data/replay.jsonl"))
save_manifest(Path("training/state/manifest.json"), m)
PY
  if [[ -f models/gofer-9x9-best.onnx ]]; then
    cp -f models/gofer-9x9-best.onnx "$BOOTSTRAP_ONNX"
    [[ "$BEST_ONNX" != models/gofer-9x9-best.onnx ]] && cp -f models/gofer-9x9-best.onnx "$BEST_ONNX" || true
  fi
  log "INIT complete manifest=$(cat "$MANIFEST")"
}

ensure_manifest() {
  if [[ ! -f "$MANIFEST" ]]; then
    python -c "from pathlib import Path; from training.manifest import default_manifest, save_manifest; save_manifest(Path('$MANIFEST'), default_manifest())"
  fi
}

if [[ "$SEED_FROM_CYCLE2" == "1" ]]; then
  init_from_cycle2
fi
ensure_manifest

cycle="$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('cycle',0)+1)")"
best_rate="$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('best_win_rate',0))")"

while [[ "$(date +%s)" -lt "$DEADLINE" ]]; do
  log "cycle $cycle selfplay=$NEW_SELFPLAY_PER_CYCLE train_resume=$BEST_PT"
  samples="${DATA_DIR}/samples-cycle${cycle}.jsonl"

  if [[ -f "$BEST_ONNX" ]]; then
    start_sidecar "$BEST_ONNX"
  fi
  go run ./cmd/gofer -selfplay -games "$NEW_SELFPLAY_PER_CYCLE" -size 9 \
    -playouts "$SELFPLAY_PLAYOUTS" -full-only true -selfplay-eval mix \
    -o "$samples" -seed "$((42 + cycle))"
  stop_sidecar

  python -c "from pathlib import Path; from training.replay import append_jsonl, trim, count_lines; append_jsonl(Path('$samples'), Path('$REPLAY')); trim(Path('$REPLAY'), $REPLAY_MAX); print(count_lines(Path('$REPLAY')))" \
    | tee -a "$LOG_FILE"

  if [[ -f "$BEST_PT" ]]; then
    cp "$BEST_PT" "${STATE_DIR}/best.pt.pre-cycle"
    python training/train_bootstrap.py \
      --data "$REPLAY" \
      --resume "$BEST_PT" \
      --out-dir "$STATE_DIR" \
      --epochs "$TRAIN_EPOCHS_RESUME" \
      --lr "$TRAIN_LR_RESUME"
  else
    python training/train_bootstrap.py \
      --data "$REPLAY" \
      --out-dir "$STATE_DIR" \
      --epochs "$TRAIN_EPOCHS_FRESH" \
      --lr "$TRAIN_LR_FRESH"
  fi

  python training/export_onnx.py --checkpoint "$BEST_PT" --out "$CANDIDATE_ONNX"

  report=".tectonix/reports/arena-cycle-${cycle}.json"
  start_sidecar "$CANDIDATE_ONNX"
  go run ./cmd/gofer -arena -games "$ARENA_GAMES" -size 9 -playouts "$ARENA_PLAYOUTS" \
    -black-eval heuristic -white-eval onnx \
    -eval-timeout 2s -arena-enhanced none -seed "$((42 + cycle))" \
    -json "$report" | tee -a "$LOG_FILE"
  stop_sidecar

  rate="$(python3 -c "import json; d=json.load(open('$report')); print(d.get('win_rate_challenger',0))")"
  wilson_low="$(python3 -c "import json; d=json.load(open('$report')); print(d.get('wilson_ci_low',0))")"
  wilson_high="$(python3 -c "import json; d=json.load(open('$report')); print(d.get('wilson_ci_high',0))")"
  threshold="$(python3 -c "print(float('$best_rate') + float('$PROMOTE_MARGIN'))")"

  promote=0
  if python3 -c "import sys; sys.exit(0 if float('$rate') > float('$threshold') or float('$rate') >= float('$WIN_TARGET') else 1)"; then
    promote=1
  fi

  history="${HISTORY_DIR}/cycle-${cycle}.json"
  python3 - <<PY
import json
from pathlib import Path
from training.manifest import load_manifest, save_manifest, utc_now
from training.replay import count_lines

arena = json.loads(Path("$report").read_text())
manifest = load_manifest(Path("$MANIFEST"))
manifest["cycle"] = int("$cycle")
manifest["replay_rows"] = count_lines(Path("$REPLAY"))
entry = {
    "cycle": int("$cycle"),
    "win_rate": float("$rate"),
    "wilson_ci_low": float("$wilson_low"),
    "wilson_ci_high": float("$wilson_high"),
    "promote_threshold": float("$threshold"),
    "promoted": bool(int("$promote")),
    "arena": arena,
    "manifest_before": dict(manifest),
}
if int("$promote"):
    manifest["best_win_rate"] = float("$rate")
    manifest["best_cycle"] = int("$cycle")
    manifest["last_promoted_at"] = utc_now()
entry["manifest_after"] = dict(manifest)
Path("$history").write_text(json.dumps(entry, indent=2) + "\n")
save_manifest(Path("$MANIFEST"), manifest)
PY

  if [[ "$promote" == "1" ]]; then
    log "PROMOTE cycle=$cycle rate=$rate threshold=$threshold target=$WIN_TARGET"
    cp "$CANDIDATE_ONNX" "$BEST_ONNX"
    cp "$BEST_ONNX" "$BOOTSTRAP_ONNX"
    rm -f "${STATE_DIR}/best.pt.pre-cycle"
    best_rate="$rate"
    if python3 -c "import sys; sys.exit(0 if float('$rate') >= float('$WIN_TARGET') else 1)"; then
      log "WIN_TARGET reached; stopping"
      exit 0
    fi
  else
    log "REJECT cycle=$cycle rate=$rate threshold=$threshold; keeping $BEST_ONNX"
    if [[ -f "${STATE_DIR}/best.pt.pre-cycle" ]]; then
      cp "${STATE_DIR}/best.pt.pre-cycle" "$BEST_PT"
      rm -f "${STATE_DIR}/best.pt.pre-cycle"
    fi
    start_sidecar "$BEST_ONNX"
    stop_sidecar
  fi

  cycle=$((cycle + 1))
done

log "deadline reached; best_win_rate=$best_rate"
exit 0
