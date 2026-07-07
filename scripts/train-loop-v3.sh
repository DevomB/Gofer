#!/usr/bin/env bash
# ML pipeline v3: persistent replay, resume training, head-to-head arena promote.
#
# Each cycle: parallel self-play (vs the current champion) -> resume-train ->
# export candidate -> arena the candidate against the CURRENT champion (not a
# fixed heuristic) -> promote only if the candidate wins decisively. There is no
# win-target stop: strength keeps climbing until candidates stop beating the
# champion or the deadline hits.
#
# Usage: WEEK_DAYS=14 PARALLEL=8 bash scripts/train-loop-v3.sh
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/gating.env"

WEEK_DAYS="${WEEK_DAYS:-14}"
PROMOTE_WIN="${PROMOTE_WIN:-0.55}"
ARENA_GAMES="${ARENA_GAMES:-200}"
NEW_SELFPLAY_PER_CYCLE="${NEW_SELFPLAY_PER_CYCLE:-200}"
SELFPLAY_PLAYOUTS="${SELFPLAY_PLAYOUTS:-200}"
ARENA_PLAYOUTS="${ARENA_PLAYOUTS:-400}"
SELFPLAY_TEMP_MOVES="${SELFPLAY_TEMP_MOVES:-16}"
ARENA_OPENING_MOVES="${ARENA_OPENING_MOVES:-8}"
REPLAY_MAX="${REPLAY_MAX:-50000}"
TRAIN_EPOCHS_FRESH="${TRAIN_EPOCHS_FRESH:-25}"
TRAIN_EPOCHS_RESUME="${TRAIN_EPOCHS_RESUME:-15}"
TRAIN_LR_FRESH="${TRAIN_LR_FRESH:-0.01}"
TRAIN_LR_RESUME="${TRAIN_LR_RESUME:-0.001}"
PARALLEL="${PARALLEL:-$(nproc 2>/dev/null || echo 8)}"
MAX_CYCLES="${MAX_CYCLES:-0}" # 0 = run until WEEK_DAYS deadline; 1 = single unattended cycle
EVAL_BACKEND="${EVAL_BACKEND:-inprocess}"
ORT_VERSION="1.26.0"
ORT_ART="${ROOT}/.tectonix/artifacts/onnxruntime-linux-x64-${ORT_VERSION}"
ORT_SO="${ORT_ART}/lib/libonnxruntime.so.${ORT_VERSION}"
GO_BUILD_TAGS=()
GOFER_EVAL_EXTRA=()

STATE_DIR="training/state"
DATA_DIR="training/data"
MANIFEST="${STATE_DIR}/manifest.json"
REPLAY="${DATA_DIR}/replay.jsonl"
BEST_PT="${STATE_DIR}/best.pt"
CANDIDATE_ONNX="models/gofer-9x9-candidate.onnx"
BEST_ONNX="models/gofer-9x9-best.onnx"
BOOTSTRAP_ONNX="models/gofer-9x9-bootstrap.onnx"
GOFER_BIN="bin/gofer"
CHAMP_PORT=8080
CHALLENGER_PORT=8081
LOG_FILE="train-v3.log"
HISTORY_DIR=".tectonix/reports/training-history"

DEADLINE="$(date -d "+${WEEK_DAYS} days" +%s 2>/dev/null || python3 -c "import time; print(int(time.time()+86400*float('${WEEK_DAYS}')))")"

mkdir -p "$STATE_DIR" "$DATA_DIR" models "$HISTORY_DIR" .tectonix/reports bin
export PATH="/usr/local/go/bin:${PATH:-}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

log() { echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"; }

ensure_ort_shared_lib() {
  if [[ -n "${ONNXRUNTIME_SHARED_LIBRARY_PATH:-}" && -f "${ONNXRUNTIME_SHARED_LIBRARY_PATH}" ]]; then
    return 0
  fi
  if [[ -f "$ORT_SO" ]]; then
    export ONNXRUNTIME_SHARED_LIBRARY_PATH="$ORT_SO"
    return 0
  fi
  log "downloading ORT ${ORT_VERSION} linux/amd64 for in-process inference"
  mkdir -p "${ROOT}/.tectonix/artifacts"
  local tgz="${ROOT}/.tectonix/artifacts/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" -o "$tgz"
  rm -rf "$ORT_ART"
  tar -xzf "$tgz" -C "${ROOT}/.tectonix/artifacts"
  export ONNXRUNTIME_SHARED_LIBRARY_PATH="$ORT_SO"
}

gofer_eval_args() {
  # Sets global GOFER_EVAL_EXTRA array for ./bin/gofer invocations.
  GOFER_EVAL_EXTRA=()
  if [[ "$EVAL_BACKEND" == "inprocess" ]]; then
    GOFER_EVAL_EXTRA=(-eval-backend inprocess -model "$1")
    if [[ -n "${2:-}" ]]; then
      GOFER_EVAL_EXTRA+=(-model-2 "$2")
    fi
  elif [[ "$EVAL_BACKEND" == "sidecar" ]]; then
    GOFER_EVAL_EXTRA=(-eval-backend sidecar)
  fi
}

# Build the engine once and reuse the binary every cycle (no per-cycle recompile).
if [[ "$EVAL_BACKEND" == "inprocess" ]]; then
  ensure_ort_shared_lib
  GO_BUILD_TAGS=(-tags=onnx)
  log "BUILD $GOFER_BIN (in-process ORT, -tags=onnx)"
else
  log "BUILD $GOFER_BIN (sidecar, no CGO)"
fi
CGO_ENABLED=1 go build "${GO_BUILD_TAGS[@]}" -o "$GOFER_BIN" ./cmd/gofer

SIDECAR_PIDS=()
start_sidecar() {
  local model="$1" port="$2"
  python training/inference_server.py --model "$model" --port "$port" &
  SIDECAR_PIDS+=($!)
  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then return 0; fi
    sleep 1
  done
  echo "sidecar on port ${port} failed to become healthy" >&2
  return 1
}

stop_sidecars() {
  pkill -f 'inference_server.py' 2>/dev/null || true
  for pid in "${SIDECAR_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  SIDECAR_PIDS=()
}
trap stop_sidecars EXIT

if [[ ! -f "$MANIFEST" ]]; then
  python -c "from pathlib import Path; from training.manifest import default_manifest, save_manifest; save_manifest(Path('$MANIFEST'), default_manifest())"
fi

cycle="$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('cycle',0)+1)")"
first_cycle="$cycle"

while [[ "$(date +%s)" -lt "$DEADLINE" ]]; do
  samples="${DATA_DIR}/samples-cycle${cycle}.jsonl"
  have_champion=0
  [[ -f "$BEST_ONNX" ]] && have_champion=1

  # ---- self-play (parallel; vs current champion when one exists) ----
  if [[ "$have_champion" == "1" ]]; then
    log "cycle $cycle selfplay=$NEW_SELFPLAY_PER_CYCLE (mix vs champion) parallel=$PARALLEL backend=$EVAL_BACKEND"
    gofer_eval_args "$BEST_ONNX"
    if [[ "$EVAL_BACKEND" != "inprocess" ]]; then
      start_sidecar "$BEST_ONNX" "$CHAMP_PORT"
    fi
    "./$GOFER_BIN" -selfplay -games "$NEW_SELFPLAY_PER_CYCLE" -size 9 \
      -playouts "$SELFPLAY_PLAYOUTS" -full-only=true -selfplay-eval mix \
      -selfplay-onnx-fraction 0.7 -selfplay-parallel "$PARALLEL" \
      -selfplay-temp-moves "$SELFPLAY_TEMP_MOVES" \
      -onnx-url "http://127.0.0.1:${CHAMP_PORT}" \
      "${GOFER_EVAL_EXTRA[@]}" \
      -o "$samples" -seed "$((42 + cycle))"
    if [[ "$EVAL_BACKEND" != "inprocess" ]]; then
      stop_sidecars
    fi
  else
    log "cycle $cycle selfplay=$NEW_SELFPLAY_PER_CYCLE (heuristic bootstrap) parallel=$PARALLEL"
    "./$GOFER_BIN" -selfplay -games "$NEW_SELFPLAY_PER_CYCLE" -size 9 \
      -playouts "$SELFPLAY_PLAYOUTS" -full-only=true -selfplay-eval heuristic \
      -selfplay-parallel "$PARALLEL" -selfplay-temp-moves "$SELFPLAY_TEMP_MOVES" \
      -o "$samples" -seed "$((42 + cycle))"
  fi

  python -m training.cycle append-replay "$samples" --max-lines "$REPLAY_MAX" | tee -a "$LOG_FILE"

  # ---- train (resume the champion's weights when present) ----
  if [[ -f "$BEST_PT" ]]; then
    cp "$BEST_PT" "${STATE_DIR}/best.pt.pre-cycle"
    python training/train_bootstrap.py \
      --data "$REPLAY" --resume "$BEST_PT" --out-dir "$STATE_DIR" \
      --epochs "$TRAIN_EPOCHS_RESUME" --lr "$TRAIN_LR_RESUME"
  else
    python training/train_bootstrap.py \
      --data "$REPLAY" --out-dir "$STATE_DIR" \
      --epochs "$TRAIN_EPOCHS_FRESH" --lr "$TRAIN_LR_FRESH"
  fi

  python training/export_onnx.py --checkpoint "$BEST_PT" --out "$CANDIDATE_ONNX"

  # ---- arena: candidate vs CURRENT champion (head-to-head) ----
  report=".tectonix/reports/arena-cycle-${cycle}.json"
  if [[ "$have_champion" == "1" ]]; then
    log "cycle $cycle arena=$ARENA_GAMES candidate vs champion parallel=$PARALLEL backend=$EVAL_BACKEND"
    gofer_eval_args "$BEST_ONNX" "$CANDIDATE_ONNX"
    if [[ "$EVAL_BACKEND" != "inprocess" ]]; then
      start_sidecar "$BEST_ONNX" "$CHAMP_PORT"
      start_sidecar "$CANDIDATE_ONNX" "$CHALLENGER_PORT"
    fi
    "./$GOFER_BIN" -arena -games "$ARENA_GAMES" -size 9 -playouts "$ARENA_PLAYOUTS" \
      -black-eval onnx -white-eval onnx2 \
      -onnx-url "http://127.0.0.1:${CHAMP_PORT}" \
      -onnx-url-2 "http://127.0.0.1:${CHALLENGER_PORT}" \
      "${GOFER_EVAL_EXTRA[@]}" \
      -arena-parallel "$PARALLEL" -arena-opening-moves "$ARENA_OPENING_MOVES" \
      -eval-timeout 2s -arena-enhanced none \
      -seed "$((42 + cycle))" -json "$report" | tee -a "$LOG_FILE"
    if [[ "$EVAL_BACKEND" != "inprocess" ]]; then
      stop_sidecars
    fi
  else
    log "cycle $cycle arena=$ARENA_GAMES candidate vs heuristic (bootstrap sanity check) parallel=$PARALLEL backend=$EVAL_BACKEND"
    gofer_eval_args "$CANDIDATE_ONNX"
    if [[ "$EVAL_BACKEND" != "inprocess" ]]; then
      start_sidecar "$CANDIDATE_ONNX" "$CHAMP_PORT"
    fi
    "./$GOFER_BIN" -arena -games "$ARENA_GAMES" -size 9 -playouts "$ARENA_PLAYOUTS" \
      -black-eval heuristic -white-eval onnx \
      -onnx-url "http://127.0.0.1:${CHAMP_PORT}" \
      "${GOFER_EVAL_EXTRA[@]}" \
      -arena-parallel "$PARALLEL" -arena-opening-moves "$ARENA_OPENING_MOVES" \
      -eval-timeout 2s -arena-enhanced none \
      -seed "$((42 + cycle))" -json "$report" | tee -a "$LOG_FILE"
    if [[ "$EVAL_BACKEND" != "inprocess" ]]; then
      stop_sidecars
    fi
  fi

  # The first cycle seeds the champion unconditionally (the net only imitates the
  # heuristic from bootstrap data; real gains come from net-vs-net self-play next).
  seed_flag=()
  [[ "$have_champion" == "0" ]] && seed_flag=(--seed-champion)
  GATING_MODE="${GATING_MODE:-normal}"
  promote="$(python -m training.cycle record-cycle "$cycle" "$report" \
    --promote-win "$PROMOTE_WIN" --history-dir "$HISTORY_DIR" \
    --gating-mode "$GATING_MODE" "${seed_flag[@]}")"
  rate="$(python3 -c "import json; print(json.load(open('$report')).get('win_rate_challenger',0))")"
  would_promote="$(python3 -c "import json; print(json.load(open('${HISTORY_DIR}/cycle-${cycle}.json')).get('would_promote', False))")"

  if [[ "$GATING_MODE" == "hold" ]]; then
    log "HOLD cycle=$cycle rate=$rate would_promote=$would_promote gating_mode=hold (no champion swap)"
    if [[ -f "${STATE_DIR}/best.pt.pre-cycle" ]]; then
      cp "${STATE_DIR}/best.pt.pre-cycle" "$BEST_PT"
      rm -f "${STATE_DIR}/best.pt.pre-cycle"
    fi
  elif [[ "$promote" == "promote" ]]; then
    if [[ "$have_champion" == "0" ]]; then
      log "SEED cycle=$cycle rate=$rate (first champion established; self-play now uses the net)"
    else
      log "PROMOTE cycle=$cycle rate=$rate promote_win=$PROMOTE_WIN (candidate is the new champion)"
    fi
    cp "$CANDIDATE_ONNX" "$BEST_ONNX"
    cp "$BEST_ONNX" "$BOOTSTRAP_ONNX"
    rm -f "${STATE_DIR}/best.pt.pre-cycle"
  else
    log "REJECT cycle=$cycle rate=$rate; keeping champion $BEST_ONNX"
    if [[ -f "${STATE_DIR}/best.pt.pre-cycle" ]]; then
      cp "${STATE_DIR}/best.pt.pre-cycle" "$BEST_PT"
      rm -f "${STATE_DIR}/best.pt.pre-cycle"
    fi
  fi

  cycle=$((cycle + 1))
  if [[ "$MAX_CYCLES" -gt 0 && $((cycle - first_cycle)) -ge "$MAX_CYCLES" ]]; then
    log "MAX_CYCLES=$MAX_CYCLES complete after cycle $((cycle - 1))"
    exit 0
  fi
done

log "deadline reached after cycle $((cycle - 1))"
exit 0
