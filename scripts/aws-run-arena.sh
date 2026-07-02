#!/usr/bin/env bash
# Launch full Gofer v2.5 pipeline on an existing Lightsail instance.
# Usage:
#   bash scripts/aws-run-arena.sh              # start (uses default instance)
#   bash scripts/aws-run-arena.sh IP status    # tail log
#   bash scripts/aws-run-arena.sh IP fetch     # download arena JSON
#   bash scripts/aws-run-arena.sh IP wait      # block until done, then fetch
#   bash scripts/aws-run-arena.sh IP destroy   # delete Lightsail instance
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${GOFER_LIGHTSAIL_KEY:-$ROOT/.tectonix/gofer-v25-run.pem}"
INSTANCE="${1:-}"
CMD="${2:-start}"
REGION="${AWS_REGION:-us-east-1}"
NAME="${GOFER_INSTANCE_NAME:-gofer-v25-arena}"

if [[ -z "$INSTANCE" ]]; then
  INSTANCE="$(aws lightsail get-instance --instance-name "$NAME" --region "$REGION" \
    --query 'instance.publicIpAddress' --output text 2>/dev/null || true)"
fi
if [[ -z "$INSTANCE" || "$INSTANCE" == "None" ]]; then
  echo "No instance IP. Create one or pass IP as first argument."
  exit 1
fi
if [[ ! -f "$KEY" ]]; then
  echo "SSH key not found: $KEY"
  exit 1
fi

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "ubuntu@${INSTANCE}")
SCP=(scp -i "$KEY" -o StrictHostKeyChecking=no)

case "$CMD" in
stop-loop)
  "${SSH[@]}" 'cd ~/Gofer 2>/dev/null || exit 0; \
    test -f week.pid && kill $(cat week.pid) 2>/dev/null || true; \
    pkill -f inference_server.py 2>/dev/null || true; \
    pkill -f train-loop-v3.sh 2>/dev/null || true; \
    pkill -f weekly-train-loop.sh 2>/dev/null || true; \
    echo stopped'
  exit 0
  ;;
start-v3)
  echo "==> start train-loop-v3 on $INSTANCE"
  "${SSH[@]}" "cd ~/Gofer && git pull --ff-only && chmod +x scripts/train-loop-v3.sh && \
    (test -f week.pid && kill \$(cat week.pid) 2>/dev/null || true); \
    pkill -f inference_server.py 2>/dev/null || true; \
    nohup env SEED_FROM_CYCLE2=${SEED_FROM_CYCLE2:-0} WEEK_DAYS=${WEEK_DAYS:-14} \
      WIN_TARGET=${WIN_TARGET:-0.75} NEW_SELFPLAY_PER_CYCLE=${NEW_SELFPLAY_PER_CYCLE:-200} \
      bash scripts/train-loop-v3.sh > train-v3.log 2>&1 & \
    echo \$! > week.pid && echo v3_loop_pid=\$(cat week.pid)"
  echo "poll: bash scripts/aws-run-arena.sh $INSTANCE week-status"
  exit 0
  ;;
fetch-all)
  mkdir -p "$ROOT/.tectonix/reports/training-history"
  "${SCP[@]}" "ubuntu@${INSTANCE}:~/Gofer/.tectonix/reports/arena-cycle-*.json" \
    "$ROOT/.tectonix/reports/" 2>/dev/null || true
  "${SCP[@]}" -r "ubuntu@${INSTANCE}:~/Gofer/.tectonix/reports/training-history/" \
    "$ROOT/.tectonix/reports/" 2>/dev/null || true
  "${SCP[@]}" "ubuntu@${INSTANCE}:~/Gofer/train-v3.log" "$ROOT/.tectonix/reports/train-v3.log" 2>/dev/null || true
  "${SCP[@]}" "ubuntu@${INSTANCE}:~/Gofer/training/state/manifest.json" \
    "$ROOT/.tectonix/reports/manifest.json" 2>/dev/null || true
  echo "fetched reports to .tectonix/reports/"
  exit 0
  ;;
seed-status)
  "${SSH[@]}" 'cd ~/Gofer && echo "=== manifest ===" && cat training/state/manifest.json 2>/dev/null || echo no manifest; \
    echo "=== artifacts ===" && ls -la training/state/best.pt models/gofer-9x9-best.onnx 2>/dev/null || true'
  exit 0
  ;;
status)
  "${SSH[@]}" 'tail -40 ~/Gofer/run.log 2>/dev/null || echo no log yet; ps aux | grep -E "remote-arena|inference_server|gofer" | grep -v grep | head -5 || true'
  exit 0
  ;;
fetch)
  mkdir -p "$ROOT/.tectonix/reports"
  "${SCP[@]}" "ubuntu@${INSTANCE}:~/Gofer/.tectonix/reports/arena-9x9-onnx-v25.json" \
    "$ROOT/.tectonix/reports/arena-9x9-onnx-v25.json" 2>/dev/null || echo "report not ready yet"
  "${SCP[@]}" "ubuntu@${INSTANCE}:~/Gofer/run.log" "$ROOT/.tectonix/reports/lightsail-run.log" 2>/dev/null || true
  exit 0
  ;;
wait)
  echo "==> waiting for arena JSON on $INSTANCE (poll 60s)..."
  while true; do
    if "${SSH[@]}" 'test -f ~/Gofer/.tectonix/reports/arena-9x9-onnx-v25.json'; then
      bash "$ROOT/scripts/aws-run-arena.sh" "$INSTANCE" fetch
      grep -E 'win_rate_challenger|wilson_ci' "$ROOT/.tectonix/reports/arena-9x9-onnx-v25.json" || true
      exit 0
    fi
    "${SSH[@]}" 'tail -2 ~/Gofer/run.log 2>/dev/null' || true
    sleep 60
  done
  ;;
destroy)
  aws lightsail delete-instance --instance-name "$NAME" --region "$REGION"
  echo "deleted $NAME"
  exit 0
  ;;
week)
  echo "==> start ${WEEK_DAYS:-7}-day training loop on $INSTANCE"
  "${SSH[@]}" "cd ~/Gofer && git pull --ff-only && chmod +x scripts/weekly-train-loop.sh && \
    (test -f week.pid && kill \$(cat week.pid) 2>/dev/null || true); \
    nohup env WEEK_DAYS=${WEEK_DAYS:-7} WIN_TARGET=${WIN_TARGET:-0.75} bash scripts/weekly-train-loop.sh > week.log 2>&1 & \
    echo \$! > week.pid && echo week_loop_pid=\$(cat week.pid)"
  echo "poll: bash scripts/aws-run-arena.sh $INSTANCE week-status"
  exit 0
  ;;
week-status)
  "${SSH[@]}" 'tail -30 ~/Gofer/train-v3.log 2>/dev/null || tail -30 ~/Gofer/week.log 2>/dev/null || tail -30 ~/Gofer/run.log 2>/dev/null || echo no log'
  exit 0
  ;;
start)
  ;;
*)
  echo "unknown command: $CMD"
  exit 1
  ;;
esac

echo "==> instance $INSTANCE ($NAME)"
"${SSH[@]}" "export DEBIAN_FRONTEND=noninteractive; command -v git >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y git; }"

echo "==> clone / update repo"
"${SSH[@]}" 'if [[ -d Gofer ]]; then cd Gofer && git pull --ff-only; else git clone https://github.com/DevomB/Gofer.git && cd Gofer; fi'

if "${SSH[@]}" 'test -f ~/Gofer/run.pid && kill -0 "$(cat ~/Gofer/run.pid)" 2>/dev/null'; then
  echo "pipeline already running pid=$(${SSH[@]} 'cat ~/Gofer/run.pid')"
  exit 0
fi

echo "==> start pipeline in background (log: ~/Gofer/run.log)"
"${SSH[@]}" 'cd ~/Gofer && chmod +x scripts/remote-arena-gate.sh && nohup bash scripts/remote-arena-gate.sh > run.log 2>&1 & echo $! > run.pid && echo started pid=$(cat run.pid)'
echo "poll:  bash scripts/aws-run-arena.sh $INSTANCE status"
echo "fetch: bash scripts/aws-run-arena.sh $INSTANCE fetch"
echo "wait:  bash scripts/aws-run-arena.sh $INSTANCE wait"
