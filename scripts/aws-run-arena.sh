#!/usr/bin/env bash
# Launch full Gofer v2.5 pipeline on an existing Lightsail instance.
# Usage: bash scripts/aws-run-arena.sh [instance-ip]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${GOFER_LIGHTSAIL_KEY:-$ROOT/.tectonix/gofer-v25-run.pem}"
INSTANCE="${1:-}"
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

echo "==> instance $INSTANCE ($NAME)"

"${SSH[@]}" "export DEBIAN_FRONTEND=noninteractive; command -v git >/dev/null || sudo apt-get update -qq && sudo apt-get install -y git"

echo "==> clone / update repo"
"${SSH[@]}" 'if [[ -d Gofer ]]; then cd Gofer && git pull --ff-only; else git clone https://github.com/DevomB/Gofer.git && cd Gofer; fi'

echo "==> start pipeline in background (log: ~/Gofer/run.log)"
"${SSH[@]}" 'cd ~/Gofer && chmod +x scripts/remote-arena-gate.sh && nohup bash scripts/remote-arena-gate.sh > run.log 2>&1 & echo $! > run.pid && echo started pid=$(cat run.pid)'

echo "==> tail log (Ctrl-C stops tail only; job keeps running on server)"
echo "    poll: bash scripts/aws-run-arena.sh $INSTANCE status"
echo "    fetch: bash scripts/aws-run-arena.sh $INSTANCE fetch"

if [[ "${2:-}" == "status" ]]; then
  "${SSH[@]}" 'tail -30 ~/Gofer/run.log 2>/dev/null || echo no log yet'
  exit 0
fi

if [[ "${2:-}" == "fetch" ]]; then
  mkdir -p "$ROOT/.tectonix/reports"
  "${SCP[@]}" "ubuntu@${INSTANCE}:~/Gofer/.tectonix/reports/arena-9x9-onnx-v25.json" \
    "$ROOT/.tectonix/reports/arena-9x9-onnx-v25.json" 2>/dev/null || echo "report not ready yet"
  "${SCP[@]}" "ubuntu@${INSTANCE}:~/Gofer/run.log" "$ROOT/.tectonix/reports/lightsail-run.log" 2>/dev/null || true
  exit 0
fi

if [[ "${2:-}" == "wait" ]]; then
  echo "==> waiting for arena JSON (poll every 60s)..."
  while true; do
    if "${SSH[@]}" 'test -f ~/Gofer/.tectonix/reports/arena-9x9-onnx-v25.json'; then
      echo "==> done"
      bash "$ROOT/scripts/aws-run-arena.sh" "$INSTANCE" fetch
      grep -E 'win_rate_challenger|wilson_ci' "$ROOT/.tectonix/reports/arena-9x9-onnx-v25.json" || true
      exit 0
    fi
    "${SSH[@]}" 'tail -3 ~/Gofer/run.log 2>/dev/null' || true
    sleep 60
  done
fi
