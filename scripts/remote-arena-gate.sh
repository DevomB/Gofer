#!/usr/bin/env bash
# Run on a fresh Ubuntu Lightsail box after cloning Gofer.
# Usage: bash scripts/remote-arena-gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> install deps (Go 1.22+, Python venv)"
export DEBIAN_FRONTEND=noninteractive
need_go() {
  if ! command -v go >/dev/null; then return 0; fi
  ver="$(go env GOVERSION 2>/dev/null | sed 's/go//')"
  major="${ver%%.*}"; minor="$(echo "$ver" | cut -d. -f2)"
  [[ "$major" -lt 1 || ( "$major" -eq 1 && "$minor" -lt 22 ) ]]
}
if need_go; then
  GO_TAR=go1.22.12.linux-amd64.tar.gz
  curl -fsSL "https://go.dev/dl/${GO_TAR}" -o "/tmp/${GO_TAR}"
  sudo rm -rf /usr/local/go
  sudo tar -C /usr/local -xzf "/tmp/${GO_TAR}"
  export PATH="/usr/local/go/bin:$PATH"
fi
if ! command -v python3 >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y python3 python3-pip python3-venv git curl
elif ! python3 -m venv --help >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv python3-pip git curl
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r training/requirements.txt

echo "==> tests (short)"
go test ./... -short -count=1

echo "==> export ONNX if missing"
test -f models/gofer-9x9-bootstrap.onnx || python training/export_onnx.py --out models/gofer-9x9-bootstrap.onnx

echo "==> optional retrain (skip if SELFPLAY_GAMES=0)"
SELFPLAY_GAMES="${SELFPLAY_GAMES:-100}"
if [[ "$SELFPLAY_GAMES" -gt 0 ]]; then
  mkdir -p training/data
  go run ./cmd/gofer -selfplay -games "$SELFPLAY_GAMES" -size 9 -playouts 100 \
    -eval heuristic -o training/data/samples.jsonl -seed 42
  python training/train_bootstrap.py --data training/data/samples.jsonl --epochs 25
  python training/export_onnx.py --checkpoint training/checkpoints/best.pt \
    --out models/gofer-9x9-bootstrap.onnx
fi

echo "==> start sidecar"
pkill -f 'inference_server.py' 2>/dev/null || true
python training/inference_server.py --model models/gofer-9x9-bootstrap.onnx --port 8080 &
SIDECAR_PID=$!
sleep 2
curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "sidecar failed"; exit 1; }

echo "==> 200-game arena (heuristic vs onnx, equal 400 playouts)"
go run ./cmd/gofer -arena -games 200 -size 9 -playouts 400 \
  -black-eval heuristic -white-eval onnx \
  -eval-timeout 2s -seed 42 -arena-enhanced none \
  -json .tectonix/reports/arena-9x9-onnx-v25.json | tee .tectonix/reports/arena-9x9-onnx-v25.log

kill "$SIDECAR_PID" 2>/dev/null || true
echo "==> done. Report: .tectonix/reports/arena-9x9-onnx-v25.json"
grep -E 'win_rate_challenger|wilson_ci' .tectonix/reports/arena-9x9-onnx-v25.json | head -5 || true
