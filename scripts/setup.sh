#!/usr/bin/env bash
# One-shot setup for a fresh Linux machine (laptop, Vultr, Lambda, ...):
# Go toolchain check, GNU Go (evaluation anchor), ONNX Runtime, Python venv, build.
# GNU Go is installed with apt-get (Debian/Ubuntu) or dnf (Fedora).
#
#   scripts/setup.sh          # CPU box
#   scripts/setup.sh gpu      # NVIDIA box: GPU ONNX Runtime + CUDA PyTorch
set -euo pipefail

FLAVOR="${1:-cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v go >/dev/null || { echo "install Go 1.22+ first: https://go.dev/dl/" >&2; exit 1; }
if ! command -v gnugo >/dev/null && [[ ! -x /usr/games/gnugo ]]; then
  if command -v apt-get >/dev/null; then
    sudo apt-get install -y gnugo python3-venv || echo "warning: could not install gnugo (the default evaluation anchors need it)"
  elif command -v dnf >/dev/null; then
    sudo dnf install -y gnugo || echo "warning: could not install gnugo (the default evaluation anchors need it)"
  else
    echo "warning: install GNU Go yourself (the default evaluation anchors need it)"
  fi
fi

scripts/install-ort.sh "$FLAVOR"

python3 -m venv .venv
if [[ "$FLAVOR" == "gpu" ]]; then
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu124
else
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/pip install -q -r training/requirements.txt

make build
echo
echo "ready. try:"
echo "  make test"
echo "  .venv/bin/python -m training.loop --run runs/9x9 --gofer bin/gofer --hours 8"
