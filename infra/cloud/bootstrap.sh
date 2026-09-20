#!/usr/bin/env bash
# Prepare a fresh Linux box, x86-64 or arm64 (rented GPU VM, RunPod/Vast
# container, Graviton, home server) to run the Gofer pipeline without Docker.
# Idempotent: safe to rerun.
#
#   bash infra/cloud/bootstrap.sh            # from a checkout
#   GOFER_CONFIG=configs/pipeline-gpu.toml bash infra/cloud/bootstrap.sh --run
#
# Installs Go (if missing or too old), ONNX Runtime 1.26.0, a Python venv with
# the learner deps (CUDA torch when nvidia-smi works), builds bin/gofer with the
# in-process ORT backend, and runs the orchestrator's unit tests as a self-check.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GO_VERSION="${GO_VERSION:-1.22.5}"
ORT_VERSION="1.26.0"
VENV="${GOFER_VENV:-$ROOT/.venv}"
ART="$ROOT/.tectonix/artifacts"

log() { echo "[bootstrap] $*"; }

# Go and ONNX Runtime spell the same machine differently, so resolve both once.
# Mirrors ORT_BUILDS in training/pipeline/procs.py; keep the two in step.
case "$(uname -m)" in
  x86_64|amd64)  GO_ARCH=amd64; ORT_NAME="onnxruntime-linux-x64-${ORT_VERSION}" ;;
  aarch64|arm64) GO_ARCH=arm64; ORT_NAME="onnxruntime-linux-aarch64-${ORT_VERSION}" ;;
  *)
    log "no pinned Go / ONNX Runtime build for $(uname -m)."
    log "The sidecar backend runs anywhere Python onnxruntime installs:"
    log "  python -m training.pipeline run --config <cfg> --set 'engine.backend=\"sidecar\"'"
    exit 1 ;;
esac

sudo_cmd() { [[ $EUID -eq 0 ]] || echo sudo; }

# A bare cloud image has neither a C toolchain nor, on Debian, the separate
# venv package. cgo needs the first for -tags=onnx and this script needs the
# second three lines later, so check both before doing any work.
ensure_build_deps() {
  local missing=()
  command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || missing+=("a C compiler")
  python3 -c "import ensurepip" >/dev/null 2>&1 || missing+=("python venv support")
  (( ${#missing[@]} )) || return 0
  log "installing: ${missing[*]}"
  if command -v apt-get >/dev/null 2>&1; then
    $(sudo_cmd) apt-get update -qq
    $(sudo_cmd) apt-get install -y -qq build-essential python3-venv
  elif command -v dnf >/dev/null 2>&1; then
    $(sudo_cmd) dnf install -y -q gcc
  else
    log "no apt-get or dnf; install ${missing[*]} by hand and rerun"
    exit 1
  fi
}
ensure_build_deps

need_go() {
  command -v go >/dev/null 2>&1 || return 0
  local have; have="$(go env GOVERSION | sed 's/^go//')"
  [[ "$(printf '%s\n1.22\n' "$have" | sort -V | head -1)" != "1.22" ]]
}

if need_go; then
  log "installing Go ${GO_VERSION}"
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${GO_ARCH}.tar.gz" -o /tmp/go.tgz
  $(sudo_cmd) rm -rf /usr/local/go && $(sudo_cmd) tar -C /usr/local -xzf /tmp/go.tgz
fi
export PATH="/usr/local/go/bin:$PATH"

ORT_LIB="$ART/${ORT_NAME}/lib/libonnxruntime.so.${ORT_VERSION}"
if [[ ! -f "$ORT_LIB" ]]; then
  log "downloading ONNX Runtime ${ORT_VERSION}"
  mkdir -p "$ART"
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_NAME}.tgz" | tar -xz -C "$ART"
fi
export ONNXRUNTIME_SHARED_LIBRARY_PATH="$ORT_LIB"

if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating venv $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
if ! python -c "import torch" 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "installing CUDA torch"
    pip install -q torch
  else
    log "no NVIDIA GPU: installing CPU torch"
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi
pip install -q -r training/requirements.txt pytest

log "building bin/gofer (in-process ORT)"
mkdir -p bin
CGO_ENABLED=1 go build -tags=onnx -o bin/gofer ./cmd/gofer

log "self-check"
python -m pytest training/pipeline -q
python -c "import torch; print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"

cat > "$ROOT/.gofer-env" <<EOF
export PATH="/usr/local/go/bin:\$PATH"
export ONNXRUNTIME_SHARED_LIBRARY_PATH="$ORT_LIB"
source "$VENV/bin/activate"
EOF
log "ready. Next: source .gofer-env && python -m training.pipeline run --config ${GOFER_CONFIG:-configs/pipeline-gpu.toml} --no-build"

if [[ "${1:-}" == "--run" ]]; then
  exec python -m training.pipeline run --config "${GOFER_CONFIG:-configs/pipeline-gpu.toml}" --no-build
fi
