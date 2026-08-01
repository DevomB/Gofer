#!/usr/bin/env bash
# Install Gofer training stack on a GPU server.
# Default install directory: ~/GoEngine (override: GOENGINE_ROOT or GOFER_ROOT).
#
# Prereq: run scope first (optional):
#   bash scripts/scope-gpu-server.sh
#
# Usage:
#   cd ~/GoEngine && bash scripts/setup-gpu-server.sh
#   GOENGINE_ROOT=~/GoEngine bash scripts/setup-gpu-server.sh
#   SKIP_SMOKE=1 bash scripts/setup-gpu-server.sh
#
# Security defaults:
#   EVAL_BACKEND=inprocess (no HTTP sidecar)
#   does not install or copy AWS/SSH private keys
set -euo pipefail

# Safe when piped: curl ... | bash  (BASH_SOURCE unset under set -u)
_script_path="${BASH_SOURCE[0]:-${0:-}}"
if [[ -z "$_script_path" || "$_script_path" == bash || "$_script_path" == -bash ]]; then
  SCRIPT_DIR="$(pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "$_script_path")" && pwd)"
fi
unset _script_path
ROOT="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd || echo "${SCRIPT_DIR}")"

GOENGINE_ROOT="${GOENGINE_ROOT:-${GOFER_ROOT:-${HOME}/GoEngine}}"
GOFER_ROOT="${GOENGINE_ROOT}"
GOFER_REPO_URL="${GOFER_REPO_URL:-https://github.com/DevomB/Gofer.git}"
GOFER_BRANCH="${GOFER_BRANCH:-main}"
VENV_DIR="${GOFER_VENV:-${GOFER_ROOT}/.venv311}"
SCOPE_CANDIDATES=(
  "${GOFER_SCOPE_REPORT:-}"
  "${HOME}/GoEngine/.tectonix/reports/gpu-scope-report.txt"
  "${GOFER_ROOT}/.tectonix/reports/gpu-scope-report.txt"
  "${HOME}/gofer-gpu-scope-report.txt"
  "${HOME}/gpu-scope-report.txt"
  "${ROOT}/.tectonix/reports/gpu-scope-report.txt"
)
ORT_VERSION="1.26.0"
GO_VERSION="1.22.5"

log() { echo "[setup-gpu] $*"; }

read_scope() {
  local key="$1" default="${2:-}"
  local file val
  for file in "${SCOPE_CANDIDATES[@]}"; do
    [[ -n "$file" && -f "$file" ]] || continue
    val="$(grep -E "^${key}=" "$file" | tail -1 | cut -d= -f2- || true)"
    if [[ -n "$val" ]]; then
      echo "$val"
      return 0
    fi
  done
  echo "$default"
}

detect_gpu_backend() {
  if command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
    echo rocm
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo cuda
  else
    echo cpu
  fi
}

pick_python() {
  local from_scope="$1"
  if [[ -n "$from_scope" && "$from_scope" != none ]] && command -v "$from_scope" >/dev/null 2>&1; then
    echo "$from_scope"
    return
  fi
  # Prefer 3.11 for ORT parity; fall back to newest available.
  for py in python3.11 python3.12 python3.10 python3; do
    if command -v "$py" >/dev/null 2>&1; then
      echo "$py"
      return
    fi
  done
  echo ""
}

rocm_pip_index() {
  local ver="${1:-none}"
  case "$ver" in
    7.*|7) echo "https://download.pytorch.org/whl/rocm6.3" ;;
    6.3*|6.3) echo "https://download.pytorch.org/whl/rocm6.3" ;;
    6.2*|6.2) echo "https://download.pytorch.org/whl/rocm6.2" ;;
    6.1*|6.1) echo "https://download.pytorch.org/whl/rocm6.1" ;;
    6.0*|6.0) echo "https://download.pytorch.org/whl/rocm6.0" ;;
    5.7*|5.7) echo "https://download.pytorch.org/whl/rocm5.7" ;;
    *) echo "https://download.pytorch.org/whl/rocm6.3" ;;
  esac
}

detect_gofer_root() {
  local from_scope="$1"
  # Explicit override (GOENGINE_ROOT / GOFER_ROOT).
  if [[ -n "${GOENGINE_ROOT:-}" && -f "${GOENGINE_ROOT}/scripts/train-loop-v3.sh" ]]; then
    echo "$GOENGINE_ROOT"
    return
  fi
  # Running from checkout: .../GoEngine/scripts/setup-gpu-server.sh
  if [[ -f "${SCRIPT_DIR}/train-loop-v3.sh" ]]; then
    echo "$SCRIPT_DIR"
    return
  fi
  if [[ -f "${SCRIPT_DIR}/../scripts/train-loop-v3.sh" ]]; then
    echo "$(cd "${SCRIPT_DIR}/.." && pwd)"
    return
  fi
  if [[ -n "$from_scope" && "$from_scope" != none && -f "${from_scope}/scripts/train-loop-v3.sh" ]]; then
    echo "$from_scope"
    return
  fi
  for d in "${HOME}/GoEngine" "${HOME}/Gofer"; do
    if [[ -f "${d}/scripts/train-loop-v3.sh" ]]; then
      echo "$d"
      return
    fi
  done
  echo "${HOME}/GoEngine"
}

try_existing_gpu_venv() {
  local candidate="${GOFER_USE_VENV:-}"
  if [[ -z "$candidate" && -f /opt/venv/bin/activate ]]; then
    candidate="/opt/venv"
  fi
  [[ -n "$candidate" && -f "${candidate}/bin/activate" ]] || return 1
  if "${candidate}/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    VENV_DIR="$candidate"
    log "reusing existing GPU venv: ${VENV_DIR} ($("${candidate}/bin/python" -c 'import torch; print(torch.__version__)'))"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install -r "${GOFER_ROOT}/training/requirements.txt"
    pip install pytest
    python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device0", torch.cuda.get_device_name(0))
PY
    return 0
  fi
  return 1
}

install_apt_basics() {
  if ! command -v apt-get >/dev/null 2>&1; then
    log "no apt-get — install git curl python3 venv gcc tmux manually if missing"
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git curl build-essential tmux ca-certificates \
      python3 python3-venv python3-pip \
      python3.11 python3.11-venv 2>/dev/null || \
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git curl build-essential tmux ca-certificates python3 python3-venv python3-pip
  else
    log "no sudo — assuming provider preinstalled build deps"
  fi
}

install_go() {
  if command -v go >/dev/null 2>&1; then
    log "go present: $(go version)"
    return 0
  fi
  local tgz="go${GO_VERSION}.linux-amd64.tar.gz"
  log "installing go ${GO_VERSION} to /usr/local/go"
  curl -fsSL "https://go.dev/dl/${tgz}" -o "/tmp/${tgz}"
  if command -v sudo >/dev/null 2>&1; then
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "/tmp/${tgz}"
  else
    rm -rf "${HOME}/.local/go"
    tar -C "${HOME}/.local" -xzf "/tmp/${tgz}"
    export PATH="${HOME}/.local/go/bin:${PATH}"
    grep -q '.local/go/bin' "${HOME}/.bashrc" 2>/dev/null || \
      echo 'export PATH=$HOME/.local/go/bin:$PATH' >>"${HOME}/.bashrc"
  fi
  export PATH="/usr/local/go/bin:${PATH}"
  go version
}

clone_or_update_repo() {
  if [[ -d "${GOFER_ROOT}/.git" ]]; then
    log "updating ${GOFER_ROOT}"
    git -C "${GOFER_ROOT}" fetch origin
    git -C "${GOFER_ROOT}" checkout "$GOFER_BRANCH"
    git -C "${GOFER_ROOT}" pull --ff-only origin "$GOFER_BRANCH" || true
  else
    log "cloning ${GOFER_REPO_URL} -> ${GOFER_ROOT}"
    git clone --branch "$GOFER_BRANCH" --depth 1 "$GOFER_REPO_URL" "$GOFER_ROOT"
  fi
}

ensure_ort_lib() {
  local art="${GOFER_ROOT}/.tectonix/artifacts/onnxruntime-linux-x64-${ORT_VERSION}"
  local so="${art}/lib/libonnxruntime.so.${ORT_VERSION}"
  if [[ -f "$so" ]]; then
    export ONNXRUNTIME_SHARED_LIBRARY_PATH="$so"
    return 0
  fi
  log "fetching ONNX Runtime ${ORT_VERSION} (in-process CPU inference)"
  mkdir -p "${GOFER_ROOT}/.tectonix/artifacts"
  local tgz="${GOFER_ROOT}/.tectonix/artifacts/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" -o "$tgz"
  tar -xzf "$tgz" -C "${GOFER_ROOT}/.tectonix/artifacts"
  export ONNXRUNTIME_SHARED_LIBRARY_PATH="$so"
}

setup_venv_and_torch() {
  local py="$1" backend="$2" rocm_ver="$3"
  if try_existing_gpu_venv; then
    return 0
  fi
  log "venv ${VENV_DIR} python=${py} gpu_backend=${backend} rocm=${rocm_ver}"
  "$py" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip wheel
  case "$backend" in
    rocm)
      local idx
      idx="$(rocm_pip_index "$rocm_ver")"
      log "pip torch from ${idx}"
      pip install torch torchvision torchaudio --index-url "$idx"
      ;;
    cuda)
      log "pip torch (cuda default index)"
      pip install torch torchvision torchaudio
      ;;
    *)
      log "pip torch CPU"
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      ;;
  esac
  pip install -r "${GOFER_ROOT}/training/requirements.txt"
  pip install pytest
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY
}

build_gofer() {
  cd "$GOFER_ROOT"
  export PATH="/usr/local/go/bin:${HOME}/.local/go/bin:${PATH}"
  export PYTHONPATH="${GOFER_ROOT}:${PYTHONPATH:-}"
  ensure_ort_lib
  log "building bin/gofer (-tags=onnx)"
  CGO_ENABLED=1 go build -tags=onnx -o bin/gofer ./cmd/gofer
  go test ./... -short -count=1
  pytest training/ -q
}

write_train_env() {
  local nproc_val="$1"
  local env_file="${GOFER_ROOT}/.gofer-train.env"
  cat >"$env_file" <<EOF
# Source before training: source ${env_file}
export GOENGINE_ROOT=${GOFER_ROOT}
export GOFER_ROOT=${GOFER_ROOT}
export PATH=/usr/local/go/bin:\${HOME}/.local/go/bin:\${PATH}
export PYTHONPATH=${GOFER_ROOT}:\${PYTHONPATH:-}
export ONNXRUNTIME_SHARED_LIBRARY_PATH=${GOFER_ROOT}/.tectonix/artifacts/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}

# Secure default — no HTTP sidecar
export EVAL_BACKEND=inprocess

# Use GPU 0 for PyTorch training; self-play/arena use CPU cores
export HIP_VISIBLE_DEVICES=\${HIP_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}

# Scale to box (override after scope review)
export PARALLEL=\${PARALLEL:-${nproc_val}}
export WEEK_DAYS=\${WEEK_DAYS:-14}
export NEW_SELFPLAY_PER_CYCLE=\${NEW_SELFPLAY_PER_CYCLE:-500}
export SELFPLAY_FULL_PLAYOUTS=\${SELFPLAY_FULL_PLAYOUTS:-400}
export SELFPLAY_FAST_PLAYOUTS=\${SELFPLAY_FAST_PLAYOUTS:-100}
export ARENA_GAMES=\${ARENA_GAMES:-400}
export ARENA_PLAYOUTS=\${ARENA_PLAYOUTS:-400}
export REPLAY_MAX=\${REPLAY_MAX:-50000}
export GATING_MODE=\${GATING_MODE:-normal}
EOF
  log "wrote ${env_file}"
}

run_smoke() {
  cd "$GOFER_ROOT"
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi
  source "${GOFER_ROOT}/.gofer-train.env"
  export SKIP_PARITY=1
  export MAX_CYCLES=1
  export NEW_SELFPLAY_PER_CYCLE=20
  export ARENA_GAMES=20
  export PARALLEL="${PARALLEL:-8}"
  log "smoke cycle (1 cycle, small counts)"
  bash scripts/lightsail-inprocess-cycle.sh
}

main() {
  local gpu_backend rocm_ver python_best py nproc_val found_repo
  gpu_backend="$(read_scope gpu_backend "")"
  [[ -z "$gpu_backend" || "$gpu_backend" == "none" ]] && gpu_backend="$(detect_gpu_backend)"
  rocm_ver="$(read_scope rocm_version none)"
  python_best="$(read_scope python_best none)"
  nproc_val="$(read_scope nproc "$(nproc 2>/dev/null || echo 8)")"
  found_repo="$(read_scope found_repo none)"
  GOFER_ROOT="$(detect_gofer_root "$found_repo")"
  GOENGINE_ROOT="$GOFER_ROOT"
  VENV_DIR="${GOFER_VENV:-${GOFER_ROOT}/.venv311}"

  log "goengine_root=${GOFER_ROOT} gpu_backend=${gpu_backend} rocm=${rocm_ver} nproc=${nproc_val}"

  install_apt_basics
  py="$(pick_python "$python_best")"
  [[ -n "$py" ]] || { echo "no python3 found"; exit 1; }

  install_go
  clone_or_update_repo
  setup_venv_and_torch "$py" "$gpu_backend" "$rocm_ver"
  build_gofer
  write_train_env "$nproc_val"

  log "setup complete (directory: ${GOFER_ROOT})"
  log "activate: source ${VENV_DIR}/bin/activate && source ${GOFER_ROOT}/.gofer-train.env"
  log "start training (tmux recommended):"
  echo "  tmux new -s gofer"
  echo "  cd ${GOFER_ROOT} && source ${VENV_DIR}/bin/activate && source .gofer-train.env"
  echo "  bash scripts/train-loop-v3.sh 2>&1 | tee train-v3.log"

  if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    run_smoke
  else
    log "SKIP_SMOKE=1 — skipped smoke cycle"
  fi
}

main "$@"
