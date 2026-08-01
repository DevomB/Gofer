#!/usr/bin/env bash
# Collect server facts for Gofer GPU training setup.
# Safe to paste into chat — no secrets, keys, or env dump.
#
# Usage (on GPU box):
#   bash scripts/scope-gpu-server.sh
#   bash scripts/scope-gpu-server.sh | tee ~/gpu-scope-report.txt
#
# Or copy this file alone and run before cloning the repo.
set -uo pipefail

_script_path="${BASH_SOURCE[0]:-${0:-}}"
if [[ -z "$_script_path" || "$_script_path" == bash || "$_script_path" == -bash ]]; then
  _script_dir="$(pwd)"
else
  _script_dir="$(cd "$(dirname "$_script_path")" && pwd)"
fi
unset _script_path
ROOT="$(cd "${_script_dir}/.." 2>/dev/null && pwd || pwd)"
unset _script_dir
if [[ -d "$ROOT/.git" && -f "$ROOT/scripts/train-loop-v3.sh" ]]; then
  REPORT_DIR="$ROOT/.tectonix/reports"
else
  REPORT_DIR="${HOME}"
fi
mkdir -p "$REPORT_DIR"
REPORT="${GOFER_SCOPE_REPORT:-${REPORT_DIR}/gpu-scope-report.txt}"

run_cmd() {
  local label="$1"
  shift
  echo "--- ${label} ---"
  if "$@" 2>&1; then
    :
  else
    echo "(exit $?)"
  fi
}

kv() {
  printf '%s=%s\n' "$1" "$2"
}

scope_body() {
  echo "=== GOFER GPU SCOPE REPORT ==="
  echo "generated_at=$(date -Is 2>/dev/null || date)"
  echo "hostname=$(hostname 2>/dev/null || echo unknown)"
  echo "report_path=${REPORT}"
  echo ""

  echo "=== USER ==="
  run_cmd "whoami" whoami
  run_cmd "id" id
  run_cmd "groups" groups
  run_cmd "umask" umask
  run_cmd "home disk" df -h "${HOME}" 2>/dev/null || df -h .

  echo ""
  echo "=== OS ==="
  if [[ -f /etc/os-release ]]; then
    run_cmd "os-release" cat /etc/os-release
  else
    echo "no /etc/os-release"
  fi
  run_cmd "uname" uname -a

  echo ""
  echo "=== CPU / MEMORY / DISK ==="
  run_cmd "nproc" nproc
  run_cmd "lscpu" bash -c 'lscpu 2>/dev/null | egrep "Model name|CPU\\(s\\)|Thread|Core|Socket" || true'
  run_cmd "free -h" free -h
  run_cmd "swapon" swapon --show 2>/dev/null || echo "no swap"
  run_cmd "disk" df -h / /tmp "${HOME}" 2>/dev/null || df -h

  echo ""
  echo "=== GPU HARDWARE ==="
  run_cmd "lspci" bash -c 'lspci 2>/dev/null | egrep -i "vga|3d|display|accelerator" || true'
  run_cmd "rocm-smi" bash -c 'rocm-smi 2>/dev/null || echo "rocm-smi not found"'
  run_cmd "nvidia-smi" bash -c 'nvidia-smi 2>/dev/null || echo "nvidia-smi not found"'

  echo ""
  echo "=== ROCM / CUDA TOOLING ==="
  run_cmd "rocminfo" bash -c 'rocminfo 2>/dev/null | head -60 || echo "rocminfo not found"'
  run_cmd "hipcc" bash -c 'command -v hipcc >/dev/null && hipcc --version || echo "hipcc not found"'
  run_cmd "nvcc" bash -c 'command -v nvcc >/dev/null && nvcc --version || echo "nvcc not found"'
  if [[ -d /opt/rocm ]]; then
    run_cmd "rocm version file" bash -c 'cat /opt/rocm/.info/version 2>/dev/null || ls /opt/rocm | head -10'
  fi

  echo ""
  echo "=== PYTHON ==="
  run_cmd "python3" bash -c 'python3 --version 2>/dev/null || echo "python3 missing"'
  for py in python3.12 python3.11 python3.10; do
    if command -v "$py" >/dev/null 2>&1; then
      echo "--- ${py} ---"
      "$py" --version 2>&1 || true
    fi
  done
  run_cmd "pip3" bash -c 'pip3 --version 2>/dev/null || echo "pip3 missing"'
  run_cmd "venv" bash -c 'python3 -m venv --help >/dev/null 2>&1 && echo "venv ok" || echo "venv missing"'

  echo ""
  echo "=== PYTORCH PROBE ==="
  python3 - <<'PY' 2>&1 || true
import sys
print("python_exe", sys.executable)
try:
    import torch
    print("torch_version", torch.__version__)
    print("torch_cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch_cuda_device_count", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"torch_cuda_device_{i}", torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch_probe", type(exc).__name__, exc)
PY

  echo ""
  echo "=== GO / BUILD TOOLS ==="
  run_cmd "go" bash -c 'go version 2>/dev/null || echo "go not installed"'
  run_cmd "git" bash -c 'git --version 2>/dev/null || echo "git missing"'
  run_cmd "curl" bash -c 'curl --version 2>/dev/null | head -1 || echo "curl missing"'
  run_cmd "gcc" bash -c 'gcc --version 2>/dev/null | head -1 || echo "gcc missing"'
  run_cmd "tmux" bash -c 'tmux -V 2>/dev/null || echo "tmux missing"'

  echo ""
  echo "=== NETWORK / EXPOSURE ==="
  run_cmd "listening tcp" bash -c 'ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null || echo "cannot list listeners"'
  run_cmd "ufw" bash -c 'command -v ufw >/dev/null && ufw status 2>/dev/null || echo "ufw n/a"'
  echo "note: do not paste provider login URLs with embedded tokens"

  echo ""
  echo "=== CONTAINER / VM HINTS ==="
  run_cmd "virt" bash -c 'systemd-detect-virt 2>/dev/null || echo "unknown"'
  [[ -f /.dockerenv ]] && echo "dockerenv=yes" || echo "dockerenv=no"

  echo ""
  echo "=== EXISTING GOFER INSTALL ==="
  found=""
  for d in "${ROOT}" "${HOME}/GoEngine" "${HOME}/Gofer"; do
    if [[ -f "${d}/scripts/train-loop-v3.sh" ]]; then
      found="$d"
      echo "found_repo=${d}"
      [[ -f "${d}/bin/gofer" ]] && echo "bin_gofer=yes" || echo "bin_gofer=no"
      [[ -f "${d}/models/gofer-9x9-best.onnx" ]] && echo "best_onnx=yes" || echo "best_onnx=no"
      [[ -f "${d}/training/state/best.pt" ]] && echo "best_pt=yes" || echo "best_pt=no"
      break
    fi
  done
  [[ -z "$found" ]] && echo "found_repo=none"

  echo ""
  echo "=== PARSEABLE SUMMARY ==="
  distro_id="unknown"
  distro_ver="unknown"
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    distro_id="${ID:-unknown}"
    distro_ver="${VERSION_ID:-unknown}"
  fi

  gpu_backend="none"
  gpu_count=0
  gpu_model="unknown"
  if command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
    gpu_backend="rocm"
    gpu_count="$(rocm-smi 2>/dev/null | awk '/^[0-9]+\s+[0-9]+\s+0x/{c++} END{print c+0}')"
    [[ "$gpu_count" == 0 ]] && gpu_count=1
    gpu_model="$(rocminfo 2>/dev/null | awk -F': ' '/Marketing Name/ && /AMD|Instinct|MI/{print $2; exit}')"
    [[ -z "$gpu_model" ]] && gpu_model="$(lspci 2>/dev/null | grep -i 'Processing accelerators' | sed 's/.*: //' | head -1)"
    [[ -z "$gpu_model" ]] && gpu_model="amd"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    gpu_backend="cuda"
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    gpu_model="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo nvidia)"
  fi

  rocm_ver="none"
  if [[ -f /opt/rocm/.info/version ]]; then
    rocm_ver="$(tr -d '\n' </opt/rocm/.info/version)"
  else
    for vf in /opt/rocm-*/.info/version; do
      [[ -f "$vf" ]] || continue
      rocm_ver="$(tr -d '\n' <"$vf")"
      break
    done
  fi
  if [[ "$rocm_ver" == none && command -v hipcc >/dev/null 2>&1 ]]; then
    rocm_ver="$(hipcc --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || echo unknown)"
  fi

  python_best="none"
  for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" >/dev/null 2>&1; then
      python_best="$py"
      break
    fi
  done

  public_listeners="unknown"
  if ss -tln 2>/dev/null | grep -qE '0\.0\.0\.0|\[::\]:'; then
    public_listeners="yes"
  elif ss -tln 2>/dev/null; then
    public_listeners="no"
  fi

  kv distro "${distro_id}-${distro_ver}"
  kv nproc "$(nproc 2>/dev/null || echo 1)"
  kv ram_gb "$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0)"
  kv disk_free_gb_home "$(df -BG "${HOME}" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}' || echo 0)"
  kv gpu_backend "$gpu_backend"
  kv gpu_count "$gpu_count"
  kv gpu_model "$gpu_model"
  kv rocm_version "$rocm_ver"
  kv python_best "$python_best"
  if command -v go >/dev/null 2>&1; then
    kv go_installed yes
    kv go_version "$(go version 2>/dev/null | awk '{print $3}')"
  else
    kv go_installed no
    kv go_version none
  fi
  kv public_listeners "$public_listeners"
  kv inprocess_recommended yes
  kv found_repo "${found:-none}"
}

scope_body | tee "$REPORT"
echo ""
echo "Wrote report: ${REPORT}"
echo "Share that file (not passwords). Then run: cd ~/GoEngine && bash scripts/setup-gpu-server.sh"
