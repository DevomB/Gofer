#!/usr/bin/env bash
# Download a small SGF corpus for supervised warm-start smoke conversion.
# Does not touch Lightsail or live training state.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${ROOT}/training/experiments/warmstart/raw"
SEVEN_Z="${ROOT}/.tectonix/artifacts/7zr.exe"
mkdir -p "${RAW}/cgos" "${RAW}/minigo" "${RAW}/aeb" "${ROOT}/.tectonix/artifacts"

log() { echo "[fetch-warmstart] $*"; }

extract_7z() {
  local archive="$1"
  local outdir="$2"
  if command -v 7z >/dev/null 2>&1; then
    7z x -y "-o${outdir}" "${archive}" >/dev/null
  elif command -v 7za >/dev/null 2>&1; then
    7za x -y "-o${outdir}" "${archive}" >/dev/null
  elif [[ -x "${SEVEN_Z}" ]]; then
    "${SEVEN_Z}" x -y "-o${outdir}" "${archive}" >/dev/null
  else
    return 1
  fi
}

fetch_cgos() {
  local base="http://www.yss-aya.com/cgos/9x9/archives"
  # Eight recent months — enough for ~100k rows after filtering.
  local archives=(
    9x9_2025_10.tar.bz2 9x9_2025_11.tar.bz2 9x9_2025_12.tar.bz2
    9x9_2026_01.tar.bz2 9x9_2026_02.tar.bz2 9x9_2026_03.tar.bz2
    9x9_2026_04.tar.bz2 9x9_2026_05.tar.bz2
  )
  for arc in "${archives[@]}"; do
    local dest="${RAW}/cgos/${arc}"
    if [[ -f "${dest}" ]]; then
      log "cgos skip ${arc}"
      continue
    fi
    log "cgos ${arc}"
    curl -fsSL "${base}/${arc}" -o "${dest}"
    tar -xjf "${dest}" -C "${RAW}/cgos"
  done
}

fetch_minigo() {
  local base="https://storage.googleapis.com/minigo-pub/v3-9x9/sgf"
  local archives=(
    0-16.tar.gz
    000017-choice-wolf.tar.gz
    000025-on-foal.tar.gz
  )
  for arc in "${archives[@]}"; do
    local dest="${RAW}/minigo/${arc}"
    if [[ -f "${dest}" ]]; then
      log "minigo skip ${arc}"
    else
      log "minigo ${arc}"
      curl -fsSL "${base}/${arc}" -o "${dest}"
    fi
    tar -xzf "${dest}" -C "${RAW}/minigo" 2>/dev/null || true
  done
}

fetch_aeb() {
  local dest="${RAW}/aeb/games.7z"
  if [[ ! -f "${dest}" ]]; then
    log "aeb games.7z"
    curl -fsSL "https://homepages.cwi.nl/~aeb/go/games/games.7z" -o "${dest}"
  fi
  if [[ ! -d "${RAW}/aeb/games" ]]; then
    if ! extract_7z "${dest}" "${RAW}/aeb"; then
      log "aeb skip extract (no 7z/7zr found)"
    fi
  fi
}

fetch_cgos
fetch_minigo
fetch_aeb
log "done: ${RAW}"
