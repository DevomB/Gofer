#!/usr/bin/env bash
# Build the paper PDF: paper/gofer-v4-pipeline.pdf
#
#   bash paper/build.sh              # build (tectonic preferred, latexmk fallback)
#   bash paper/build.sh --analysis   # also recompute the exact gate tables/figures first
#
# Tectonic (https://tectonic-typesetting.github.io) is a single binary that
# fetches the LaTeX packages it needs on first use; set TECTONIC=/path/to/tectonic
# if it is not on PATH. latexmk needs a full TeX Live / MiKTeX installation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="gofer-v4-pipeline.pdf"

if [[ "${1:-}" == "--analysis" ]]; then
  (cd "$ROOT" && python paper/analysis/gate_oc.py)
fi

cd "$HERE"
TECTONIC="${TECTONIC:-$(command -v tectonic || true)}"
if [[ -n "$TECTONIC" ]]; then
  "$TECTONIC" --keep-logs main.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  echo "need tectonic (set TECTONIC=...) or latexmk" >&2
  exit 1
fi
mv -f main.pdf "$OUT"
echo "wrote paper/$OUT"
