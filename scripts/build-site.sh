#!/usr/bin/env bash
# Build the static site in web/: the in-browser engine (WebAssembly + the
# released network). The page itself (index.html, site.css, site.js) is
# hand-written and needs no build step. Outputs are committed so any static
# host can serve web/ as-is; rerun this after changing the engine or
# models/gofer-9x9.onnx.
#
#   scripts/build-site.sh            # then: python3 -m http.server -d web 8000
#
# The page's data (web/data/*.json) comes from a training run instead:
#
#   go run -tags onnx ./cmd/sitedata -run runs/9x9
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GOOS=js GOARCH=wasm go build -trimpath -ldflags="-s -w" -o web/engine/gofer.wasm ./cmd/gofer-wasm
WASM_EXEC="$(go env GOROOT)/lib/wasm/wasm_exec.js"
[[ -f "$WASM_EXEC" ]] || WASM_EXEC="$(go env GOROOT)/misc/wasm/wasm_exec.js" # Go < 1.24
cp "$WASM_EXEC" web/engine/wasm_exec.js
cp models/gofer-9x9.onnx web/engine/gofer-9x9.onnx

ls -l web/engine
