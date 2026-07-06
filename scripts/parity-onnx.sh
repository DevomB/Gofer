#!/usr/bin/env bash
# ONNX parity harness: Python inference_server.Session vs Go onnxruntime_go.
# Requires onnxruntime==1.26.0 on Python side (matches onnxruntime_go v1.31.0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LIMIT="${GOFER_PARITY_LIMIT:-500}"
MODEL="${GOFER_PARITY_MODEL:-models/gofer-9x9-best.onnx}"
SAMPLES="${GOFER_PARITY_SAMPLES:-training/data/samples.jsonl}"
REF="${GOFER_PARITY_REF:-.tectonix/reports/parity-ref.jsonl}"
ORT_VERSION="1.26.0"

if [[ ! -f "$MODEL" ]]; then
  MODEL="models/gofer-9x9-bootstrap.onnx"
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing ONNX model; run: python training/export_onnx.py --out models/gofer-9x9-bootstrap.onnx"
  exit 1
fi
if [[ ! -f "$SAMPLES" ]]; then
  echo "missing samples file: $SAMPLES"
  exit 1
fi

echo "==> ORT version due diligence"
PY_ORT="$(python -c 'import onnxruntime as ort; print(ort.__version__)' 2>/dev/null || true)"
echo "    python onnxruntime installed: ${PY_ORT:-not installed}"
echo "    target (pinned): onnxruntime==${ORT_VERSION}"
echo "    go wrapper: github.com/yalue/onnxruntime_go v1.31.0 (ORT C API ${ORT_VERSION})"

if [[ "${PY_ORT:-}" != "${ORT_VERSION}" ]]; then
  echo "==> installing onnxruntime==${ORT_VERSION} for parity run"
  pip install -q "onnxruntime==${ORT_VERSION}"
fi

echo "==> resolve Go ORT shared library"
GOMODCACHE="$(go env GOMODCACHE)"
export ONNXRUNTIME_GO_MODROOT="${GOMODCACHE}/github.com/yalue/onnxruntime_go@v1.31.0"
go get github.com/yalue/onnxruntime_go@v1.31.0

if [[ -z "${ONNXRUNTIME_SHARED_LIBRARY_PATH:-}" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64|Linux-amd64)
      ART="${ROOT}/.tectonix/artifacts/onnxruntime-linux-x64-${ORT_VERSION}"
      SO="${ART}/lib/libonnxruntime.so.${ORT_VERSION}"
      if [[ ! -f "$SO" ]]; then
        echo "==> downloading ORT ${ORT_VERSION} linux/amd64 (no bundled .so in onnxruntime_go)"
        mkdir -p "${ROOT}/.tectonix/artifacts"
        TGZ="${ROOT}/.tectonix/artifacts/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
        URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
        curl -fsSL "$URL" -o "$TGZ"
        rm -rf "$ART"
        tar -xzf "$TGZ" -C "${ROOT}/.tectonix/artifacts"
      fi
      export ONNXRUNTIME_SHARED_LIBRARY_PATH="$SO"
      ;;
    MINGW*-x86_64|MSYS*-x86_64|CYGWIN*-x86_64)
      DLL="${ONNXRUNTIME_GO_MODROOT}/test_data/onnxruntime.dll"
      if [[ ! -f "$DLL" ]]; then
        echo "missing bundled Windows DLL at $DLL"
        exit 1
      fi
      export ONNXRUNTIME_SHARED_LIBRARY_PATH="$DLL"
      ;;
    *)
      echo "set ONNXRUNTIME_SHARED_LIBRARY_PATH to libonnxruntime for $(uname -s)-$(uname -m)"
      exit 1
      ;;
  esac
fi
echo "    ONNXRUNTIME_SHARED_LIBRARY_PATH=${ONNXRUNTIME_SHARED_LIBRARY_PATH}"

echo "==> python reference path (inference_server.Session)"
python training/parity_harness.py --model "$MODEL" --samples "$SAMPLES" --limit "$LIMIT" --out "$REF"

echo "==> go ORT path (onnx build tag)"
CGO_ENABLED=1 go test -tags=onnx ./cmd/gofer/ -run TestONNXParityPythonRef -count=1 -v

echo "==> done — parity passed on ${LIMIT} positions"
