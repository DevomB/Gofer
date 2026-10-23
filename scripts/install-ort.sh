#!/usr/bin/env bash
# Download ONNX Runtime into third_party/onnxruntime (CPU by default).
#
#   scripts/install-ort.sh          # CPU build
#   scripts/install-ort.sh gpu      # CUDA 12 build (needs an NVIDIA driver + CUDA 12 / cuDNN 9)
#
# The Go engine finds third_party/onnxruntime/lib automatically; elsewhere set
# ONNXRUNTIME_SHARED_LIBRARY_PATH to the libonnxruntime.so it prints.
set -euo pipefail

VERSION="${ORT_VERSION:-1.26.0}"
FLAVOR="${1:-cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) PLATFORM=linux-x64 ;;
  Linux-aarch64) PLATFORM=linux-aarch64 ;;
  Darwin-arm64) PLATFORM=osx-arm64 ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m); install ONNX Runtime manually" >&2; exit 1 ;;
esac
NAME="onnxruntime-${PLATFORM}-${VERSION}"
if [[ "$FLAVOR" == "gpu" ]]; then
  [[ "$PLATFORM" == linux-x64 ]] || { echo "GPU builds are linux-x64 only" >&2; exit 1; }
  NAME="onnxruntime-linux-x64-gpu-${VERSION}"
fi

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${NAME}.tgz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "downloading $URL"
curl -fsSL "$URL" -o "$TMP/ort.tgz"
tar -xzf "$TMP/ort.tgz" -C "$TMP"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$TMP/$NAME" "$DEST"
LIB="$(ls "$DEST"/lib/libonnxruntime.so."$VERSION" "$DEST"/lib/libonnxruntime."$VERSION".dylib 2>/dev/null | head -1)"
echo "installed $LIB"
echo "export ONNXRUNTIME_SHARED_LIBRARY_PATH=$LIB"
