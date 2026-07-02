#!/usr/bin/env bash
# Deprecated compatibility wrapper. The v3 loop owns replay, resume, and promotion.
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/gating.env"
exec bash "$ROOT/scripts/train-loop-v3.sh"
