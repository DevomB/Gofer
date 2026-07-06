#!/usr/bin/env bash
# LEGACY — superseded by scripts/train-loop-v3.sh (ML pipeline v3, ADR 0003).
# This file is a compatibility alias only; do not extend it.
# Kept so old docs/commands that reference weekly-train-loop.sh still work.
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/gating.env"
exec bash "$ROOT/scripts/train-loop-v3.sh"
