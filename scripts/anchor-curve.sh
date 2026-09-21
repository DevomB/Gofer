#!/usr/bin/env bash
# Absolute strength curve: every saved generation against the heuristic, with
# the flags the seed gate used, so each number is directly comparable to
# generation 1's recorded Elo.
#
#   bash scripts/anchor-curve.sh                      # defaults below
#   RUN=cpu GAMES=40 bash scripts/anchor-curve.sh
#   BUCKET= RUN=cpu bash scripts/anchor-curve.sh      # local run dir, no S3
#
# The pipeline takes this measurement itself when gating.anchor_every is set.
# This is for runs that did not, and for reading a run while it is still going.
#
# Run it on a machine that is NOT running the pipeline: a 400-playout arena
# contends for the same cores and corrupts the run's per-stage timings. Results
# are cached per generation, so re-running only measures what is new.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUCKET="${BUCKET-gofer-runs-211125325681}"
RUN="${RUN:-cpu}"
GAMES="${GAMES:-40}"
PLAYOUTS="${PLAYOUTS:-400}"
SEED="${SEED:-4242}"
OUT="${OUT:-.tectonix/reports/anchors}"

# .exe first: on Windows a stale extensionless bin/gofer from an older build
# shadows the real one, and the difference only shows up as "rebuild with
# -tags=onnx" once an arena is already running.
BIN="${BIN:-}"
if [[ -z "$BIN" ]]; then
  for cand in bin/gofer.exe bin/gofer; do
    [[ -f "$cand" ]] && { BIN="$cand"; break; }
  done
fi
[[ -n "$BIN" && -f "$BIN" ]] || { echo "no engine at bin/gofer[.exe]; build it first" >&2; exit 1; }

# The in-process backend needs the ORT shared library; find the pinned one.
if [[ -z "${ONNXRUNTIME_SHARED_LIBRARY_PATH:-}" ]]; then
  lib="$(ls .tectonix/artifacts/onnxruntime-*/lib/onnxruntime.dll \
            .tectonix/artifacts/onnxruntime-*/lib/libonnxruntime.so.* 2>/dev/null | head -1 || true)"
  [[ -n "$lib" ]] && export ONNXRUNTIME_SHARED_LIBRARY_PATH="$ROOT/$lib"
fi

MODELS="runs/$RUN/models"
mkdir -p "$MODELS" "$OUT"
if [[ -n "$BUCKET" ]]; then
  aws s3 sync "s3://$BUCKET/runs/$RUN/models/" "$MODELS/" \
    --exclude '*' --include 'gen-*.onnx' --only-show-errors
fi

shopt -s nullglob
gens=("$MODELS"/gen-*.onnx)
(( ${#gens[@]} )) || { echo "no gen-*.onnx under $MODELS" >&2; exit 1; }

for onnx in "${gens[@]}"; do
  tag="$(basename "$onnx" .onnx)"
  rep="$OUT/anchor-$RUN-$tag.json"
  if [[ ! -f "$rep" ]]; then
    echo "measuring $tag ($GAMES games @ $PLAYOUTS playouts)..." >&2
    # -arena-play-all: the eval names differ here, so without it the in-match
    # reject stop fires and truncates the sample to whatever it had seen.
    "$BIN" -arena -games "$GAMES" -size 9 -komi 6.5 -playouts "$PLAYOUTS" \
      -black-eval heuristic -white-eval onnx -eval-backend inprocess \
      -model "$onnx" -arena-enhanced none -arena-play-all \
      -seed "$SEED" -json "$rep" >/dev/null
  fi
done

# Elo comes from the pipeline's own stats module: a second copy of the formula
# here is a second thing to drift.
python - "$OUT" "$RUN" <<'PY'
import json, sys
from pathlib import Path
from training.pipeline import stats

out, run = Path(sys.argv[1]), sys.argv[2]
# One 40-game arena carries an Elo interval about 200 wide, so the interval is
# the result and the point estimate on its own is not.
print(f"{'gen':<10}{'score':>8}{'elo':>8}{'games':>7}   95% CI (Elo)")
for rep in sorted(out.glob(f"anchor-{run}-gen-*.json")):
    t = stats.tally_from_arena(json.loads(rep.read_text()))
    elo, lo, hi = t.elo()
    tag = rep.stem.split("-", 2)[2]
    print(f"{tag:<10}{t.score:>8.3f}{elo:>+8.0f}{t.games:>7}   [{lo:+.0f}, {hi:+.0f}]")
PY
