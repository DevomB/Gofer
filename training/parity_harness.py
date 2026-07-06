"""ONNX parity harness — Python reference path (inference_server.Session, no HTTP)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import onnxruntime as ort

# Reuse sidecar session + softmax so parity matches production inference code.
from inference_server import Session, BOARD_SIZE, POLICY_SIZE  # noqa: E402


def load_positions(samples_path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with samples_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "header":
                continue
            if "features_spatial" not in row or "features_global" not in row:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Python ORT reference for Go parity harness")
    p.add_argument("--model", required=True, help="path to policy+value ONNX export")
    p.add_argument("--samples", default="training/data/samples.jsonl")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--out", default=".tectonix/reports/parity-ref.jsonl")
    args = p.parse_args()

    print(f"onnxruntime python {ort.__version__}", file=sys.stderr)
    if ort.__version__ != "1.26.0":
        print(
            f"warning: parity harness expects onnxruntime==1.26.0 (matches onnxruntime_go v1.31.0); got {ort.__version__}",
            file=sys.stderr,
        )

    positions = load_positions(Path(args.samples), args.limit)
    if len(positions) < 100:
        print(f"need at least 100 positions, got {len(positions)}", file=sys.stderr)
        return 1

    session = Session(str(Path(args.model)))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_ns = 0
    with out_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(positions):
            spatial = row["features_spatial"]
            globals_ = row["features_global"]
            if len(spatial) != 8 * BOARD_SIZE * BOARD_SIZE:
                print(f"skip row {i}: bad spatial len {len(spatial)}", file=sys.stderr)
                continue
            if len(globals_) != 4:
                print(f"skip row {i}: bad global len {len(globals_)}", file=sys.stderr)
                continue
            t0 = time.perf_counter_ns()
            results = session.eval_batch([spatial], [globals_])
            total_ns += time.perf_counter_ns() - t0
            r = results[0]
            if len(r["policy"]) != POLICY_SIZE:
                print(f"skip row {i}: policy len {len(r['policy'])}", file=sys.stderr)
                continue
            rec = {
                "index": i,
                "board_hash": row.get("board_hash"),
                "value": r["value"],
                "policy": r["policy"],
            }
            out.write(json.dumps(rec) + "\n")

    n = len(positions)
    ms_per = (total_ns / n) / 1e6
    print(f"python_ref positions={n} ms_per_pos={ms_per:.4f} out={out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
