"""HTTP ONNX Runtime sidecar for Gofer batched eval."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

SCHEMA_VERSION = 2
BOARD_SIZE = 9
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1
LOG_EVERY_N = 50


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def pick_providers() -> list[str]:
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        try:
            import torch  # noqa: F401

            if torch.cuda.is_available():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
    return ["CPUExecutionProvider"]


class Session:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.providers = pick_providers()
        self.sess = ort.InferenceSession(model_path, providers=self.providers)
        self.spatial_name = "spatial_input"
        self.global_name = "global_input"
        self.request_count = 0

    def reload(self, model_path: str) -> None:
        self.model_path = model_path
        self.sess = ort.InferenceSession(model_path, providers=self.providers)

    def eval_batch(self, spatial: list[list[float]], globals_: list[list[float]]) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        batch = len(spatial)
        sp = np.array(spatial, dtype=np.float32).reshape(batch, 8, BOARD_SIZE, BOARD_SIZE)
        gl = np.array(globals_, dtype=np.float32).reshape(batch, 4)
        feeds = {self.spatial_name: sp, self.global_name: gl}
        logits, value = self.sess.run(None, feeds)
        results = []
        for i in range(batch):
            policy = softmax(logits[i]).astype(np.float32).tolist()
            if len(policy) != POLICY_SIZE:
                raise ValueError(f"policy len {len(policy)} != {POLICY_SIZE}")
            results.append({"value": float(value[i]), "policy": policy})
        self.request_count += 1
        if self.request_count % LOG_EVERY_N == 0:
            ms = (time.perf_counter() - t0) * 1000
            print(
                f"sidecar batch={batch} latency_ms={ms:.1f} requests={self.request_count}",
                file=sys.stderr,
            )
        return results


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            body = json.dumps(
                {
                    "status": "ok",
                    "schema_version": SCHEMA_VERSION,
                    "policy_size": POLICY_SIZE,
                    "spatial_shape": [8, BOARD_SIZE, BOARD_SIZE],
                    "global_shape": [4],
                    "model": session.model_path,
                    "providers": session.providers,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/v1/eval":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw)
                if req.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("schema_version mismatch")
                results = session.eval_batch(req["spatial"], req["globals"])
                body = json.dumps({"results": results}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                msg = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Gofer ONNX inference sidecar")
    p.add_argument("--model", required=True, help="path to .onnx model")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    model_path = str(Path(args.model))
    session = Session(model_path)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))

    def on_hup(_signum: int, _frame: object) -> None:
        session.reload(model_path)
        print(f"sidecar reloaded model={model_path}", file=sys.stderr)

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, on_hup)

    print(
        f"sidecar listening on http://127.0.0.1:{args.port} model={model_path} providers={session.providers}",
        file=sys.stderr,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
