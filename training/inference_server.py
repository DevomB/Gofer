"""HTTP ONNX Runtime sidecar for Gofer batched eval."""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import onnxruntime as ort

SCHEMA_VERSION = 2
BOARD_SIZE = 9
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class Session:
    def __init__(self, model_path: str) -> None:
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.spatial_name = "spatial_input"
        self.global_name = "global_input"

    def eval_batch(self, spatial: list[list[float]], globals_: list[list[float]]) -> list[dict[str, Any]]:
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
    session = Session(args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"sidecar listening on http://127.0.0.1:{args.port} model={args.model}")
    server.serve_forever()


if __name__ == "__main__":
    main()
