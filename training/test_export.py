"""PyTorch vs ONNX Runtime export parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_onnx import export_onnx
from model import BOARD_SIZE, GoferBootstrapNet

ROOT = Path(__file__).resolve().parent


def test_export_ort_matches_pytorch(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pt"
    onnx_path = tmp_path / "model.onnx"

    net = GoferBootstrapNet()
    torch.save(net.state_dict(), ckpt)
    export_onnx(onnx_path, ckpt)

    spatial = torch.zeros(1, 8, BOARD_SIZE, BOARD_SIZE)
    global_in = torch.zeros(1, 4)
    net.eval()
    with torch.no_grad():
        logits_pt, value_pt = net(spatial, global_in)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {
        "spatial_input": spatial.numpy(),
        "global_input": global_in.numpy(),
    }
    logits_ort, value_ort = sess.run(None, feeds)

    assert np.max(np.abs(logits_pt.numpy() - logits_ort)) < 1e-4
    assert np.max(np.abs(value_pt.numpy() - value_ort)) < 1e-4
