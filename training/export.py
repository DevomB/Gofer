"""Export a checkpoint to ONNX (inputs spatial/global, outputs policy/value)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.features import NUM_GLOBALS, NUM_PLANES
from training.model import GoferNet


class _Inference(torch.nn.Module):
    def __init__(self, net: GoferNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, spatial: torch.Tensor, glob: torch.Tensor):
        policy, value, _ = self.net(spatial, glob)
        return policy, value


def export(model: GoferNet, path: Path, verify: bool = True) -> None:
    model = model.cpu().eval()
    s = model.cfg.board_size
    spatial = torch.zeros(2, NUM_PLANES, s, s)
    glob = torch.zeros(2, NUM_GLOBALS)
    tmp = path.with_suffix(".tmp.onnx")
    # The wrapper must be in eval mode too: export traces (and afterwards
    # restores) the mode of the module it is given, BatchNorm included.
    torch.onnx.export(
        _Inference(model).eval(),
        (spatial, glob),
        tmp,
        input_names=["spatial", "global"],
        output_names=["policy", "value"],
        dynamic_axes={"spatial": {0: "batch"}, "global": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    model.eval()
    if verify:
        verify_onnx(model, tmp)
    tmp.replace(path)


def verify_onnx(model: GoferNet, path: Path, atol: float = 1e-3) -> None:
    import onnxruntime as ort

    s = model.cfg.board_size
    rng = np.random.default_rng(0)
    spatial = (rng.random((5, NUM_PLANES, s, s)) < 0.3).astype(np.float32)
    glob = rng.standard_normal((5, NUM_GLOBALS)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    p_onnx, v_onnx = sess.run(None, {"spatial": spatial, "global": glob})
    with torch.no_grad():
        p_t, v_t, _ = model(torch.from_numpy(spatial), torch.from_numpy(glob))
    if not (np.allclose(p_onnx, p_t.numpy(), atol=atol) and np.allclose(v_onnx, v_t.numpy(), atol=atol)):
        raise RuntimeError(f"ONNX export of {path} does not match PyTorch")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    from training.model import ModelConfig

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GoferNet(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    export(model, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
