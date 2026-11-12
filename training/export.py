"""Export a checkpoint to ONNX (inputs spatial/global, outputs policy/value)."""

from __future__ import annotations

import argparse
import contextlib
import logging
from pathlib import Path

import numpy as np
import torch

from training.features import NUM_GLOBALS, NUM_PLANES
from training.model import GoferNet

OPSET = 18


class _Inference(torch.nn.Module):
    def __init__(self, net: GoferNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, spatial: torch.Tensor, glob: torch.Tensor):
        policy, value, _ = self.net(spatial, glob)
        return policy, value


@contextlib.contextmanager
def _quiet_exporter():
    """The exporter warns about every torchvision operator it cannot register
    when torchvision is not installed; the network uses none of them."""
    log = logging.getLogger("torch.onnx._internal.exporter._registration")
    level = log.level
    log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(level)


def export(model: GoferNet, path: Path, verify: bool = True) -> None:
    model = model.cpu().eval()
    s = model.cfg.board_size
    spatial = torch.zeros(2, NUM_PLANES, s, s)
    glob = torch.zeros(2, NUM_GLOBALS)
    tmp = path.with_suffix(".tmp.onnx")
    batch = {0: torch.export.Dim.DYNAMIC}
    # The wrapper must be in eval mode too: the exporter traces the module it
    # is given in the mode it is in, BatchNorm included.
    with _quiet_exporter():
        program = torch.onnx.export(
            _Inference(model).eval(),
            (spatial, glob),
            input_names=["spatial", "global"],
            output_names=["policy", "value"],
            dynamic_shapes=(batch, batch),
            opset_version=OPSET,
            dynamo=True,
            verbose=False,
        )
    program.save(tmp)
    model.eval()
    if verify:
        verify_onnx(model, tmp)
    tmp.replace(path)


def verify_onnx(model: GoferNet, path: Path, atol: float = 1e-3) -> None:
    """Compare the exported graph with PyTorch on random positions. The batch
    sizes differ from the one the graph was traced with (2) and include 1, so
    a graph whose batch dimension came out fixed fails here, not in self-play."""
    import onnxruntime as ort

    s = model.cfg.board_size
    rng = np.random.default_rng(0)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 5):
        spatial = (rng.random((batch, NUM_PLANES, s, s)) < 0.3).astype(np.float32)
        glob = rng.standard_normal((batch, NUM_GLOBALS)).astype(np.float32)
        p_onnx, v_onnx = sess.run(None, {"spatial": spatial, "global": glob})
        with torch.no_grad():
            p_t, v_t, _ = model(torch.from_numpy(spatial), torch.from_numpy(glob))
        if p_onnx.shape != tuple(p_t.shape) or v_onnx.shape != tuple(v_t.shape):
            raise RuntimeError(f"ONNX export of {path} has output shapes {p_onnx.shape}, {v_onnx.shape} for a batch of {batch}")
        if not (np.allclose(p_onnx, p_t.numpy(), atol=atol) and np.allclose(v_onnx, v_t.numpy(), atol=atol)):
            raise RuntimeError(f"ONNX export of {path} does not match PyTorch")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=0,
                    help="board size to export for (default: the checkpoint's own). An ONNX file plays one "
                         "size; the weights are the same for every size")
    args = ap.parse_args(argv)
    from training.model import ModelConfig

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    if args.size:
        cfg.board_size = args.size
    model = GoferNet(cfg)
    model.load_state_dict(ck["model"])
    export(model, args.out)
    print(f"wrote {args.out} ({cfg.board_size}x{cfg.board_size})")


if __name__ == "__main__":
    main()
