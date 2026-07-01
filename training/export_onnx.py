"""Export Gofer bootstrap net to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GoferBootstrapNet, BOARD_SIZE, POLICY_SIZE


def export_onnx(out_path: Path, checkpoint: Path | None = None, seed: int = 42) -> None:
    torch.manual_seed(seed)
    net = GoferBootstrapNet()
    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
    net.eval()

    spatial = torch.zeros(1, 8, BOARD_SIZE, BOARD_SIZE)
    global_in = torch.zeros(1, 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (spatial, global_in),
        str(out_path),
        input_names=["spatial_input", "global_input"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "spatial_input": {0: "batch"},
            "global_input": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    print(f"wrote {out_path} policy_size={POLICY_SIZE}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models/gofer-9x9-bootstrap.onnx")
    p.add_argument("--checkpoint", default="", help="optional .pt checkpoint")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    export_onnx(Path(args.out), ckpt, args.seed)


if __name__ == "__main__":
    main()
