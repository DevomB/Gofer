"""Export Gofer bootstrap net to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GoferBootstrapNet, BOARD_SIZE, POLICY_SIZE


class PolicyValueNet(torch.nn.Module):
    """Policy + value only — skips ownership head at inference."""

    def __init__(self, net: GoferBootstrapNet) -> None:
        super().__init__()
        self.net = net

    def forward(
        self, spatial_input: torch.Tensor, global_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value, _ = self.net(spatial_input, global_input)
        return logits, value


def export_onnx(
    out_path: Path,
    checkpoint: Path | None = None,
    seed: int = 42,
    *,
    with_ownership: bool = False,
) -> None:
    torch.manual_seed(seed)
    net = GoferBootstrapNet()
    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
    net.eval()

    spatial = torch.zeros(1, 8, BOARD_SIZE, BOARD_SIZE)
    global_in = torch.zeros(1, 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if with_ownership:
        export_model = net
        output_names = ["policy_logits", "value", "ownership"]
        dynamic_axes = {
            "spatial_input": {0: "batch"},
            "global_input": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
            "ownership": {0: "batch"},
        }
    else:
        export_model = PolicyValueNet(net)
        output_names = ["policy_logits", "value"]
        dynamic_axes = {
            "spatial_input": {0: "batch"},
            "global_input": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        }
    torch.onnx.export(
        export_model,
        (spatial, global_in),
        str(out_path),
        input_names=["spatial_input", "global_input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        dynamo=False,
    )
    heads = "policy+value+ownership" if with_ownership else "policy+value"
    print(f"wrote {out_path} policy_size={POLICY_SIZE} outputs={heads}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models/gofer-9x9-bootstrap.onnx")
    p.add_argument("--checkpoint", default="", help="optional .pt checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--with-ownership",
        action="store_true",
        help="export all three heads (training/debug); default is policy+value only",
    )
    args = p.parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    export_onnx(Path(args.out), ckpt, args.seed, with_ownership=args.with_ownership)


if __name__ == "__main__":
    main()
