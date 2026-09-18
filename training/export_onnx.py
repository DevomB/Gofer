"""Export a Gofer checkpoint to ONNX (any architecture).

The architecture is read from checkpoint metadata (best.ckpt / last.pt, or the
best.ckpt sitting next to a bare best.pt) and otherwise inferred from the
state_dict shapes, so legacy champions export exactly as before.

Inference graphs expose only ``policy_logits`` and ``value`` (the Go engine's
contract, docs/model-input-schema.md); ``--with-ownership`` adds the ownership
head for debugging. After export the graph is checked against PyTorch with
ONNX Runtime and model metadata (arch, sha of weights, step) is embedded.

    python training/export_onnx.py --checkpoint training/state/run/best.pt --out models/cand.onnx
    python training/export_onnx.py --checkpoint best.pt --out cand.onnx --quantize-int8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gofer_train.nets import ArchConfig, build_net, forward_all, infer_arch, load_weights  # noqa: E402
from gofer_train.trainer import load_state  # noqa: E402
from model import BOARD_SIZE, POLICY_SIZE, GoferBootstrapNet  # noqa: E402,F401

OPSET = 18


class PolicyValueNet(torch.nn.Module):
    """Policy + value only — the inference contract."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, spatial_input: torch.Tensor, global_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = forward_all(self.net, spatial_input, global_input)
        return out.policy_logits, out.value


class PolicyValueOwnershipNet(PolicyValueNet):
    def forward(self, spatial_input: torch.Tensor, global_input: torch.Tensor):  # type: ignore[override]
        out = forward_all(self.net, spatial_input, global_input)
        return out.policy_logits, out.value, out.ownership


def load_for_export(checkpoint: Path | None, seed: int = 42, arch: ArchConfig | None = None):
    """(net in eval mode, arch, step). No checkpoint = seeded random init."""
    torch.manual_seed(seed)
    step = None
    if checkpoint and checkpoint.exists():
        state, found = load_state(checkpoint)
        arch = found or arch or infer_arch(state)
        net = build_net(arch)
        load_weights(net, state)
        side = checkpoint.with_suffix(".ckpt")
        if side.exists():
            step = torch.load(side, map_location="cpu", weights_only=False).get("step")
    else:
        arch = arch or ArchConfig()
        net = build_net(arch)
    net.eval()
    return net, arch, step


def verify(onnx_path: Path, wrapper: torch.nn.Module, arch: ArchConfig, *, n: int = 16, tol: float = 1e-3) -> float:
    """Max |torch - ORT| over random inputs for every output; raises if above ``tol``."""
    import onnxruntime as ort

    s = arch.board_size
    g = torch.Generator().manual_seed(0)
    spatial = (torch.rand(n, 8, s, s, generator=g) > 0.7).float()
    glob = torch.rand(n, 4, generator=g)
    wrapper.eval()
    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(spatial, glob)]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"spatial_input": spatial.numpy(), "global_input": glob.numpy()})
    err = max(float(np.max(np.abs(a - b))) for a, b in zip(ref, got))
    if err > tol:
        raise RuntimeError(f"ONNX parity failed: max abs diff {err:.2e} > {tol:.0e}")
    return err


def _embed_metadata(path: Path, meta: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(path))
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(model, str(path))


def export_onnx(
    out_path: Path,
    checkpoint: Path | None = None,
    seed: int = 42,
    *,
    with_ownership: bool = False,
    check: bool = True,
    quantize_int8: bool = False,
) -> dict:
    net, arch, step = load_for_export(checkpoint, seed)
    s = arch.board_size
    spatial = torch.zeros(1, 8, s, s)
    global_in = torch.zeros(1, 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = ["policy_logits", "value"] + (["ownership"] if with_ownership else [])
    # eval() on the wrapper, not just the net: the exporter restores the wrapper's
    # train flag recursively afterwards, which would flip BN back to batch stats.
    wrapper = (PolicyValueOwnershipNet(net) if with_ownership else PolicyValueNet(net)).eval()
    torch.onnx.export(
        wrapper,
        (spatial, global_in),
        str(out_path),
        input_names=["spatial_input", "global_input"],
        output_names=outputs,
        dynamic_axes={name: {0: "batch"} for name in ["spatial_input", "global_input", *outputs]},
        opset_version=OPSET,
        dynamo=False,
    )
    weights_sha = hashlib.sha256(
        b"".join(t.detach().cpu().numpy().tobytes() for t in net.state_dict().values())
    ).hexdigest()[:16]
    meta = {
        "gofer.arch": json.dumps(arch.to_dict()),
        "gofer.weights_sha": weights_sha,
        "gofer.step": str(step if step is not None else ""),
        "gofer.checkpoint": str(checkpoint or ""),
    }
    _embed_metadata(out_path, meta)
    info = {"path": str(out_path), "arch": arch.name, "outputs": outputs, **meta}
    if check:
        info["parity_max_abs_diff"] = verify(out_path, wrapper, arch)
    if quantize_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q_path = out_path.with_name(out_path.stem + ".int8.onnx")
        quantize_dynamic(str(out_path), str(q_path), weight_type=QuantType.QInt8)
        info["int8_path"] = str(q_path)
    policy_size = s * s + 1
    heads = "+".join(outputs).replace("policy_logits", "policy")
    print(f"wrote {out_path} arch={arch.name} policy_size={policy_size} outputs={heads}"
          + (f" parity={info['parity_max_abs_diff']:.1e}" if "parity_max_abs_diff" in info else ""))
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="models/gofer-9x9-bootstrap.onnx")
    p.add_argument("--checkpoint", default="", help=".pt / .ckpt checkpoint (empty = seeded random init)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-ownership", action="store_true",
                   help="export ownership too (training/debug); default is policy+value only")
    p.add_argument("--no-check", action="store_true", help="skip the ONNX Runtime parity check")
    p.add_argument("--quantize-int8", action="store_true",
                   help="also write <out>.int8.onnx (dynamic int8 weights; CPU inference)")
    p.add_argument("--json", action="store_true", help="print export info as JSON")
    args = p.parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    info = export_onnx(
        Path(args.out),
        ckpt,
        args.seed,
        with_ownership=args.with_ownership,
        check=not args.no_check,
        quantize_int8=args.quantize_int8,
    )
    if args.json:
        print(json.dumps(info))


if __name__ == "__main__":
    main()
