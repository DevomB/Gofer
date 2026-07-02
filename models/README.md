# Bootstrap ONNX model

9×9 ResNet-small for Gofer. Production artifacts (v3):

| File | Role |
|------|------|
| `gofer-9x9-best.onnx` | Arena-verified deployed model |
| `gofer-9x9-candidate.onnx` | Pre-arena export this cycle |
| `gofer-9x9-bootstrap.onnx` | CLI compat alias → `best.onnx` |

State on disk: `training/state/best.pt` (matches deployed ONNX weights).

Regenerate fixture:

```bash
pip install -r training/requirements.txt
python training/export_onnx.py --out models/gofer-9x9-bootstrap.onnx
```

Trained export (resume from state):

```bash
python training/train_bootstrap.py --data training/data/replay.jsonl --resume training/state/best.pt --out-dir training/state
python training/export_onnx.py --checkpoint training/state/best.pt --out models/gofer-9x9-candidate.onnx
```

Input shapes: `spatial_input [N,8,9,9]`, `global_input [N,4]`. See [docs/model-input-schema.md](../docs/model-input-schema.md).
