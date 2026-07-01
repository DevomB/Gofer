# Bootstrap ONNX model

9×9 ResNet-small for Gofer v2.5. Regenerate:

```bash
pip install -r training/requirements.txt
python training/export_onnx.py --out models/gofer-9x9-bootstrap.onnx
```

Trained export:

```bash
python training/train_bootstrap.py --data training/data/samples.jsonl
python training/export_onnx.py --checkpoint training/checkpoints/best.pt --out models/gofer-9x9-bootstrap.onnx
```

Input shapes: `spatial_input [N,8,9,9]`, `global_input [N,4]`. See [docs/model-input-schema.md](../docs/model-input-schema.md).
