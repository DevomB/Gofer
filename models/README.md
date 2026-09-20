# Models

## Tracked

`gofer-9x9-bootstrap.onnx` is the only model in git. It is a fixture, not a
trained net: random weights at the current export shape. Its job is to give the
default `-model` flag, `make sidecar`, `TestONNXParity` and CI something with
the right input and output signature to load. Regenerate it whenever the export
shape changes:

```bash
pip install -r training/requirements.txt
python training/export_onnx.py --out models/gofer-9x9-bootstrap.onnx
```

Input shapes: `spatial_input [N,8,9,9]`, `global_input [N,4]`. Full contract in
[docs/model-input-schema.md](../docs/model-input-schema.md).

## Everything else here is local

`models/champions/*.onnx` and any `gofer-9x9-{best,candidate}.onnx` are
gitignored build output. The pipeline writes a promoted champion into
`models/champions/` beside a `.json` card holding its gate statistics, Elo and
sha256, and records it in `index.json`, which names both `best` and
`previous_best` so a promotion can be rolled back
(`python -m training.pipeline rollback`). The card and index are trackable; the
weights are not, because published champions live in GitHub Releases.

A local champion is therefore reproducible from `index.json` plus a release
download, and a stale `.onnx` sitting in this directory is never authoritative.
