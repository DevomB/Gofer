# Models

## gofer-9x9.onnx

| | |
|--|--|
| Board | 9×9, komi 7.5, area scoring, positional superko |
| Architecture | 6 residual blocks × 64 channels, global-pooling blocks, KataGo-style heads (`training/model.py`) |
| Parameters | 476,932 |
| Inputs | `spatial` [N, 13, 9, 9], `global` [N, 4] (feature schema 3, see [docs/architecture.md](../docs/architecture.md)) |
| Outputs | `policy` [N, 82] logits (81 points + pass), `value` [N] in [−1, 1] for the side to move |
| Training | Generation 17 of a from-scratch self-play run: 10,200 games, 197k training positions, 3.6 h wall time on a 4-vCPU cloud VM, no human data |

Strength is reported in the [README](../README.md#results) and the full run
summary is in [docs/results](../docs/results/).

Reproduce:

```bash
.venv/bin/python -m training.loop --run runs/9x9 --gofer bin/gofer --gens 17
cp runs/9x9/models/gen-0017.onnx models/gofer-9x9.onnx
```

Self-play is stochastic, so a rerun produces a network of similar but not
identical strength.
