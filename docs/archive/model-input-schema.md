# Feature tensor schema (bootstrap net, v2.5)

Schema version: **2** (`FeatureSchemaVersion` in Go, `schema_version` in sidecar JSON).

## Spatial input

Shape: `[batch, 8, H, W]` — exported ONNX name `spatial_input`.

Row-major NCHW per board; Go sends flat `8×H×W` per row in `spatial[]`.

| Plane | Index | Value |
|-------|-------|-------|
| Own stones | 0 | 1 at current player's stones |
| Opponent stones | 1 | 1 at opponent stones |
| Empty | 2 | 1 at empty intersections |
| Ko point | 3 | 1 at simple-ko prohibited point |
| To-move | 4 | all 1 if Black to move, else 0 |
| History t-3 | 5 | 1 at stone played 3 ply ago |
| History t-2 | 6 | 1 at stone played 2 ply ago |
| History t-1 | 7 | 1 at stone played 1 ply ago |

Pass moves leave history planes zero at that ply.

## Global input

Shape: `[batch, 4]` — ONNX name `global_input`.

| Index | Feature |
|-------|---------|
| 0 | `komi / 10` |
| 1 | `move_num / (H×W + 1)` |
| 2 | 1 if Black to move else 0 |
| 3 | 1 if White to move else 0 |

## Outputs

| Head | Shape | Activation |
|------|-------|------------|
| Policy | `H×W + 1` | softmax (sidecar) |
| Value | 1 | tanh, current player perspective |

Pass index is last (`H×W`).

## Bootstrap 9×9

- Spatial: 8 × 9 × 9 = 648 floats
- Globals: 4 floats
- Policy: 82 floats

Golden hash: `cmd/gofer/testdata/features_golden_v2.json`
