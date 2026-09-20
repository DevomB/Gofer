# Research traceability

Maps each mechanism in Wu (2020), *Accelerating Self-Play Learning in Go*
(arXiv:1902.10565), to where Gofer implements it and what holds it in place.
Gofer is not KataGo and shares no code with it.

**Legend:** `[PAPER]` in the paper · `[POST-PAPER]` later KataGo work ·
`[GOFER]` our own decision.

The engine is one `package main` under `cmd/gofer`; the learner is
`training/gofer_train`; the loop around them is `training/pipeline`.

## Search

| Mechanism | Where | Status | Evidence |
|---|---|---|---|
| PUCT / MCTS `[PAPER]` | `mcts.go`, `arena.go` | done | `TestPUCTFormula`, `TestDeterministicPlayout` |
| First-play urgency `[PAPER]` | `mcts.go` `selectChildLocked` | done | `TestSelectionPrefersMovesGoodForTheMover` |
| Dirichlet root noise `[PAPER]` | `evaluator.go` `blendDirichlet` | done | `noise_test.go` |
| Forced playouts `[PAPER]` k=2 | `mcts.go` `runForcedRootPlayouts` | done | `TestForcedRootPlayouts` |
| Policy target pruning `[PAPER]` | `mcts_policy.go` | done | `TestRootPolicyPruned` |
| Transposition table `[GOFER]` | `tt.go` | done | `tt_test.go` |
| LCB move selection `[POST-PAPER]` | — | deferred | see known-issues |
| Score maximization `[POST-PAPER]` | — | deferred | needs a trusted score head |

Forced playouts and policy target pruning are the two mechanisms whose
misimplementation is documented in [ADR 0008](decisions/0008-search-correctness.md):
the paper forces a floor only on root children that already have visits, and
applies it after the search, not before.

## Self-play

| Mechanism | Where | Status | Evidence |
|---|---|---|---|
| Playout cap randomization `[PAPER]` | `selfplay.go` | done | `selfplay_test.go` |
| Rules randomization `[PAPER]` | `selfplay.go` | done | `selfplay_test.go` |
| Board-size randomization `[PAPER]` 9–19 | `selfplay.go` | done | `selfplay_test.go` |
| Opponent-reply policy target `[PAPER]` | `Sample.PolicyNext` | done | `shard_test.go` |
| Ownership target `[PAPER]` | `scoring.go` `OwnershipLabel` | done | required by `WriteSampleShard` |
| Score belief pdf/cdf `[PAPER]` | `Sample.ScorePDF`/`ScoreCDF` | fields only | margin is trained, distribution is not |

`Sample.PolicyOpp` is the older field and holds the *previous* ply's policy, not
a reply target; the learner reads `PolicyNext`. See [ADR 0007](decisions/0007-pipeline-orchestrator.md).

## Learner

| Mechanism | Where | Status | Evidence |
|---|---|---|---|
| Global pooling `[PAPER]` | `gofer_train/nets.py` `kind="gpool"` | done | `test_gofer_train.py`, `training/configs/distill-gpool.toml` |
| Multi-head loss (policy/value/ownership/score) `[PAPER]` | `gofer_train/losses.py` | done | `test_gofer_train.py` |
| Masked policy loss on full-search rows `[PAPER]` | `gofer_train/losses.py` | done | `test_gofer_train.py` |
| D4 symmetry augmentation `[PAPER]` | `gofer_train/data.py` | done | `test_gofer_train.py` |
| Weight averaging `[PAPER]` (EMA, not SWA) | `gofer_train/trainer.py` | done | `test_gofer_train.py` |
| Distillation from the champion `[GOFER]` | `gofer_train/trainer.py` | done | `training/configs/distill-gpool.toml` |
| Progressive net scaling `[PAPER]` | — | measured, not adopted | [ADR 0005](decisions/0005-net-size-ablation.md) |
| Game-specific features (ladders, pass-alive) `[PAPER]` §4.2 | — | deferred | `docs/model-input-schema.md` fixes 8 planes |

## Loop

| Mechanism | Where | Status | Evidence |
|---|---|---|---|
| Champion/challenger gating `[PAPER]` | `match.go`, `pipeline/runner.py` | done | `match_test.go`, `test_pipeline.py` |
| SPRT sequential gate `[GOFER]` | `pipeline/stats.py` | done | `test_pipeline.py`, `pipeline/gate_oc.py` |
| Power-law replay window `[PAPER]` | `pipeline/replay_index.py` | done | `test_pipeline.py` |
