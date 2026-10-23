# v2 audit: why training did not work

Gofer v3 is a rewrite of the engine core and the training pipeline. This page
records what was wrong with v2 (commit `6083eb3`) and how each problem was
confirmed, so the reasons for the rewrite are reproducible rather than
anecdotal. Each item was measured on a 4-vCPU machine with the v2 code.

## Search (the root cause)

| # | Defect | Evidence | Effect |
|---|--------|----------|--------|
| 1 | **Values were backed up with the wrong sign.** A leaf stored the network value from *its own* side-to-move perspective, and the parent maximized it, so every node chose the move best for the opponent. | Position with Black far ahead: every Black move showed Q = -1 after one visit; the search put 738 of 800 visits on pass. | The engine played the opponent's best moves; self-play visit counts (the policy targets) pointed at bad moves. |
| 2 | **The root lost its `Expanded` flag on every fresh search.** `expandLocked` held a `*Node` into the arena slice, appended 82 children (capacity 64 → reallocation), then wrote `Expanded = true` into the discarded copy. The same happened to whichever node expanded when the arena crossed 128, 256, 512... | Fresh 400-playout search: 1 network evaluation, 0 child visits. Over GTP with 800 playouts it opened A9, B9, A8, C9, A7, which is just scan order. | No search on the first moves of every game; random nodes silently stopped growing. |
| 3 | **Transposition table without key check.** `Get` returned whatever entry sat in `hash & mask`. Pass leaves stones unchanged and the hash ignored side to move, so pass collided with its parent. | Pass child received the root's own value (+0.875 in the position above). | Wrong values for collisions; pass looked as good as the current position. |
| 4 | **Forced playouts flattened policy targets.** Every root child was forced to `2 + sqrt(prior·N)` visits — about 246 forced visits against a 200-playout budget — and the pruning step removed nothing. | Exported "full search" policy targets were near-uniform. | The policy head was trained toward uniform. |
| 5 | **The network ran on one core.** One batching goroutine ran batches serially and ONNX Runtime was pinned to one thread. | Self-play used 1.1–1.25 of 4 cores regardless of parallelism (590 → 780 games/h from 1 to 16 parallel games). | A bigger machine would not have trained faster. |

## Rules and data

| # | Defect | Effect |
|---|--------|--------|
| 6 | Legal-move generation restored a full board copy and rescanned every opponent group per candidate (≈O(n³)); each playout cloned the full undo history. | ~0.5 ms of Go-side work per network evaluation, comparable to the network itself. |
| 7 | Superko and Tromp-Taylor rules kept a global `seen` set inside the ruleset object, which search playouts wrote into. | Positions visited during search became "illegal" for the real game. |
| 8 | Simple ko was set after *any* single-stone capture, even by a stone connected to a group. | Legal moves were forbidden. |
| 9 | Self-play and arena games were cut off at 83 moves (board area + 2) and scored unfinished with area scoring and no dead-stone removal. | Noisy or wrong value and ownership targets. |
| 10 | 30% of self-play games used a hand-written heuristic evaluator. | Training data mixed in a much weaker policy. |
| 11 | No pass features: the network could not tell that the opponent had just passed. | Pass decisions were blind. |
| 12 | No symmetry augmentation, a constant learning rate, JSONL float features parsed into Python objects. | 8× less effective data; slow loading. |
| 13 | Arena gating (200 games at 400 playouts) cost about 5× the self-play that produced the cycle's data. | Most of each cycle's compute was spent on evaluation. |

## Fixed in v3 along the way

- ONNX export traced BatchNorm in training mode when the module wrapper was
  not put in `eval()`; the exported network did not match PyTorch (max error
  0.86) and the export silently switched the model back to training mode. v3
  verifies every export against PyTorch.
- Random networks ended self-play games by passing: Black passed at some
  point, White passed back and won on komi (Black won 0% of early games). v3
  only searches pass when the opponent just passed or when every other move
  fills an own eye, so games are played to a settled, exactly scored end.

## Regression tests

Each search defect has a test in `internal/mcts/search_test.go`
(`TestFindsCapture`, `TestPassesToWinAfterOpponentPass`,
`TestDoesNotPassWhenLosing`, `TestVisitAccountingAndReuse`,
`TestConservativePass`). The board is differential-tested against a naive
flood-fill reference on random games from 5×5 to 19×19, with and without
superko (`internal/game/board_test.go`).
