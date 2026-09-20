# ADR 0008: Search correctness — node storage, value sign, exploration, terminal scoring

## Status

Accepted Sep 2026. Engine-level fixes. They invalidate every strength number the project had recorded, including its reference baseline and all self-play data produced before them.

## Context

Building the v4 promotion gate ([ADR 0007](0007-pipeline-orchestrator.md)) produced an instrument that could not measure anything: every candidate scored exactly 0.500 against its champion, because every net-vs-net game was won by White in 12–32 moves and the arena alternates colours. Four defects were behind it. Each was found by asking why an experiment with a predictable answer gave the wrong one.

### 1. The search never left the root (9x9 only)

`Arena.Get` returns `&a.nodes[i]`, a pointer into a slice, and `AddChild` appends to that slice. `expandLocked` took its node pointer *before* adding children and wrote `n.Expanded = true` *after*. Once the children outgrew the slice's capacity, `append` reallocated and that write landed in the discarded array.

`NewArena` preallocates 64 nodes; a 9x9 root has 82 children. So on every 9x9 search the root stayed "unexpanded": the descent loop broke immediately, the transposition table then answered every later playout with the root's own value, and **no child was ever visited**. `bestRootMove` picks the most-visited child, all of them zero, so it returned the first legal move no matter how long the engine searched. Boards with fewer than 64 children — every unit test — were unaffected.

### 2. Selection preferred the opponent's best move

`puctScore` used `q := c.Mean()` without negating it. `backupLocked` flips the value sign at every level and evaluators return the value from the side to move, so a child's mean is the *opponent's* view. Maximising it maximises the opponent's outcome, and more search made the engine weaker.

### 3. First-play urgency was a flat constant

Unvisited children were scored at a fixed `-0.2`, far below a typical node value. Once a child had been visited, nothing could outscore it, so all playouts went down one line: at 200 playouts a single move held 100% of the visits. The visit distribution *is* the policy training target, so this would have produced one-hot policy targets.

### 4. Forced playouts flooded the root

Wu forces extra playouts on "each child of the root that has received any playouts", so a move the search never looked at gets none. Gofer forced a floor of `k` on *every* child, before the main search rather than after it. On 9x9 that is 82 children, so at a 200-playout budget the forced pass outnumbered the search itself. It flattened the visit distribution, which is exactly the policy training target, and it made the policy-target pruning (drop children below two visits) a no-op, because everything had been forced above the threshold.

### 5. `-arena-enhanced baseline` silently meant "both"

Forced-playout mode was resolved by comparing evaluator *names*: the baseline side was whichever engine's name matched `BlackEval`. The project's own reproducible baseline runs `-black-eval heuristic -white-eval heuristic` — the same name on both sides — so both engines were enhanced. Since forced playouts act at the root, this masked the root defects above in exactly the measurement meant to detect strength changes: re-running that command after fixing the search moved its win counts not at all, while self-play game length went from 12–20 moves to 68.

### 6. Measurement runs stopped early on the statistic they measured

`RunMatch` ends a match as soon as its promotion gate is decided. The accept branch is guarded by `minGamesBeforePromote`, which tests raise to disable it, but the reject branch is not. A 200-game bias measurement therefore silently played 169 games, stopping precisely when the challenger was behind — optional stopping correlated with the quantity under test. The same rule was live inside every batch of the v4 gate, stacking a second stopping rule on the stream the SPRT was already testing sequentially.

### 7. The search could not see the end of the game

Two consecutive passes end a game in every game loop, but `isTerminal` only recognised "no legal move except pass", and `leafValue` scored finished positions with the evaluator. Passing into a pass therefore looked ordinary, and with komi the engine walked into lost endings.

## Decision

- Re-fetch the node after appending children, so the "expanded" flag is written to the live array.
- Force playouts only on root children that already have visits, after the main search rather than before it.
- Resolve `-arena-enhanced baseline` by the baseline *role* across the colour swap, not by evaluator name.
- Add `MatchConfig.PlayAllGames` (`-arena-play-all`) to disable the in-match stop. Measurement runs and the v4 gate both use it: the gate needs the fixed batch size its error control assumes.
- Score children from the parent's side: `q := -c.Mean()`.
- First-play urgency is the node's own value minus `cfg.FPU * sqrt(explored prior)`, with no reduction at a noised root, following KataGo.
- `gameOver` treats two consecutive passes as the end of play and `terminalValue` scores such a position exactly with `Ruleset.Score`. Terminal results are never served from or stored in the transposition table, whose key covers stones but not pass history. The expensive "no legal move" test runs once per node at expansion; the hot path keeps the O(1) pass test.
- Arena and self-play seeds are derived by mixing (splitmix64) rather than linearly from the game index. Roles alternate on game parity, and consecutive seeds draw correlated openings, so the old scheme biased role attribution — the quantity the gate reads.

## Consequences

The project's reproducible baseline (`make reproduce-9x9-baseline`: 200 games, identical heuristic evaluators, Black searching 600 playouts against White's 200, komi 6.5) was re-run as each group of defects came off:

| Engine state | Black | White | win rate | Elo | median game |
|---|---|---|---|---|---|
| original, as committed | 7 | 193 | 0.035 | -576 | 11 moves |
| after defects 1-2 | 133 | 67 | 0.665 | +119 | ~20 moves |
| after defects 3-5 | 133 | 67 | 0.665 | +119 | 21 moves |
| after defects 6-8 | 149 | 51 | 0.745 | +186 | 76 moves |
| after the transposition-key fix | 136 | 64 | 0.680 | +131 | 55 moves |

The last row is the same command after the transposition table was made to compare keys
(Sep 2026). The drop from +186 is **not** distinguishable from sampling noise: a second
seed on the fixed engine gives 142/200, the two pool to 278/400 = +143 Elo with a 95%
interval of [+106, +180], and a two-proportion z-test on the before/after difference,
using the pooled variance under H0, gives z = 1.27, p = 0.20. What the two
post-fix runs do agree on is the median game length, 55 moves in both against 76 before.

More directly: +186 is [+131, +241] and +131 is [+80, +182]. Those overlap over almost
their entire length, so there was never a 55-Elo drop to account for. A single 200-game
arena carries a 95% interval about **105 Elo wide**; 400 games narrows it to 74. Any Elo
figure from this project should be quoted with its interval.

The middle two rows are identical in aggregate although every game differs: `-arena-enhanced` was silently enhancing both sides, so the measurement was not sensitive to the root defects that had just been fixed. The longest game in either middle run (33 moves) is shorter than the median game of the final run (76). A controlled 40-game pair at identical settings gives 0/40 before and 24/40 after.

Search now converts into strength, which is the premise the entire training loop rests on. Two further checks: with identical evaluators and fair komi, 200 games split 99–101 by colour; and a 200-playout search now spreads its visits over many moves instead of one.

- The net-size ablation ([ADR 0005](0005-net-size-ablation.md)) was run on a frozen
  pre-fix replay snapshot, so its val-loss comparison is void too. Its conclusion was to
  change nothing, so nothing followed from it, but the file now says so.
- **Every strength number recorded before these fixes is void**, including `.tectonix/reports/arena-9x9-baseline.json`, and every self-play shard produced before them carries policy targets from a search that did not search.
- Policy targets are sharper at the same settings: over 154 full-search positions at 200 playouts, the top move went from 0.159 to 0.199 of the visits, the number of moves with any visits from 41.5 to 24.3, and the entropy from 2.98 to 2.60 nats (uniform over 82 moves is 4.41).
- Forced-playout mode measurably changes play: at 600 vs 200 playouts over 60 games, Black wins 67% with `none` (median game 76 moves), 62% with `baseline` (41 moves) and 70% with `both` (21 moves). The reproducible baseline command now uses `none`, so it measures search scaling rather than forced-playout behaviour.
- Fair komi for equal heuristic engines at 50 playouts moved to about **0.5** (58% for Black at komi 0, 51% at 0.5, 46% at 1.0, 29% at 6.5). The tests that encoded the old value, itself fitted to the broken search, were updated.
- CI could not have caught any of this: `.github/workflows/ci.yml` runs `go test ./... -short`, and every arena-scale test skips in short mode, while the unit tests use boards small enough to avoid the reallocation entirely. Three short-mode tests were added: root children are visited, visits spread across moves, and selection prefers the move that is worse for the opponent.
