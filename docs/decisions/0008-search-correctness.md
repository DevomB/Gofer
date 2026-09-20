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

### 4. The search could not see the end of the game

Two consecutive passes end a game in every game loop, but `isTerminal` only recognised "no legal move except pass", and `leafValue` scored finished positions with the evaluator. Passing into a pass therefore looked ordinary, and with komi the engine walked into lost endings.

## Decision

- Re-fetch the node after appending children, so the "expanded" flag is written to the live array.
- Score children from the parent's side: `q := -c.Mean()`.
- First-play urgency is the node's own value minus `cfg.FPU * sqrt(explored prior)`, with no reduction at a noised root, following KataGo.
- `gameOver` treats two consecutive passes as the end of play and `terminalValue` scores such a position exactly with `Ruleset.Score`. Terminal results are never served from or stored in the transposition table, whose key covers stones but not pass history. The expensive "no legal move" test runs once per node at expansion; the hot path keeps the O(1) pass test.
- Arena and self-play seeds are derived by mixing (splitmix64) rather than linearly from the game index. Roles alternate on game parity, and consecutive seeds draw correlated openings, so the old scheme biased role attribution — the quantity the gate reads.

## Consequences

Controlled experiment, identical settings before and after (40 games, seed 42, identical heuristic evaluators, symmetric configuration, Black searching 600 playouts against White's 200, komi 6.5):

| | Black wins | median game length |
|---|---|---|
| before (commit `6ef7ce3`) | 0/40 | 28 moves |
| after | 24/40 (60%) | 80 moves |

Search now converts into strength, which is the premise the entire training loop rests on. Two further checks: with identical evaluators and fair komi, 200 games split 99–101 by colour; and a 200-playout search now spreads its visits over many moves instead of one.

- **Every strength number recorded before these fixes is void**, including `.tectonix/reports/arena-9x9-baseline.json`, and every self-play shard produced before them carries policy targets from a search that did not search.
- Fair komi for equal heuristic engines at 50 playouts moved to about **0.5** (58% for Black at komi 0, 51% at 0.5, 46% at 1.0, 29% at 6.5). The tests that encoded the old value, itself fitted to the broken search, were updated.
- CI could not have caught any of this: `.github/workflows/ci.yml` runs `go test ./... -short`, and every arena-scale test skips in short mode, while the unit tests use boards small enough to avoid the reallocation entirely. Three short-mode tests were added: root children are visited, visits spread across moves, and selection prefers the move that is worse for the opponent.
