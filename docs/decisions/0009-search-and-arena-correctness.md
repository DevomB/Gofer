# ADR 0009: The value cache, the arena's trees, root priors, scoring, and the teacher

## Status

Accepted Sep 2026. Like [ADR 0008](0008-search-correctness.md) this invalidates what came
before it: every arena result the project has recorded, including ADR 0008's own baseline
table, every gate and anchor, and every network trained so far.

## Context

Networks trained on their own games got weaker (+53, −35, −168 Elo against the heuristic
over three arms of one run), and their self-play collapsed toward passing (opening pass
share 0.007 on heuristic shards, 0.087 on net shards; `c0889a8`). `8c5a6d7` traced the
collapse to the value function. Following it further found the defects below. Each was
reproduced on the unchanged code before it was fixed.

### 1. A pass child read its parent's value from the parent's side

The transposition table was keyed on the stones alone. A pass leaves the stones unchanged,
so the pass child found its parent's entry, stored from the parent's side to move, and
returned it before it was ever expanded. The parent read passing as exactly the negation of
its own evaluation, and never looked past the pass to the opponent's winning pass-back.

It acted at every node, not only the root: after a good move, the opponent's pass read as a
win for the opponent, so good moves were marked down and bad ones up wherever pass had any
prior. The heuristic gives pass about 0.5% and rarely triggered it; a network with 1–4%
triggered it everywhere. With a 9% pass prior, the level seen in collapsed shards, pass
took 357 of 400 visits on an empty board in a position the evaluator scores equally.

The same saved networks, empty 9×9 board, 400 playouts, before and after:

| network | top move before | top move after |
|---|---|---|
| seedfix gen-0001 | C2, 12.5% | E5, 70% |
| seedfix gen-0002 | D6, 17% | E5, 76% |
| seedfix candidate-0005 | pass, 29% | E5, 74% |
| cpu candidate-0001 | C4, 20% | E5, 66% |

On the collapsed candidate-0005, pass took 62% / 49% / 12% / 3.5% of the visits at
50 / 200 / 800 / 3200 playouts before, valued −0.215 at every budget because the value was
pinned; at 800 and 3200 the old search's top move was J1, the corner point. After, pass is
valued −0.79 to −0.92 and its share falls with search, 86% / 34% / 9% / 1.9%. The 86% at 50
playouts is first-play urgency anchored to a root mean that the pass child's early visits
drag down; it needs a network with a 4.4% pass prior to show.

The table also stunted the tree: a node whose position was already in the table returned
before expanding, on every visit, and the table persisted across the moves of a game. The
search benchmark reset the tree but not the table, so after its first iteration every root
child was answered from the table; its 0.86 ms per 200 playouts measured a one-level search.

### 2. A stone bordering two regions counted for only one

`floodEmpty` marked stones as seen in the slice shared by the whole board scan. A stone
bordering two empty regions was credited to the first region scanned; a later region
touching that colour only through such stones was scored as the other colour's territory.
It had been that way since the first version. Over 100 heuristic self-play games at the
cycle-1 settings the old scorer disagreed with the fixed one on 12 final boards and flipped
the winner on 1. `Score` feeds `terminalValue` in the search and every value, score and
ownership label; the heuristic's territory estimate used the same function.

### 3. Root noise reached only the first move of a game

Noise was mixed into the priors when the root was expanded. Self-play reuses the tree, and
the move just played is always expanded, so every later root arrived already expanded. In a
60-move game noise reached 1 search. The first-play urgency code still assumed a noised
root and skipped its reduction there.

### 4. Root temperature divided Q instead of softening the priors

The 1.03 from Wu's hyperparameters is a softmax temperature on the root policy. Gofer
divided visited root children's mean values by it.

### 5. Forced playouts ran outside the budget, and pruning never removed them

They ran after the search as a separate pass, to `k + ⌊√(P·(N+1))⌋` rather than Wu's
`√(k·P·ΣN)`, and the policy target kept them: "pruning" only dropped children with fewer
than two visits. Noise was also on for fast searches, which Wu turns off because they only
choose the move played.

### 6. The heuristic ignored komi

Its value was the area margin without komi, so it disagreed with the rules about who was
winning by exactly 6.5 points, and compared leaf estimates to scored endings on different
terms.

### 7. The arena advanced only the mover's tree

`playArenaGame` called `AdvanceTree` on the engine that moved and not on its opponent, so
every engine searched from a root one ply behind the board: its children were the
opponent's candidate replies in the old position, their values stored for the other side.
When the most-visited of those was an occupied point, the rules refused it and the arena
played a pass instead, silently. In 20 heuristic games at 200 playouts, 177 of 1024 moves
(17%) were refused choices turned into passes, in all 20 games. It had been that way since
v2.0.0. `applyMove` ignored refusals too, so inside the search a refused move left the side
to move unchanged while the path descended.

### 8. The heuristic prior scored every capture at its floor

`policyPriorAt` rewarded neighbouring own stones and penalised neighbouring opponent
stones. A capturing point always touches the stones it captures, so captures sat at the
0.01 floor, and first-play urgency then kept them from ever being tried: with Black's E5 in
atari after `C3 D5 E5 F5 C7 E6 G3`, White's capture at E4 had prior 0.001 and 0 of 4000
visits. Its value gain, about 2/81, was too small to matter even if tried.

### 9. Most games were scored at the move cap

Every game loop stopped at `size²+2` moves, 83 on 9×9, and scored the board as if the game
were over. At the cycle-1 settings 86 of 100 heuristic self-play games on the old engine
reached it; the last CPU run's cycle 1 averaged 81.5 moves a game. So most value, score and
ownership labels described an arbitrary position at move 83, and the learner trains value
and ownership on every row. Across 78 recorded arena reports 262 of 6181 games (4.2%)
reached move 83, including 37 of the reproducible baseline's 200, 7 of the last run's 40
seed-gate games, and 9 of an anchor's 40.

Two things kept games from ending. A losing side sees any leaf estimate as better than a
scored loss, so it never passes; and the heuristic's prior rates a point ringed by its own
stones highest of all, so its search spent its moves filling its own eyes.

### 10. The rules refused every snapback

`chineseRules.Play` marked a ko whenever a move captured one stone. After a throw-in is
taken, the capturer usually belongs to a larger group with one liberty, and retaking there
captures that group: a snapback, not a ko. The rules refused it, so the engine could never
play one, and `-convert-sgf` rejected every recorded game containing one as illegal. The
test fixtures for ko encoded the same mistake: neither was a ko.

### 11. Two silent fallbacks

An unknown evaluator name, or an unknown `-selfplay-eval` mode, became the heuristic, so a
typo passed for a real ONNX experiment. An engine's move the rules refused became a pass in
GTP, `-play`, `-watch` and the arena, and self-play ignored the refusal entirely.

### Also

- Tromp–Taylor rules, the superko wrapper, and rules and board-size randomization were
  reachable only through `RulesRandomize`, which nothing set. Both repetition checks kept
  one history shared by every search branch, and superko rejected every pass. The
  traceability map listed both randomizations as done.
- The sidecar still required feature schema 2 after `8c5a6d7` moved Go to 3, so every
  sidecar evaluation fell back to the heuristic. Pipeline smoke has failed on it since.

## Decision

- Delete the transposition table. Expansion returns its evaluation to the backup directly.
  A value cache would need a key covering every network input, including three history
  planes, and must not stop nodes expanding; none is needed until a profile says so.
  `Uniform` states the zero value the table always gave it, and the random-rollout path,
  reachable from nothing else, is deleted.
- Score regions by marking only empty points as seen.
- Prepare the root for every search: a reused root is re-evaluated and its priors rebuilt
  with the temperature, then noise. Temperature applies to priors, not Q. Fast searches run
  without noise.
- Force playouts inside the search (a tried root child below `√(k·P·ΣN)` is selected
  first) and prune as Wu does: each child but the most visited loses up to its forced
  playouts while its PUCT stays below the best child's; children left with one visit are
  dropped.
- Include komi in the heuristic's margin.
- Advance both arena engines on every move. A move the rules refuse is a panic in the arena
  and in `applyMove`: the board and the tree disagree.
- Add the stones a move captures, or saves from atari into at least two liberties, to the
  heuristic prior, five per stone.
- Delete the unreachable rules code. Pin the sidecar schema to Go's with a test.
- Every game loop reports why it stopped: `two_passes`, `only_pass` or `move_cap`. A
  self-play game stopped at the cap contributes no rows; an arena game stopped there counts
  toward no win, loss or draw, and the arena refuses to report rather than drop it, which
  would bias the sample toward games that end. Shard metadata and arena reports record
  every ending.
- The cap is `3·size²+2`, 245 on 9×9. It only stops a game that never ends.
- The search skips an obvious own eye: every orthogonal neighbour the mover's, and fewer
  than two opponent diagonals, the edge counting as one. Filling one never captures. False
  eyes, and every point beside an opponent stone, stay selectable; the rules still allow
  all of them, so human and SGF play are unaffected.
- A capture of one stone makes a ko only when the capturer stands alone with that point as
  its one liberty.
- Unknown evaluator names and self-play modes fail before any work, listing what is
  accepted. A refused engine move is an error everywhere: self-play and the arena abort with
  the move, player, board hash and position; GTP answers with a failure; `-play` and
  `-watch` stop and report it.

## Consequences

The cap and the eye filter were chosen by a probe: heuristic self-play at the cycle-1
settings (200/50 playouts, cap randomisation 0.2, 16 sampled opening moves), 100 games per
arm on identical seeds, with every other change here in place. No arm made an illegal
selection.

| arm | own-eye filter | cap | finished | at the cap | length median / p95 / max | wall | board occupied |
|---|---|---|---|---|---|---|---|
| control | off | 83 | 1 | 99 | 83 / 83 / 83 | 219 s | 61% |
| cap only | off | 164 | 60 | 40 | 158 / 164 / 164 | 616 s | 79% |
| proposed | on | 164 | 96 | 4 | 116 / 162 / 164 | 410 s | 94% |
| diagnostic | on | 245 | 100 | 0 | 116 / 162 / 231 | 414 s | 94% |

164 still stopped 4% of games, above the 1% bar, so the cap is 245. The shipped code
reproduces the diagnostic arm exactly. Once capped games are discarded, the control arm
yields 80 training rows from 100 games; the chosen configuration yields 11,349.

`TestSelfplayGamesFinish` holds the result: 20 games at these settings, none at the cap.

Games are longer and every playout now expands a node, so the full test suite took 92
minutes on a 4-core laptop, against an 80-minute limit in the nightly job. 49 of those
minutes were `TestHeuristicFairKomiSweep`, which asserts nothing; it now runs only with
`GOFER_KOMI_SWEEP=1`. The calibration it informs still holds: at komi 0.5 identical
heuristics split 109–91 by colour over 200 games.

- Search now expands every new node. That is the cost of searching: the corrected
  benchmark is about 150 µs per playout against 4 µs for the table-served one, and a
  network arena needed 31% more evaluations (1.10M against 0.84M for the same 100 games,
  measured on the stale-tree arena). The CI benchmark baseline predates ADR 0008 and has failed on
  every push since; it needs regenerating on CI hardware after this lands.
- Every arena number recorded before this is void, including ADR 0008's table (reproduced
  exactly by the old code on the old arena: 136/200), every gate, anchor and seed gate, and
  the comparison of 39, 55 and 43 wins out of 100 taken while finding these defects.
- Every shard produced so far carries targets from these searches and labels from the old
  scorer, and every network trained on them is void as evidence about the loop.

- Two local self-play datasets, `training/data/samples.jsonl` (80 games, 26% scored at the
  cap) and the net-size ablation snapshot (3283 games, 20%), are quarantined, not deleted,
  in `training/data/quarantine/`. `samples.jsonl` was the default `--data` of the learner,
  its benchmark and the parity harness. The supervised SGF set runs to 123 moves with no
  pile-up at 83 and was not affected, but games with a snapback are missing from it.

### Open

- A third of finished self-play games end in a wipeout. 44 of the 100 ended by two passes,
  with a median margin of 7.5 points; 56 ended because the side to move had no legal move
  left, with a median margin of 65.5 and 32 of them over 60. A losing side still sees any
  leaf estimate as better than a scored loss, so it never passes: it plays on until its own
  groups are gone. Those games are legal and scored by the rules, but their ownership and
  score labels record the collapse, not the position that decided the game. Filtering moves
  inside pass-alive territory (Benson), already the listed upgrade for scoring, is the
  conservative next step; resignation would also need the learner to mask ownership and
  score on resigned games.
- A sidecar is not checked at startup: `CheckSidecarHealth` is only called from tests, and
  it does not compare `schema_version`. The fallback guard still refuses the run at the end.
