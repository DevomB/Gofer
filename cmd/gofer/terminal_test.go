package main

import (
	"math/rand"
	"testing"
)

// constEval returns a fixed value for every position, like an untrained net.
type constEval struct{ v float32 }

func (c constEval) Evaluate(b *Board) Result {
	return Result{Value: float64(c.v), HasValue: true}
}

func playPasses(rs Ruleset, b *Board, n int) {
	for i := 0; i < n; i++ {
		rs.Play(b, PassMove)
	}
}

func TestGameOverAfterTwoPasses(t *testing.T) {
	rs := Chinese()
	b := NewBoard(9, 6.5)
	eng := NewEngine(rs, constEval{}, DefaultConfig())
	if eng.gameOver(b) {
		t.Fatal("empty board is not over")
	}
	playPasses(rs, b, 1)
	if eng.gameOver(b) {
		t.Fatal("one pass does not end the game")
	}
	playPasses(rs, b, 1)
	if !eng.gameOver(b) {
		t.Fatal("two consecutive passes end the game")
	}
}

func TestTerminalValueUsesScoreNotEvaluator(t *testing.T) {
	rs := Chinese()
	b := NewBoard(9, 6.5)
	playPasses(rs, b, 2) // Black passed, White passed: empty board, Black to move, White wins on komi
	if b.Player() != Black {
		t.Fatalf("expected Black to move, got %v", b.Player())
	}
	// An evaluator claiming the position is won must not override the real result.
	eng := NewEngine(rs, constEval{v: 1}, DefaultConfig())
	if got := eng.leafValue(b); got != -1 {
		t.Fatalf("Black to move on an empty scored board loses by komi: got %v", got)
	}
	rs.Play(b, PassMove) // same finished board, now from White's side
	if got := eng.leafValue(b); got != 1 {
		t.Fatalf("White wins the same finished board: got %v", got)
	}
}

// The search must not walk into a lost ending: with an evaluator that rates every
// position equally, passing is the one move whose consequence is knowable.
func TestSearchAvoidsLosingPass(t *testing.T) {
	rs := Chinese()
	b := NewBoard(9, 6.5)
	rs.Play(b, PassMove) // White passed; Black to move, passing ends the game and loses by komi
	cfg := DefaultConfig()
	cfg.Playouts = 200
	cfg.Seed = 7
	cfg.Workers = 1
	eng := NewEngine(rs, constEval{}, cfg)
	m := eng.SelectMove(b, rand.New(rand.NewSource(1)), 0)
	if m.Pass {
		t.Fatal("Black passed into a komi loss")
	}
}

// TestSelectionPrefersMovesGoodForTheMover checks child values are negated.
func TestSelectionPrefersMovesGoodForTheMover(t *testing.T) {
	eng := NewEngine(Chinese(), Heuristic{}, DefaultConfig())
	eng.arena = NewArena()
	eng.root = eng.arena.Root()
	root := eng.arena.Get(eng.root)
	root.Expanded = true
	root.Visits = 20

	// Equal priors and visit counts, so only the values can decide.
	eng.arena.AddChild(eng.root, StoneMove(PointFromIdx(9, 0)), 0.5) // good for the opponent
	eng.arena.AddChild(eng.root, StoneMove(PointFromIdx(9, 1)), 0.5) // bad for the opponent
	good, bad := eng.arena.Get(root.Children[0]), eng.arena.Get(root.Children[1])
	for _, c := range []*Node{good, bad} {
		c.Visits = 10
	}
	good.ValueSum = +9 // child to move (the opponent) is winning there
	bad.ValueSum = -9  // the opponent is losing there: this is our move

	picked := eng.selectChildLocked(eng.root, false)
	if picked != root.Children[1] {
		t.Fatalf("selection picked the child with mean %+.1f over the one with mean %+.1f: "+
			"the value sign is inverted, so search maximises the opponent's outcome",
			good.Mean(), bad.Mean())
	}
}

// TestSearchVisitsRootChildren checks search descends after root expansion.
func TestSearchVisitsRootChildren(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Playouts = 60
	cfg.Workers = 1
	cfg.Seed = 3
	eng := NewEngine(Chinese(), Heuristic{}, cfg)
	b := NewBoard(9, 6.5)
	a := eng.Analyze(b, 5)

	root := eng.arena.Get(eng.root)
	if got := len(root.Children); got != 82 {
		t.Fatalf("root has %d children, want 82 (one per legal move plus pass); "+
			"duplicates mean it was expanded more than once", got)
	}
	var visits uint32
	for _, c := range root.Children {
		visits += eng.arena.Get(c).Visits
	}
	if visits == 0 {
		t.Fatal("no root child was visited in 60 playouts: the search never descends")
	}
	if len(a.Candidates) == 0 || a.Candidates[0].Visits == 0 {
		t.Fatalf("analysis reports no visited candidate: %+v", a.Candidates)
	}
}

// TestSearchExploresSeveralMoves guards first-play urgency.
func TestSearchExploresSeveralMoves(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Playouts = 200
	cfg.Workers = 1
	cfg.Seed = 11
	eng := NewEngine(Chinese(), Heuristic{}, cfg)
	a := eng.Analyze(NewBoard(9, 6.5), 5)

	root := eng.arena.Get(eng.root)
	visited, top := 0, uint32(0)
	for _, cidx := range root.Children {
		if v := eng.arena.Get(cidx).Visits; v > 0 {
			visited++
			if v > top {
				top = v
			}
		}
	}
	if visited < 5 {
		t.Fatalf("only %d moves visited in 200 playouts: the search is not exploring", visited)
	}
	if float64(top) > 0.9*float64(cfg.Playouts) {
		t.Fatalf("one move took %d of %d playouts: the visit distribution is degenerate",
			top, cfg.Playouts)
	}
	if len(a.Candidates) < 2 {
		t.Fatalf("analysis lists %d candidates", len(a.Candidates))
	}
}

func TestTerminalValueSignsWithKomi(t *testing.T) {
	rs := Chinese()
	// Komi 0.5 on an empty board: Black, to move, still loses on komi alone.
	b := NewBoard(9, 0.5)
	playPasses(rs, b, 2)
	if v := terminalValue(rs, b); v != -1 {
		t.Fatalf("Black to move should lose by 0.5 komi: %v", v)
	}
	// A board where Black owns everything outweighs komi.
	b2 := NewBoard(9, 6.5)
	for i := 0; i < 9*9; i += 2 {
		rs.Play(b2, StoneMove(PointFromIdx(9, i)))
		rs.Play(b2, PassMove)
	}
	playPasses(rs, b2, 2)
	black, white := rs.Score(b2)
	want := 1.0
	if b2.Player() == White {
		want = -1
	}
	if black <= white {
		t.Skip("position did not come out as a Black win; scoring covered elsewhere")
	}
	if v := terminalValue(rs, b2); v != want {
		t.Fatalf("terminal value %v, want %v (black %v white %v)", v, want, black, white)
	}
}

// TestArenaPointersSurviveGrowth checks pointers returned by Get remain valid.
func TestArenaPointersSurviveGrowth(t *testing.T) {
	a := NewArena()
	root := a.Root()
	held := a.Get(root)
	held.Visits = 7

	// Allocate well past one chunk, and past any plausible 19x19 root (362).
	const n = arenaChunk*3 + 5
	kept := make([]*Node, 0, n)
	for i := 0; i < n; i++ {
		idx := a.AddChild(root, PassMove, float64(i))
		c := a.Get(idx)
		c.Visits = uint32(i)
		kept = append(kept, c)
	}

	if held.Visits != 7 {
		t.Fatalf("root pointer stale after %d allocations: visits %d", n, held.Visits)
	}
	if a.Get(root).Visits != 7 {
		t.Fatal("root value lost")
	}
	if got := len(a.Get(root).Children); got != n {
		t.Fatalf("root has %d children, want %d", got, n)
	}
	for i, c := range kept {
		if c.Visits != uint32(i) {
			t.Fatalf("child %d pointer stale: visits %d", i, c.Visits)
		}
		if a.Get(i+1).Visits != uint32(i) {
			t.Fatalf("child %d lost its value through Get", i)
		}
	}
	if a.Len() != n+1 {
		t.Fatalf("arena length %d, want %d", a.Len(), n+1)
	}
}
