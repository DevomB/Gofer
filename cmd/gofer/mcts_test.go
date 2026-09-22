package main

import (
	"math"
	"testing"
	"time"
)

func treeNodes(eng *Engine) int {
	eng.mu.Lock()
	defer eng.mu.Unlock()
	if eng.arena == nil {
		return 0
	}
	return eng.arena.Len()
}

func TestPUCTFormula(t *testing.T) {
	q, prior, pv := 0.5, 0.1, 100.0
	visits, c := uint32(10), 1.1
	got := q + c*prior*math.Sqrt(pv)/(1+float64(visits))
	want := 0.5 + 1.1*0.1*10/11
	if got < want-1e-9 || got > want+1e-9 {
		t.Fatalf("puct got %v want %v", got, want)
	}
}

func TestDeterministicPlayout(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Seed = 42
	cfg.Playouts = 6
	b := NewBoard(5, 6.5)
	m1 := NewEngine(r, Uniform{}, cfg).BestMove(b)
	m2 := NewEngine(r, Uniform{}, cfg).BestMove(b)
	if m1 != m2 {
		t.Fatalf("deterministic seed mismatch %v vs %v", m1, m2)
	}
}

func TestRootPolicySumsToOne(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 15
	e := NewEngine(r, nil, cfg)
	b := NewBoard(5, 6.5)
	legal := r.LegalMoves(b)
	_ = e.BestMove(b)
	pi := e.RootPolicy(legal)
	var sum float32
	for _, p := range pi {
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("policy sum %v", sum)
	}
}

func TestMCTSTreeReuse(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	cfg := DefaultConfig()
	cfg.Playouts = 30
	cfg.Workers = 1
	eng := NewEngine(r, Uniform{}, cfg)
	m1 := eng.BestMove(b)
	len1 := treeNodes(eng)
	if len1 == 0 {
		t.Fatal("expected nodes after search")
	}
	eng.BestMove(b)
	if treeNodes(eng) < len1 {
		t.Fatalf("tree shrank on reuse: %d -> %d", len1, treeNodes(eng))
	}
	r.Play(b, m1)
	eng.AdvanceTree(m1)
	if treeNodes(eng) == 0 {
		t.Fatal("tree cleared after advancing to best child")
	}
	eng.ResetArena()
	if treeNodes(eng) != 0 {
		t.Fatal("reset failed")
	}
}

func TestAnalyzeCandidates(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 30
	e := NewEngine(r, Heuristic{}, cfg)
	b := NewBoard(5, 6.5)
	a := e.Analyze(b, 3)
	if a.Playouts <= 0 || len(a.Candidates) == 0 {
		t.Fatalf("analyze: playouts=%d cands=%d", a.Playouts, len(a.Candidates))
	}
}

func TestThinkTimeSearch(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.ThinkTime = 50 * time.Millisecond
	e := NewEngine(r, Heuristic{}, cfg)
	b := NewBoard(5, 6.5)
	a := e.Analyze(b, 3)
	if a.Playouts <= 0 {
		t.Fatal("expected playouts during think window")
	}
}

// Wu 2020 §3.2 forces visits inside the search budget, and only on root children
// the search has already tried.
func TestForcedRootPlayouts(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 20
	cfg.ForcedRootPlayouts = 2
	cfg.Workers = 1
	e := NewEngine(r, Uniform{}, cfg)
	b := NewBoard(5, 6.5)
	_ = e.BestMove(b)
	e.mu.Lock()
	defer e.mu.Unlock()
	root := e.arena.Get(e.root)
	var total uint32
	untouched := 0
	for _, cidx := range root.Children {
		v := e.arena.Get(cidx).Visits
		total += v
		if v == 0 {
			untouched++
		}
	}
	if total != uint32(cfg.Playouts) {
		t.Fatalf("root children hold %d visits after a %d-playout search: forcing ran outside the budget",
			total, cfg.Playouts)
	}
	if untouched == 0 {
		t.Fatalf("all %d children were visited: the floor is being applied to untried moves",
			len(root.Children))
	}
}

// rootWithChildren builds an expanded root whose children have the given priors,
// visits and means (each mean from the child's side to move).
func rootWithChildren(e *Engine, children []struct {
	prior  float64
	visits uint32
	mean   float64
}) *Node {
	e.arena = NewArena()
	e.root = e.arena.Root()
	root := e.arena.Get(e.root)
	root.Expanded = true
	for i, ch := range children {
		idx := e.arena.AddChild(e.root, StoneMove(PointFromIdx(9, i)), ch.prior)
		c := e.arena.Get(idx)
		c.Visits = ch.visits
		c.ValueSum = ch.mean * float64(ch.visits)
		root.Visits += ch.visits
	}
	return root
}

// A tried root child below sqrt(k*P*N) is taken before PUCT is consulted, however
// bad its value, and an untried one never is.
func TestForcedChildIsSelectedFirst(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ForcedRootPlayouts = 2
	e := NewEngine(Chinese(), Heuristic{}, cfg)
	root := rootWithChildren(e, []struct {
		prior  float64
		visits uint32
		mean   float64
	}{
		{0.2, 10, -0.2}, // the best move so far
		{0.5, 1, 0.9},   // tried once and bad; floor sqrt(2*0.5*11) = 3.3
		{0.3, 0, 0},     // untried: PUCT's pick
	})
	if got := e.selectChildLocked(e.root, true); got != root.Children[1] {
		t.Fatalf("selected child %d, want the under-forced child %d", got, root.Children[1])
	}
}

// Pruning removes the visits forcing added, not visits a move earned.
func TestPruningRemovesOnlyForcedVisits(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ForcedRootPlayouts = 2
	e := NewEngine(Chinese(), Heuristic{}, cfg)
	root := rootWithChildren(e, []struct {
		prior  float64
		visits uint32
		mean   float64
	}{
		{0.5, 100, -0.5}, // best: parent value +0.5
		{0.05, 3, 0.3},   // parent value -0.3: its 3 visits are all forcing (floor 3.4)
		{0.3, 10, -0.45}, // parent value +0.45: close enough to the best to have earned them
	})
	legal := make([]Move, len(root.Children))
	for i, cidx := range root.Children {
		legal[i] = e.arena.Get(cidx).Move
	}
	pi := e.RootPolicyPruned(legal)
	want := []float32{100.0 / 110, 0, 10.0 / 110}
	for i := range want {
		if math.Abs(float64(pi[i]-want[i])) > 1e-6 {
			t.Fatalf("pruned policy %v, want %v", pi, want)
		}
	}
}

func TestRootPolicyPruned(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 25
	cfg.ForcedRootPlayouts = 2
	e := NewEngine(r, Uniform{}, cfg)
	b := NewBoard(5, 6.5)
	legal := r.LegalMoves(b)
	_ = e.BestMove(b)
	pi := e.RootPolicyPruned(legal)
	var sum float32
	for _, p := range pi {
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("pruned policy sum %v", sum)
	}
}

func TestOwnershipLabel(t *testing.T) {
	b := NewBoard(5, 6.5)
	r := Chinese()
	_ = r.Play(b, StoneMove(At(2, 2)))
	own := OwnershipLabel(b)
	if len(own) != 25 {
		t.Fatalf("len %d", len(own))
	}
	if own[2*5+2] != 1 {
		t.Fatalf("stone ownership %v", own[2*5+2])
	}
}
