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
	return len(eng.arena.nodes)
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

func TestTTStoresAfterSearch(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 20
	e := NewEngine(r, nil, cfg)
	b := NewBoard(5, 6.5)
	_ = e.BestMove(b)
	if _, ok := e.TT.Get(b.Hash()); !ok {
		t.Fatal("expected TT entry after search")
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
