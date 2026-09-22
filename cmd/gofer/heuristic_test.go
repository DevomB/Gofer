package main

import (
	"strings"
	"testing"
)

// An empty board is lost for Black by exactly komi under area scoring, and the
// heuristic has to agree with the rules about it or the search compares leaf
// estimates and scored endings on different scales.
func TestHeuristicCountsKomi(t *testing.T) {
	b := NewBoard(9, 6.5)
	want := -6.5 / 81
	if got := (Heuristic{}).Evaluate(b).Value; got != want {
		t.Fatalf("Black to move on an empty board: value %v, want %v", got, want)
	}
	Chinese().Play(b, PassMove)
	if got := (Heuristic{}).Evaluate(b).Value; got != -want {
		t.Fatalf("White to move on an empty board: value %v, want %v", got, -want)
	}
}

// setupBoard plays GTP vertices from an empty 9x9 board with komi 6.5.
func setupBoard(t *testing.T, moves string) *Board {
	t.Helper()
	b := NewBoard(9, 6.5)
	if err := applySetupMoves(Chinese(), b, strings.Split(moves, ",")); err != nil {
		t.Fatal(err)
	}
	return b
}

func topHeuristicPrior(b *Board) int {
	p := heuristicPolicy(b)
	best := 0
	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}
	return best
}

// After these moves Black's E5 has one liberty, E4. White's prior used to score
// E4 at its floor, because it touches no white stone and one black one, and
// first-play urgency kept a 4000-playout search from ever trying it.
func TestHeuristicCapturesAStoneInAtari(t *testing.T) {
	b := setupBoard(t, "C3,D5,E5,F5,C7,E6,G3")
	e4 := At(4, 5).Idx(9)
	if got := topHeuristicPrior(b); got != e4 {
		t.Fatalf("White's highest prior is %v, want the capture at E4", PointFromIdx(9, got))
	}
	cfg := DefaultConfig()
	cfg.Playouts = 400
	cfg.Workers = 1
	if m := NewEngine(Chinese(), Heuristic{}, cfg).BestMove(b); m.Pass || m.Point.Idx(9) != e4 {
		t.Fatalf("search played %+v, want the capture at E4", m)
	}
}

// One ply earlier Black can save E5 by extending to E4, from one liberty to three.
func TestHeuristicSavesAStoneInAtari(t *testing.T) {
	b := setupBoard(t, "C3,D5,E5,F5,C7,E6")
	if got, e4 := topHeuristicPrior(b), At(4, 5).Idx(9); got != e4 {
		t.Fatalf("Black's highest prior is %v, want the extension to E4", PointFromIdx(9, got))
	}
}
