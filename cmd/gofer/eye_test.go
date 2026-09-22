package main

import (
	"slices"
	"testing"
)

// boardFromRows builds a Black-to-move board: X Black, O White, anything else empty.
func boardFromRows(t *testing.T, rows []string) *Board {
	t.Helper()
	b := NewBoard(len(rows), 0)
	for y, row := range rows {
		for x, c := range row {
			switch c {
			case 'X':
				setStone(t, b, x, y, Black)
			case 'O':
				setStone(t, b, x, y, White)
			}
		}
	}
	return b
}

// The search skips only a point it can never want: its own true eye. Everything
// with an opponent stone beside it, and every false eye, stays selectable.
func TestSearchSkipsOnlyObviousOwnEyes(t *testing.T) {
	for _, tc := range []struct {
		name     string
		rows     []string
		x, y     int
		filtered bool
	}{
		{"corner eye", []string{".X...", "XX...", ".....", ".....", "....."}, 0, 0, true},
		{"edge eye", []string{".X.X.", ".XXX.", ".....", ".....", "....."}, 2, 0, true},
		{"centre eye", []string{".....", ".XXX.", ".X.X.", ".XXX.", "....."}, 2, 2, true},
		{"false eye, two opponent diagonals", []string{".....", ".OXO.", ".X.X.", ".OXO.", "....."}, 2, 2, false},
		{"false eye on the edge", []string{".X.X.", ".OX..", ".....", ".....", "....."}, 2, 0, false},
		{"capture in the opponent's eye", []string{".OX..", "OOX..", "XXX..", ".....", "....."}, 0, 0, false},
		{"snapback throw-in", []string{"..OX.", "OOOX.", "XXX..", ".....", "....."}, 1, 0, false},
		{"snapback recapture", []string{"O.OX.", "OOOX.", "XXX..", ".....", "....."}, 1, 0, false},
		{"ko capture", []string{".....", ".XO..", "XO.O.", ".XO..", "....."}, 2, 2, false},
		{"liberty shared with the opponent", []string{".....", ".XO..", "X.O..", ".XO..", "....."}, 1, 2, false},
	} {
		b := boardFromRows(t, tc.rows)
		m := StoneMove(At(tc.x, tc.y))
		if !slices.Contains(Chinese().LegalMoves(b), m) {
			t.Fatalf("%s: the fixture's point is not legal", tc.name)
		}
		moves := searchMoves(Chinese(), b)
		if slices.Contains(moves, m) == tc.filtered {
			t.Errorf("%s: filtered = %v, want %v", tc.name, !slices.Contains(moves, m), tc.filtered)
		}
		if !slices.Contains(moves, PassMove) {
			t.Errorf("%s: pass was filtered", tc.name)
		}
	}
}

// The heuristic's prior rates a point ringed by its own stones highest of all, so
// before the filter its search filled its own eyes and never ran out of moves.
func TestSearchNeverFillsItsOwnEye(t *testing.T) {
	b := boardFromRows(t, []string{".X...", "XX...", ".....", ".....", "....."})
	cfg := DefaultConfig()
	cfg.Playouts = 50
	cfg.Workers = 1
	eng := NewEngine(Chinese(), Heuristic{}, cfg)
	eng.BestMove(b)
	for _, cidx := range eng.arena.Get(eng.root).Children {
		if m := eng.arena.Get(cidx).Move; !m.Pass && m.Point == At(0, 0) {
			t.Fatal("the search considered filling its own corner eye")
		}
	}
}
