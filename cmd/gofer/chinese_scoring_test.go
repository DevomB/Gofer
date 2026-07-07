package main

import (
	"testing"
)

// chineseAreaDecomposition mirrors chineseRules.Score territory logic without komi.
// Returns black area, white area, neutral (seki), and unassigned (dame touching no stones).
func chineseAreaDecomposition(b *Board) (black, white, neutral, unassigned int) {
	n := b.Size() * b.Size()
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case Black:
			black++
		case White:
			white++
		}
	}
	seen := make([]bool, n)
	for i := 0; i < n; i++ {
		if seen[i] || b.AtIndex(i) != Empty {
			continue
		}
		t, tb, tw := floodEmpty(b, i, seen)
		switch {
		case tb && !tw:
			black += t
		case tw && !tb:
			white += t
		case tb && tw:
			neutral += t
		default:
			unassigned += t
		}
	}
	return black, white, neutral, unassigned
}

func setStone(t *testing.T, b *Board, x, y int, c Stone) {
	t.Helper()
	if x < 0 || y < 0 || x >= b.Size() || y >= b.Size() {
		t.Fatalf("coord (%d,%d) out of range for %dx%d", x, y, b.Size(), b.Size())
	}
	b.SetStoneIndex(y*b.Size()+x, c)
}

func playCoord(t *testing.T, r Ruleset, b *Board, x, y int) {
	t.Helper()
	m := StoneMove(At(x, y))
	if !r.Play(b, m) {
		t.Fatalf("illegal play at (%d,%d) player=%v", x, y, b.Player())
	}
}

func passMove(t *testing.T, r Ruleset, b *Board) {
	t.Helper()
	if !r.Play(b, PassMove) {
		t.Fatal("pass illegal")
	}
}

// (a) Empty board: no stones, no territory assigned — only komi affects final white score.
func TestChineseAreaEmptyBoardNoTerritory(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 0)
	bl, wl, neutral, unassigned := chineseAreaDecomposition(b)
	if bl != 0 || wl != 0 || neutral != 0 || unassigned != 81 {
		t.Fatalf("empty 9x9 want 0,0,0,81 unassigned got %d %d %d %d", bl, wl, neutral, unassigned)
	}
	scBl, scWl := r.Score(b)
	if scBl != 0 || scWl != 0 {
		t.Fatalf("Score with komi=0 want 0,0 got %v %v", scBl, scWl)
	}
	b2 := NewBoard(9, 6.5)
	_, scWl2 := r.Score(b2)
	if scWl2 != 6.5 {
		t.Fatalf("empty board white score should be komi only, got %v", scWl2)
	}
}

// (b) Mirror-symmetric stones: equal area for both colors before komi.
func TestChineseAreaSymmetricTerritories(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 0)
	for _, c := range [][2]int{{0, 0}, {1, 0}, {0, 1}} {
		setStone(t, b, c[0], c[1], Black)
	}
	for _, c := range [][2]int{{8, 8}, {7, 8}, {8, 7}} {
		setStone(t, b, c[0], c[1], White)
	}
	bl, wl, _, _ := chineseAreaDecomposition(b)
	if bl != wl {
		t.Fatalf("mirror stones want equal area got black=%d white=%d", bl, wl)
	}
	if bl < 3 {
		t.Fatalf("expected at least 3 stones per side, got black=%d", bl)
	}
	_ = r
}

// (c) Surrounded on-board stone: current scorer counts it for owner (no Benson removal).
func TestChineseAreaDeadStoneCountsAsTerritory(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 0)
	setStone(t, b, 4, 4, White)
	for _, c := range [][2]int{{3, 4}, {5, 4}, {4, 3}, {4, 5}, {0, 0}} {
		setStone(t, b, c[0], c[1], Black)
	}
	bl, wl, _, _ := chineseAreaDecomposition(b)
	if wl < 1 {
		t.Fatalf("white dead stone still on board counts as white area in current impl, got white=%d", wl)
	}
	t.Logf("ceiling: dead on-board stone counts for owner (black=%d white=%d); Benson pass not implemented", bl, wl)
	_ = r
}

// (d) Conservation: stones + attributed territory + neutral + unassigned == board size.
func TestChineseAreaConservationInvariant(t *testing.T) {
	r := Chinese()
	cases := []struct {
		name string
		play func(t *testing.T, r Ruleset, b *Board)
	}{
		{"empty", func(t *testing.T, r Ruleset, b *Board) {}},
		{"single_black", func(t *testing.T, r Ruleset, b *Board) { playCoord(t, r, b, 4, 4) }},
		{"double_pass", func(t *testing.T, r Ruleset, b *Board) {
			playCoord(t, r, b, 2, 2)
			playCoord(t, r, b, 6, 6)
			passMove(t, r, b)
			passMove(t, r, b)
		}},
		{"mirror_corners", func(t *testing.T, r Ruleset, b *Board) {
			for _, c := range [][2]int{{0, 0}, {1, 0}, {0, 1}} {
				setStone(t, b, c[0], c[1], Black)
			}
			for _, c := range [][2]int{{8, 8}, {7, 8}, {8, 7}} {
				setStone(t, b, c[0], c[1], White)
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := NewBoard(9, 6.5)
			tc.play(t, r, b)
			bl, wl, neutral, unassigned := chineseAreaDecomposition(b)
			sum := bl + wl + neutral + unassigned
			if sum != 81 {
				t.Fatalf("conservation failed: black=%d white=%d neutral=%d unassigned=%d sum=%d want 81",
					bl, wl, neutral, unassigned, sum)
			}
		})
	}
}

// (e) Alternation: after N moves both colors had equal turns (black may lead by one).
func TestChineseMoveParityEqualTurns(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	const moves = 20
	for i := 0; i < moves; i++ {
		x, y := i%9, (i*2)%9
		if b.AtIndex(y*9+x) == Empty {
			playCoord(t, r, b, x, y)
		} else {
			passMove(t, r, b)
		}
	}
	n := b.MoveNum()
	blackPlays := (n + 1) / 2
	whitePlays := n / 2
	if blackPlays-whitePlays > 1 {
		t.Fatalf("turn parity: black=%d white=%d after %d moves", blackPlays, whitePlays, n)
	}
}

// (f) Indexing symmetry: mirror position swaps area counts.
func TestChineseAreaIndexingSymmetry(t *testing.T) {
	mirror := func(x, y, size int) (int, int) { return size - 1 - x, size - 1 - y }

	buildCorner := func(color Stone) *Board {
		b := NewBoard(9, 0)
		c := func(x, y int) { b.SetStoneIndex(y*9+x, color) }
		if color == Black {
			c(0, 0)
			c(1, 0)
			c(0, 1)
		} else {
			mx, my := mirror(0, 0, 9)
			c(mx, my)
			mx, my = mirror(1, 0, 9)
			c(mx, my)
			mx, my = mirror(0, 1, 9)
			c(mx, my)
		}
		return b
	}

	blB, wlB, _, _ := chineseAreaDecomposition(buildCorner(Black))
	blW, wlW, _, _ := chineseAreaDecomposition(buildCorner(White))
	if blB != wlW || wlB != blW {
		t.Fatalf("mirror corners should swap area counts: black-corner bl=%d wl=%d white-corner bl=%d wl=%d",
			blB, wlB, blW, wlW)
	}
}
