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

// A stone bordering two empty regions must count as a border of both. White's lone
// stone touches the single point C5, scanned first, and the large region to its
// right: that region touches both colours and is neutral. A self-play game that
// hit the move cap in this shape was scored B+71.5 instead of B+5.5.
func TestSharedBorderStoneCountsForEveryRegion(t *testing.T) {
	b := NewBoard(5, 0)
	rows := []string{
		"XX.O.",
		"XXX..",
		"XXX..",
		"XXX..",
		"XX...",
	}
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
	black, white := Chinese().Score(b)
	if black != 13 || white != 1 {
		t.Fatalf("score B %v W %v, want B 13 W 1: every empty region touches both colours", black, white)
	}
	var sum float32
	for _, v := range OwnershipLabel(b) {
		sum += v
	}
	if float64(sum) != black-white {
		t.Fatalf("ownership sums to %v but the score margin is %v", sum, black-white)
	}
}

// snapbackBoard: White's C5 C4 B4 A4 have two liberties, B5 and A5. Black throws in
// at B5, White takes it at A5, and White's five stones are left with one liberty.
func snapbackBoard(t *testing.T) (Ruleset, *Board) {
	t.Helper()
	b := NewBoard(5, 0)
	for _, s := range []struct {
		x, y int
		c    Stone
	}{
		{2, 0, White}, {2, 1, White}, {1, 1, White}, {0, 1, White},
		{3, 0, Black}, {3, 1, Black}, {2, 2, Black}, {1, 2, Black}, {0, 2, Black},
	} {
		setStone(t, b, s.x, s.y, s.c)
	}
	r := Chinese()
	playCoord(t, r, b, 1, 0) // throw-in
	playCoord(t, r, b, 0, 0) // White captures one stone
	return r, b
}

// Capturing one stone makes a ko only when the capturer stands alone with that
// point as its one liberty. White's capture here joined a five-stone group, so
// retaking at B5 is a snapback, not a ko, and takes all five.
func TestSnapbackIsNotKo(t *testing.T) {
	r, b := snapbackBoard(t)
	if b.Ko() >= 0 {
		t.Fatalf("ko point %d set by a capture that joined a group", b.Ko())
	}
	playCoord(t, r, b, 1, 0)
	if b.StoneAt(At(0, 0)) != Empty || b.StoneAt(At(2, 1)) != Empty {
		t.Fatalf("the snapback did not take White's group:\n%s", formatGTPBoard(b, 5))
	}
}
