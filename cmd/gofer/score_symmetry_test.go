package main

import "testing"

// Mirror all stones on the board; scores should swap (before komi, which is symmetric on white).
func TestChineseScoreMirrorSymmetry(t *testing.T) {
	r := Chinese()
	size := 9
	// Black L-corner at (0,0); white mirror L at (8,8). Position is mirror-symmetric
	// with colors swapped, so Score(mirror) should exchange black and white area.
	build := func() *Board {
		b := NewBoard(size, 0)
		for _, c := range [][2]int{{0, 0}, {1, 0}, {0, 1}} {
			b.SetStoneIndex(c[1]*size+c[0], Black)
		}
		for _, c := range [][2]int{{8, 8}, {7, 8}, {8, 7}} {
			b.SetStoneIndex(c[1]*size+c[0], White)
		}
		return b
	}
	mirrorBoard := func(b *Board) *Board {
		m := NewBoard(size, 0)
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				s := b.StoneAt(At(x, y))
				if s == Empty {
					continue
				}
				mx, my := size-1-x, size-1-y
				m.SetStoneIndex(my*size+mx, s)
			}
		}
		return m
	}
	orig := build()
	mir := mirrorBoard(orig)
	bl, wl := r.Score(orig)
	mbl, mwl := r.Score(mir)
	if bl != mwl || wl != mbl {
		t.Fatalf("mirror should swap area scores: orig B=%.0f W=%.0f mirrored B=%.0f W=%.0f", bl, wl, mbl, mwl)
	}
}

// Enclosed territories: only one color touches each empty region.
func TestChineseScoreEnclosedTerritories(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 0)
	// Black surrounds center 3x3 at (3..5, 3..5) with a gap at (4,4) for one white dead stone.
	for y := 3; y <= 5; y++ {
		for x := 3; x <= 5; x++ {
			if x == 4 && y == 4 {
				b.SetStoneIndex(y*9+x, White)
				continue
			}
			b.SetStoneIndex(y*9+x, Black)
		}
	}
	// Outside liberty so enclosure is meaningful.
	b.SetStoneIndex(0, Black)
	bl, wl, neutral, _ := chineseAreaDecomposition(b)
	if neutral != 0 {
		t.Fatalf("expected no neutral in single-color enclosure, neutral=%d", neutral)
	}
	// 8 black stones in ring + 1 white inside + 72 outside empty touches black via (0,0).
	if bl < 8 || wl != 1 {
		t.Fatalf("want black>=8 white=1 (dead stone counts for owner) got black=%d white=%d", bl, wl)
	}
	_, wScore := r.Score(b)
	if wScore != 1 {
		t.Fatalf("komi=0 white score should equal white area=1 got %v", wScore)
	}
	_ = bl
}
