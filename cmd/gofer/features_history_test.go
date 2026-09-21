package main

import "testing"

// planeOf reports which history plane (5, 6 or 7) carries the given point, or
// -1 if none does.
func planeOf(spatial []float32, n int, idx int) int {
	for p := firstHistoryPlane; p < firstHistoryPlane+historyPlanes; p++ {
		if spatial[p*n+idx] == 1 {
			return p
		}
	}
	return -1
}

func historyPlaneSums(spatial []float32, n int) [historyPlanes]float32 {
	var out [historyPlanes]float32
	for h := 0; h < historyPlanes; h++ {
		for i := 0; i < n; i++ {
			out[h] += spatial[(firstHistoryPlane+h)*n+i]
		}
	}
	return out
}

// The schema says plane 7 is "the stone played 1 ply ago". The history was
// left-aligned, so after one move that stone was in plane 5 and plane 7 was
// empty -- and an empty plane 7 is exactly how a pass is encoded. For the first
// two plies of every game a real move and a pass were the same input.
func TestHistoryPlanesAreRightAligned(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	n := 9 * 9

	// Zero moves: every history plane empty.
	if sums := historyPlaneSums(spatialOf(b), n); sums != [historyPlanes]float32{0, 0, 0} {
		t.Errorf("empty board history planes = %v, want all zero", sums)
	}

	type step struct {
		pt        Point
		wantPlane int
	}
	// After each move the move just played must be in plane 7 (t-1), the one
	// before it in 6, and the one before that in 5.
	steps := []step{
		{Point{X: 2, Y: 2}, firstHistoryPlane + 2},
		{Point{X: 4, Y: 4}, firstHistoryPlane + 2},
		{Point{X: 6, Y: 6}, firstHistoryPlane + 2},
		{Point{X: 1, Y: 7}, firstHistoryPlane + 2},
	}
	played := make([]Point, 0, len(steps))
	for i, s := range steps {
		if !r.Play(b, Move{Point: s.pt}) {
			t.Fatalf("step %d: illegal move %v", i, s.pt)
		}
		played = append(played, s.pt)
		spatial := spatialOf(b)

		if got := planeOf(spatial, n, s.pt.Idx(9)); got != s.wantPlane {
			t.Errorf("after %d move(s): last move in plane %d, want %d (t-1)", i+1, got, s.wantPlane)
		}
		// The previous move, when there is one, must be exactly one plane older.
		if len(played) >= 2 {
			prev := played[len(played)-2]
			if got := planeOf(spatial, n, prev.Idx(9)); got != firstHistoryPlane+1 {
				t.Errorf("after %d move(s): previous move in plane %d, want %d (t-2)", i+1, got, firstHistoryPlane+1)
			}
		}
	}
}

// A pass writes nothing, so it must shift the older moves along rather than
// leaving them where they were: after "move, pass" the move is t-2, not t-1.
func TestPassShiftsHistoryWithoutWritingAPlane(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	n := 9 * 9
	pt := Point{X: 3, Y: 3}

	if !r.Play(b, Move{Point: pt}) {
		t.Fatal("illegal opening move")
	}
	if got := planeOf(spatialOf(b), n, pt.Idx(9)); got != firstHistoryPlane+2 {
		t.Fatalf("move is in plane %d, want t-1", got)
	}

	if !r.Play(b, PassMove) {
		t.Fatal("pass rejected")
	}
	spatial := spatialOf(b)
	if got := planeOf(spatial, n, pt.Idx(9)); got != firstHistoryPlane+1 {
		t.Errorf("after a pass the earlier move is in plane %d, want %d (t-2)", got, firstHistoryPlane+1)
	}
	// t-1 is empty, which is what tells the network the last move was a pass.
	sums := historyPlaneSums(spatial, n)
	if sums[2] != 0 {
		t.Errorf("t-1 plane has %v stones after a pass, want 0", sums[2])
	}
}

func spatialOf(b *Board) []float32 {
	spatial, _ := BuildFeaturesV2(b)
	return spatial
}
