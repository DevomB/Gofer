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

// The newest move belongs in t-1, regardless of history length.
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
		if len(played) >= 2 {
			prev := played[len(played)-2]
			if got := planeOf(spatial, n, prev.Idx(9)); got != firstHistoryPlane+1 {
				t.Errorf("after %d move(s): previous move in plane %d, want %d (t-2)", i+1, got, firstHistoryPlane+1)
			}
		}
		if len(played) >= 3 {
			older := played[len(played)-3]
			if got := planeOf(spatial, n, older.Idx(9)); got != firstHistoryPlane {
				t.Errorf("after %d move(s): older move in plane %d, want %d (t-3)", i+1, got, firstHistoryPlane)
			}
		}
	}
}

// A pass leaves t-1 empty and shifts older moves back.
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
	sums := historyPlaneSums(spatial, n)
	if sums[2] != 0 {
		t.Errorf("t-1 plane has %v stones after a pass, want 0", sums[2])
	}
}

func spatialOf(b *Board) []float32 {
	spatial, _ := BuildFeaturesV2(b)
	return spatial
}
