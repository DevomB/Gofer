package main

import "testing"

// Candidates expose child values in the root player's frame.
func TestRootCandidatesReportTheParentsView(t *testing.T) {
	a := NewArena()
	root := a.Root()
	a.Get(root).Visits = 100

	type child struct {
		move   Move
		mean   float64
		visits uint32
		prior  float64
	}
	children := []child{
		{PassMove, 0.216, 87, 0.50},
		{Move{Point: Point{X: 3, Y: 5}}, 0.463, 65, 0.30}, // parent value -0.463
		{Move{Point: Point{X: 4, Y: 5}}, 0.610, 20, 0.10}, // parent value -0.610: worst
	}
	for _, ch := range children {
		idx := a.AddChild(root, ch.move, ch.prior)
		c := a.Get(idx)
		c.Visits = ch.visits
		c.ValueSum = ch.mean * float64(ch.visits)
	}

	cands := rootCandidates(a, root, 5, DefaultConfig())
	if len(cands) != 3 {
		t.Fatalf("got %d candidates, want 3", len(cands))
	}

	for i, want := range []float64{-0.216, -0.463, -0.610} {
		if got := cands[i].Value; got < want-1e-9 || got > want+1e-9 {
			t.Errorf("candidate %d value = %.3f, want %.3f (parent's frame)", i, got, want)
		}
	}

	if !cands[0].Move.Pass {
		t.Errorf("most-visited candidate is %v, want pass", cands[0].Move)
	}
	if cands[0].Value < cands[1].Value {
		t.Errorf("most-visited move has the worse value (%.3f < %.3f): report is inverted",
			cands[0].Value, cands[1].Value)
	}

	if cands[0].Prior != 0.50 {
		t.Errorf("prior = %.3f, want 0.50", cands[0].Prior)
	}
	if cands[0].PUCT == 0 {
		t.Error("PUCT not reported; visit share cannot be attributed without it")
	}
}
