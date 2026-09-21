package main

import "testing"

// Node means are stored from each node's own side to move, so a child's raw
// mean is the OPPONENT's view of the position after that move. puctScore has
// always negated it; the analysis did not, and printed the result as
// "winrate".
//
// The cost was not a wrong move -- selection was correct throughout -- it was a
// wrong conclusion. A collapsed network's analysis showed pass at 0.216 against
// 0.463 for the best real move, which reads as the search rejecting pass while
// visiting it most, i.e. a policy prior overriding the search. Negated, those
// are -0.216 and -0.463: the search preferred pass, and the visit counts agree
// with it. The diagnosis drawn from the inverted number pointed at the policy
// target; the corrected number points at the value function.
func TestRootCandidatesReportTheParentsView(t *testing.T) {
	a := NewArena()
	root := a.Root()
	a.Get(root).Visits = 100

	// Child means are the opponent's view. Lower is better for the parent.
	type child struct {
		move   Move
		mean   float64
		visits uint32
		prior  float64
	}
	children := []child{
		{PassMove, 0.216, 87, 0.50},                       // parent value -0.216: best
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

	// Reported value is the parent's, so it is the negation of the child mean.
	for i, want := range []float64{-0.216, -0.463, -0.610} {
		if got := cands[i].Value; got < want-1e-9 || got > want+1e-9 {
			t.Errorf("candidate %d value = %.3f, want %.3f (parent's frame)", i, got, want)
		}
	}

	// The move with the most visits must also be the one the parent's frame
	// rates highest; if these disagree the report is inverted again.
	if !cands[0].Move.Pass {
		t.Errorf("most-visited candidate is %v, want pass", cands[0].Move)
	}
	if cands[0].Value < cands[1].Value {
		t.Errorf("most-visited move has the worse value (%.3f < %.3f): report is inverted",
			cands[0].Value, cands[1].Value)
	}

	// Prior is carried so a visit share can be attributed; without it a
	// prior-dominated target and a search-chosen one look identical.
	if cands[0].Prior != 0.50 {
		t.Errorf("prior = %.3f, want 0.50", cands[0].Prior)
	}
	if cands[0].PUCT == 0 {
		t.Error("PUCT not reported; visit share cannot be attributed without it")
	}
}
