package main

// Inference is an external NN adapter (M11 — mock/sidecar hook).
type Inference struct {
	// ponytail: no ONNX runtime in-
	// Ceiling: mock weights only.
	// Upgrade: sidecar ONNX/Torch via HTTP or CGO.
	MockValue  float64
	MockPolicy []float32
}

// Evaluate implements Evaluator via mock/sidecar hook.
func (inf Inference) Evaluate(b *Board) Result {
	n := b.Size()*b.Size() + 1
	p := inf.MockPolicy
	if len(p) != n {
		p = make([]float32, n)
		for i := range p {
			p[i] = 1
		}
	}
	return Result{Value: inf.MockValue, Policy: p}
}

// GatingHarness compares win rates for model promotion (M11).
type GatingHarness struct {
	Games            int
	MinWinRateMargin float64 // e.g. 0.55 vs 0.45
}

// Pass returns true if candidate beats baseline by margin over Games.
func (g GatingHarness) Pass(baselineWin, candidateWin float64) bool {
	if g.Games <= 0 {
		return true
	}
	margin := g.MinWinRateMargin
	if margin == 0 {
		margin = 0.55
	}
	return candidateWin >= margin && candidateWin > baselineWin
}
