package main

// Inference is a mock neural-network evaluator hook for external runtimes.
type Inference struct {
	// ponytail: mock weights only; wire ONNX or sidecar when a model exists.
	MockValue  float64
	MockPolicy []float32
}

// Evaluate implements Evaluator.
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

// GatingHarness compares win rates for model promotion.
type GatingHarness struct {
	Games            int
	MinWinRateMargin float64 // e.g. 0.55 vs 0.45
}

// Pass returns true if candidate beats baseline by the configured margin.
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
