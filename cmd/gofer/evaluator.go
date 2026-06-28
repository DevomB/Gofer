package main

// Result holds leaf evaluation from an Evaluator.
type Result struct {
	Value  float64   // from current player perspective, in [-1,1]
	Policy []float32 // optional move priors indexed by point (len size*size+1 for pass)
}

// Evaluator scores positions and optional policy priors (M7 boundary).
type Evaluator interface {
	Evaluate(b *Board) Result
}

// Uniform returns equal priors and zero value (M4-M5 placeholder).
type Uniform struct{}

func (Uniform) Evaluate(b *Board) Result {
	n := b.Size()*b.Size() + 1
	p := make([]float32, n)
	for i := range p {
		p[i] = 1
	}
	return Result{Value: 0, Policy: p}
}

// Heuristic uses stone count diff as value (M7 ponytail).
type Heuristic struct{}

func (Heuristic) Evaluate(b *Board) Result {
	bl, wl := 0, 0
	n := b.Size() * b.Size()
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case Black:
			bl++
		case White:
			wl++
		}
	}
	v := float64(bl-wl) / float64(n)
	if b.Player() == White {
		v = -v
	}
	return Result{Value: v, Policy: nil}
}

// Mock returns fixed value/policy for tests.
type Mock struct {
	Value  float64
	Policy []float32
}

func (m Mock) Evaluate(b *Board) Result {
	return Result{Value: m.Value, Policy: m.Policy}
}
