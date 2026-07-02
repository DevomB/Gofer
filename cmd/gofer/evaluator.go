package main

import (
	"math"
	"math/rand"
)

const (
	// dirichletAlpha ~ 10/avg-legal-moves; 9x9 opens with ~70 moves. The AZ 19x19
	// value (0.03) is far too peaky here and kills root exploration on 9x9.
	dirichletAlpha  = 0.15
	dirichletBlend  = 0.25
	maxRolloutMoves = 150
)

// Result holds leaf evaluation from an Evaluator.
type Result struct {
	Value    float64   // from current player perspective, in [-1,1]
	Policy   []float32 // optional move priors indexed by point (len size*size+1 for pass)
	HasValue bool      // true when Value is a real estimate (false => caller may roll out)
}

// Evaluator scores positions and optional policy priors.
type Evaluator interface {
	Evaluate(b *Board) Result
}

// Uniform returns equal priors and zero value.
type Uniform struct{}

func (Uniform) Evaluate(b *Board) Result {
	n := b.Size()*b.Size() + 1
	p := make([]float32, n)
	for i := range p {
		p[i] = 1
	}
	return Result{Value: 0, Policy: p}
}

// Heuristic combines material, liberties, territory, and move priors.
type Heuristic struct{}

func (Heuristic) Evaluate(b *Board) Result {
	size := b.Size()
	n := size * size
	blS, wlS := 0, 0
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case Black:
			blS++
		case White:
			wlS++
		}
	}
	blL, wlL := groupLibertyTotals(b, Black), groupLibertyTotals(b, White)
	blT, wlT := estimateTerritory(b)
	score := float64(blS-wlS) + 0.15*float64(blL-wlL) + float64(blT-wlT)
	v := score / float64(max(n, 1))
	if b.Player() == White {
		v = -v
	}
	v = clamp(v, -1, 1)
	return Result{Value: v, Policy: heuristicPolicy(b), HasValue: true}
}

func groupLibertyTotals(b *Board, color Color) int {
	n := b.Size() * b.Size()
	seen := make([]bool, n)
	total := 0
	for i := 0; i < n; i++ {
		if seen[i] || b.AtIndex(i) != color {
			continue
		}
		for _, idx := range collectGroup(b, i, color) {
			seen[idx] = true
		}
		total += libertyCount(b, i, color)
	}
	return total
}

func estimateTerritory(b *Board) (black, white int) {
	n := b.Size() * b.Size()
	seen := make([]bool, n)
	for i := 0; i < n; i++ {
		if b.AtIndex(i) != Empty || seen[i] {
			continue
		}
		t, tb, tw := floodEmpty(b, i, seen)
		if tb && !tw {
			black += t
		}
		if tw && !tb {
			white += t
		}
	}
	return black, white
}

func heuristicPolicy(b *Board) []float32 {
	size := b.Size()
	n := size*size + 1
	p := make([]float32, n)
	player := b.Player()
	opp := player.Opposite()
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			pt := At(x, y)
			if b.StoneAt(pt) != Empty {
				continue
			}
			p[pt.Idx(size)] = policyPriorAt(b, pt, player, opp)
		}
	}
	p[size*size] = 0.02
	return p
}

func policyPriorAt(b *Board, pt Point, player, opp Color) float32 {
	score := float32(0.05)
	for _, nb := range b.Neighbors(pt.Idx(b.Size())) {
		switch b.AtIndex(nb) {
		case player:
			score += 1
		case opp:
			score -= 0.4
		}
	}
	if score < 0.01 {
		return 0.01
	}
	return score
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Mock returns fixed value/policy for tests.
type Mock struct {
	Value  float64
	Policy []float32
}

func (m Mock) Evaluate(b *Board) Result {
	return Result{Value: m.Value, Policy: m.Policy, HasValue: true}
}

func uniformPriors(n int) []float64 {
	if n == 0 {
		return nil
	}
	p := 1.0 / float64(n)
	out := make([]float64, n)
	for i := range out {
		out[i] = p
	}
	return out
}

func uniformPolicy32(n int) []float32 {
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	for i := range out {
		out[i] = 1 / float32(n)
	}
	return out
}

func policyPriors(b *Board, moves []Move, policy []float32) []float64 {
	size := b.Size()
	sum := float64(0)
	raw := make([]float64, len(moves))
	for i, m := range moves {
		idx := size * size
		if !m.Pass {
			idx = m.Point.Idx(size)
		}
		if idx >= 0 && idx < len(policy) {
			raw[i] = float64(policy[idx])
		} else {
			raw[i] = 1
		}
		sum += raw[i]
	}
	if sum == 0 {
		return uniformPriors(len(moves))
	}
	for i := range raw {
		raw[i] /= sum
	}
	return raw
}

// blendDirichlet mixes symmetric Dirichlet(alpha) noise into root priors (AZ SE-4.1).
func blendDirichlet(priors []float64, rng *rand.Rand) []float64 {
	noise := sampleDirichlet(len(priors), dirichletAlpha, rng)
	out := make([]float64, len(priors))
	for i := range out {
		out[i] = (1-dirichletBlend)*priors[i] + dirichletBlend*noise[i]
	}
	return out
}

// sampleDirichlet draws a symmetric Dirichlet(alpha) vector via normalized Gamma samples.
func sampleDirichlet(n int, alpha float64, rng *rand.Rand) []float64 {
	out := make([]float64, n)
	sum := 0.0
	for i := range out {
		out[i] = sampleGamma(alpha, rng)
		sum += out[i]
	}
	if sum <= 0 {
		return uniformPriors(n)
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// sampleGamma draws from Gamma(shape, 1) using Marsaglia-Tsang, with the
// shape<1 boost trick. Correct for the small alpha we use at the root.
func sampleGamma(shape float64, rng *rand.Rand) float64 {
	if shape < 1 {
		return sampleGamma(shape+1, rng) * math.Pow(rng.Float64(), 1/shape)
	}
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}
