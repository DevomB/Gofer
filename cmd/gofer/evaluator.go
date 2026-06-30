package main

import (
	"math"
	"math/rand"
	"sync/atomic"
)

const (
	dirichletAlpha  = 0.03
	dirichletBlend  = 0.25
	maxRolloutMoves = 150
)

// Result holds leaf evaluation from an Evaluator.
type Result struct {
	Value  float64   // from current player perspective, in [-1,1]
	Policy []float32 // optional move priors indexed by point (len size*size+1 for pass)
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
	return Result{Value: v, Policy: heuristicPolicy(b)}
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
			score := float32(0.05)
			for _, nb := range b.Neighbors(pt.Idx(size)) {
				switch b.AtIndex(nb) {
				case player:
					score += 1
				case opp:
					score -= 0.4
				}
			}
			if score < 0.01 {
				score = 0.01
			}
			p[pt.Idx(size)] = score
		}
	}
	p[size*size] = 0.02
	return p
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
	return Result{Value: m.Value, Policy: m.Policy}
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

func blendDirichlet(priors []float64, rng *rand.Rand) []float64 {
	out := make([]float64, len(priors))
	sum := 0.0
	noise := make([]float64, len(priors))
	for i := range noise {
		noise[i] = math.Pow(rng.Float64(), 1/dirichletAlpha)
		sum += noise[i]
	}
	for i := range out {
		n := noise[i] / sum
		out[i] = (1-dirichletBlend)*priors[i] + dirichletBlend*n
	}
	return out
}

func (e *Engine) leafValue(b *Board) float64 {
	hash := b.Hash()
	e.mu.Lock()
	if v, ok := e.TT.Get(hash); ok && v.Depth != 0 {
		e.mu.Unlock()
		return v.Value
	}
	e.mu.Unlock()

	res := e.Eval.Evaluate(b)
	if res.Value != 0 {
		e.mu.Lock()
		e.TT.Store(hash, Entry{Depth: 1, Value: res.Value})
		e.mu.Unlock()
		return res.Value
	}
	v := e.randomPlayout(b)
	e.mu.Lock()
	e.TT.Store(hash, Entry{Depth: 1, Value: v})
	e.mu.Unlock()
	return v
}

func (e *Engine) randomPlayout(b *Board) float64 {
	rng := e.playoutRand()
	br := b.Clone()
	player := br.Player()
	passes := 0
	for move := 0; move < maxRolloutMoves && passes < 2; move++ {
		moves := e.Rules.LegalMoves(br)
		if len(moves) == 0 {
			break
		}
		m := moves[rng.Intn(len(moves))]
		e.Rules.Play(br, m)
		if m.Pass {
			passes++
		} else {
			passes = 0
		}
	}
	bl, wl := e.Rules.Score(br)
	diff := bl - wl
	if player == White {
		diff = wl - bl
	}
	if diff > 0 {
		return 1
	}
	if diff < 0 {
		return -1
	}
	return 0
}

func (e *Engine) playoutRand() *rand.Rand {
	seq := atomic.AddUint64(&e.rngSeq, 1)
	return rand.New(rand.NewSource(e.cfg.Seed + int64(seq)))
}

func (e *Engine) isTerminal(b *Board) bool {
	for _, m := range e.Rules.LegalMoves(b) {
		if !m.Pass {
			return false
		}
	}
	return true
}
