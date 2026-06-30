package main

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
)

const (
	defaultCPUCT    = 1.1
	defaultFPU      = 0.2
	dirichletAlpha  = 0.03
	dirichletBlend  = 0.25
	rootTemperature = 1.03
	maxRolloutMoves = 150
	virtualLoss     = 3
)

// SearchConfig holds MCTS parameters.
type SearchConfig struct {
	CPUCT           float64
	FPU             float64
	Playouts        int
	Seed            int64
	RootNoise       bool
	RootTemperature float64
	Workers         int // 0 = GOMAXPROCS
}

// DefaultConfig returns search defaults aligned with Wu 2020.
func DefaultConfig() SearchConfig {
	return SearchConfig{
		CPUCT:           defaultCPUCT,
		FPU:             defaultFPU,
		Playouts:        100,
		Seed:            1,
		RootNoise:       false,
		RootTemperature: rootTemperature,
	}
}

// Engine runs MCTS search.
type Engine struct {
	Rules  Ruleset
	Eval   Evaluator
	TT     *Table
	cfg    SearchConfig
	rng    *rand.Rand
	arena  *Arena
	root   int
	mu     sync.Mutex
	rngSeq uint64
}

// NewEngine constructs an MCTS search engine.
func NewEngine(r Ruleset, ev Evaluator, cfg SearchConfig) *Engine {
	if cfg.CPUCT == 0 {
		cfg = DefaultConfig()
	}
	if ev == nil {
		ev = Uniform{}
	}
	return &Engine{
		Rules: r,
		Eval:  ev,
		TT:    NewTable(1 << 16),
		cfg:   cfg,
		rng:   rand.New(rand.NewSource(cfg.Seed)),
	}
}

// ResetArena clears the search tree.
func (e *Engine) ResetArena() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.arena = nil
	e.root = 0
}

// AdvanceTree moves the search root to the child matching m, or resets on miss.
func (e *Engine) AdvanceTree(m Move) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.arena == nil {
		return
	}
	n := e.arena.Get(e.root)
	for _, cidx := range n.Children {
		c := e.arena.Get(cidx)
		if movesEqual(c.Move, m) {
			e.root = cidx
			return
		}
	}
	e.arena = nil
	e.root = 0
}

// BestMove runs MCTS and returns the most visited root child move.
func (e *Engine) BestMove(b *Board) Move {
	if e.arena == nil {
		e.arena = NewArena()
		e.root = e.arena.Root()
	}
	e.runPlayouts(b, e.cfg.Playouts)
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.selectBestRootLocked(e.root)
}

func (e *Engine) runPlayouts(b *Board, playouts int) {
	workers := e.cfg.Workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers < 1 {
		workers = 1
	}
	if workers == 1 || playouts < workers {
		for i := 0; i < playouts; i++ {
			e.runPlayout(b, e.root)
		}
		return
	}
	perWorker := playouts / workers
	extra := playouts % workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		n := perWorker
		if w < extra {
			n++
		}
		if n == 0 {
			continue
		}
		wg.Add(1)
		go func(count int) {
			defer wg.Done()
			for i := 0; i < count; i++ {
				e.runPlayout(b, e.root)
			}
		}(n)
	}
	wg.Wait()
}

// RootPolicy returns visit-weighted policy over legal moves.
func (e *Engine) RootPolicy(legal []Move) []float32 {
	pi := make([]float32, len(legal))
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.arena == nil || len(e.arena.nodes) == 0 {
		return uniformPolicy32(len(legal))
	}
	root := e.arena.Get(e.root)
	var total uint32
	for _, cidx := range root.Children {
		total += e.arena.Get(cidx).Visits
	}
	if total == 0 {
		return uniformPolicy32(len(legal))
	}
	inv := 1 / float32(total)
	for _, cidx := range root.Children {
		c := e.arena.Get(cidx)
		for i, m := range legal {
			if movesEqual(c.Move, m) {
				pi[i] = float32(c.Visits) * inv
				break
			}
		}
	}
	return pi
}

func (e *Engine) runPlayout(b *Board, root int) {
	br := b.Clone()
	path := make([]int, 0, 24)
	path = append(path, root)
	node := root

	for {
		e.mu.Lock()
		n := e.arena.Get(node)
		if !n.Expanded || len(n.Children) == 0 {
			e.mu.Unlock()
			break
		}
		child := e.selectChildLocked(node, node == e.root)
		move := e.arena.Get(child).Move
		e.mu.Unlock()
		path = append(path, child)
		e.applyMove(br, move)
		node = child
	}

	e.mu.Lock()
	n := e.arena.Get(node)
	if !n.Expanded {
		if v, ok := e.ttGetLocked(br.Hash()); ok {
			e.backupLocked(path, v)
			e.mu.Unlock()
			return
		}
		e.expandLocked(node, br)
	}
	e.mu.Unlock()

	value := e.leafValue(br)
	e.backup(path, value)
}

func (e *Engine) backup(path []int, value float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.backupLocked(path, value)
}

func (e *Engine) backupLocked(path []int, value float64) {
	for i := len(path) - 1; i >= 0; i-- {
		nd := e.arena.Get(path[i])
		if i > 0 && nd.Visits >= virtualLoss {
			nd.Visits -= virtualLoss
		}
		nd.Visits++
		nd.ValueSum += value
		value = -value
	}
}

func (e *Engine) applyMove(br *Board, m Move) {
	if m.Pass {
		e.Rules.Play(br, PassMove)
	} else {
		e.Rules.Play(br, StoneMove(m.Point))
	}
}

func (e *Engine) expandLocked(node int, b *Board) {
	n := e.arena.Get(node)
	if n.Expanded {
		return
	}
	if e.isTerminal(b) {
		n.Expanded = true
		return
	}
	moves := e.Rules.LegalMoves(b)
	res := e.Eval.Evaluate(b)
	priors := uniformPriors(len(moves))
	if len(res.Policy) > 0 {
		priors = policyPriors(b, moves, res.Policy)
	}
	if node == e.root && e.cfg.RootNoise {
		priors = blendDirichlet(priors, e.rng)
	}
	for i, m := range moves {
		e.arena.AddChild(node, m, priors[i])
	}
	n.Expanded = true
	e.ttStoreLocked(b.Hash(), Entry{Depth: 1, Value: res.Value})
}

func (e *Engine) selectChildLocked(node int, isRoot bool) int {
	n := e.arena.Get(node)
	parentVisits := float64(n.Visits)
	if parentVisits == 0 {
		parentVisits = 1
	}
	best := -1
	bestScore := math.Inf(-1)
	for _, cidx := range n.Children {
		c := e.arena.Get(cidx)
		score := e.puctScore(c, parentVisits, isRoot)
		if score > bestScore {
			bestScore = score
			best = cidx
		}
	}
	if best >= 0 {
		e.arena.Get(best).Visits += virtualLoss
	}
	return best
}

func (e *Engine) puctScore(c *Node, parentVisits float64, isRoot bool) float64 {
	q := c.Mean()
	if c.Visits == 0 {
		q = -e.cfg.FPU
	}
	u := e.cfg.CPUCT * c.Prior * math.Sqrt(parentVisits) / (1 + float64(c.Visits))
	if isRoot && e.cfg.RootTemperature != 1 && c.Visits > 0 {
		q /= e.cfg.RootTemperature
	}
	return q + u
}

func (e *Engine) ttGetLocked(hash uint64) (float64, bool) {
	entry, ok := e.TT.Get(hash)
	if !ok || entry.Depth == 0 {
		return 0, false
	}
	return entry.Value, true
}

func (e *Engine) ttStoreLocked(hash uint64, entry Entry) {
	e.TT.Store(hash, entry)
}

func (e *Engine) leafValue(b *Board) float64 {
	hash := b.Hash()
	e.mu.Lock()
	if v, ok := e.ttGetLocked(hash); ok {
		e.mu.Unlock()
		return v
	}
	e.mu.Unlock()

	res := e.Eval.Evaluate(b)
	if res.Value != 0 {
		e.mu.Lock()
		e.ttStoreLocked(hash, Entry{Depth: 1, Value: res.Value})
		e.mu.Unlock()
		return res.Value
	}
	v := e.randomPlayout(b)
	e.mu.Lock()
	e.ttStoreLocked(hash, Entry{Depth: 1, Value: v})
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

func (e *Engine) selectBestRootLocked(root int) Move {
	n := e.arena.Get(root)
	if len(n.Children) == 0 {
		return PassMove
	}
	best := n.Children[0]
	maxV := uint32(0)
	for _, cidx := range n.Children {
		c := e.arena.Get(cidx)
		if c.Visits > maxV {
			maxV = c.Visits
			best = cidx
		}
	}
	return e.arena.Get(best).Move
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

func movesEqual(a, b Move) bool {
	if a.Pass != b.Pass {
		return false
	}
	if a.Pass {
		return true
	}
	return a.Point == b.Point
}

// PUCTScore exposes the PUCT formula for unit tests.
func PUCTScore(q, prior, parentVisits float64, visits uint32, cPUCT float64) float64 {
	u := cPUCT * prior * math.Sqrt(parentVisits) / (1 + float64(visits))
	return q + u
}

// TTHitRate returns the fraction of transposition-table lookups that hit.
func (e *Engine) TTHitRate(b *Board, probes int) float64 {
	if probes <= 0 {
		return 0
	}
	hits := 0
	for i := 0; i < probes; i++ {
		if _, ok := e.TT.Get(b.Hash()); ok {
			hits++
		}
	}
	return float64(hits) / float64(probes)
}

// TreeSize returns the number of nodes in the search tree.
func (e *Engine) TreeSize() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.arena == nil {
		return 0
	}
	return len(e.arena.nodes)
}
