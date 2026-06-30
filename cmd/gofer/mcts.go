package main

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

const (
	defaultCPUCT    = 1.1
	defaultFPU      = 0.2
	rootTemperature = 1.03
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
	return e.arena.bestRootMove(e.root)
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
		if v, ok := e.TT.Get(br.Hash()); ok && v.Depth != 0 {
			e.backupLocked(path, v.Value)
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
	e.TT.Store(b.Hash(), Entry{Depth: 1, Value: res.Value})
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
		score := puctScore(c, parentVisits, isRoot, e.cfg)
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
