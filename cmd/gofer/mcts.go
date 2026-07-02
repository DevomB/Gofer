package main

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

const (
	defaultCPUCT      = 1.1
	defaultFPU        = 0.2
	rootTemperature   = 1.03
	virtualLoss       = 3
	defaultForcedRoot = 2
	minPolicyVisits   = 2
)

// SearchConfig holds MCTS parameters.
type SearchConfig struct {
	CPUCT              float64
	FPU                float64
	Playouts           int
	Seed               int64
	RootNoise          bool
	RootTemperature    float64
	Workers            int           // 0 = GOMAXPROCS
	ThinkTime          time.Duration // if >0, search until deadline instead of fixed playouts
	ForcedRootPlayouts int           // paper k=2 at root; 0 disables
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

// Close releases batched evaluator resources if present.
func (e *Engine) Close() {
	if c, ok := e.Eval.(*BatchedEvaluator); ok {
		c.Close()
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

// MoveCandidate is a root move with search statistics.
type MoveCandidate struct {
	Move    Move
	Visits  uint32
	WinRate float64
	Share   float64
}

// Analysis holds search results for a position.
type Analysis struct {
	Playouts   int
	RootValue  float64
	Best       Move
	Candidates []MoveCandidate
	PV         []Move
}

// SetLimits configures playout count or think-time (think-time takes precedence when >0).
func (e *Engine) SetLimits(playouts int, think time.Duration) {
	if think > 0 {
		e.cfg.ThinkTime = think
		return
	}
	e.cfg.ThinkTime = 0
	if playouts > 0 {
		e.cfg.Playouts = playouts
	}
}

// ConfigureSelfplayMove sets per-move search options for mixed playout caps (paper SE-4.1).
func (e *Engine) ConfigureSelfplayMove(playouts int, fullSearch bool) {
	e.cfg.Playouts = playouts
	e.cfg.ThinkTime = 0
	e.cfg.RootNoise = true
	if fullSearch {
		e.cfg.ForcedRootPlayouts = defaultForcedRoot
	} else {
		e.cfg.ForcedRootPlayouts = 0
	}
}

// BestMove runs MCTS and returns the most visited root child move.
func (e *Engine) BestMove(b *Board) Move {
	e.runSearch(b)
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.arena.bestRootMove(e.root)
}

// SelectMove runs MCTS and returns a root move. temperature<=0 plays the most
// visited move; temperature>0 samples proportional to visits^(1/temperature).
func (e *Engine) SelectMove(b *Board, rng *rand.Rand, temperature float64) Move {
	e.runSearch(b)
	e.mu.Lock()
	defer e.mu.Unlock()
	if temperature <= 0 {
		return e.arena.bestRootMove(e.root)
	}
	return e.sampleRootMoveLocked(rng, temperature)
}

func (e *Engine) sampleRootMoveLocked(rng *rand.Rand, temperature float64) Move {
	root := e.arena.Get(e.root)
	if len(root.Children) == 0 {
		return PassMove
	}
	weights := make([]float64, len(root.Children))
	sum := 0.0
	invT := 1.0 / temperature
	for i, cidx := range root.Children {
		v := float64(e.arena.Get(cidx).Visits)
		if v <= 0 {
			continue
		}
		weights[i] = math.Pow(v, invT)
		sum += weights[i]
	}
	if sum <= 0 {
		return e.arena.bestRootMove(e.root)
	}
	target := rng.Float64() * sum
	for i, cidx := range root.Children {
		target -= weights[i]
		if target <= 0 {
			return e.arena.Get(cidx).Move
		}
	}
	return e.arena.Get(root.Children[len(root.Children)-1]).Move
}

// Analyze runs search and returns ranked candidates plus a principal variation.
func (e *Engine) Analyze(b *Board, topN int) Analysis {
	playouts := e.runSearch(b)
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.analyzeLocked(topN, playouts)
}

func (e *Engine) runSearch(b *Board) int {
	if e.arena == nil {
		e.arena = NewArena()
		e.root = e.arena.Root()
	}
	e.ensureRootExpanded(b)
	if e.cfg.ForcedRootPlayouts > 0 {
		e.runForcedRootPlayouts(b)
	}
	if e.cfg.ThinkTime > 0 {
		deadline := time.Now().Add(e.cfg.ThinkTime)
		n := 0
		for time.Now().Before(deadline) {
			e.runPlayout(b, e.root)
			n++
		}
		return n
	}
	e.runPlayouts(b, e.cfg.Playouts)
	return e.cfg.Playouts
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

func (e *Engine) ensureRootExpanded(b *Board) {
	e.mu.Lock()
	defer e.mu.Unlock()
	root := e.arena.Get(e.root)
	if !root.Expanded {
		e.expandLocked(e.root, b)
	}
}

func (e *Engine) runForcedRootPlayouts(b *Board) {
	k := e.cfg.ForcedRootPlayouts
	if k <= 0 {
		k = defaultForcedRoot
	}
	e.mu.Lock()
	root := e.arena.Get(e.root)
	children := append([]int(nil), root.Children...)
	e.mu.Unlock()
	for _, cidx := range children {
		e.mu.Lock()
		c := e.arena.Get(cidx)
		target := k + int(math.Sqrt(c.Prior*float64(e.cfg.Playouts+1)))
		need := target - int(c.Visits)
		e.mu.Unlock()
		for need > 0 {
			e.runPlayoutForced(b, cidx)
			need--
		}
	}
}

func (e *Engine) runPlayoutForced(b *Board, firstChild int) {
	br := b.Clone()
	e.mu.Lock()
	move := e.arena.Get(firstChild).Move
	e.mu.Unlock()
	path := []int{e.root, firstChild}
	e.applyMove(br, move)
	node := firstChild

	for {
		e.mu.Lock()
		n := e.arena.Get(node)
		if !n.Expanded || len(n.Children) == 0 {
			e.mu.Unlock()
			break
		}
		child := e.selectChildLocked(node, false)
		move = e.arena.Get(child).Move
		e.mu.Unlock()
		path = append(path, child)
		e.applyMove(br, move)
		node = child
	}

	e.mu.Lock()
	n := e.arena.Get(node)
	if !n.Expanded {
		e.expandLocked(node, br)
	}
	e.mu.Unlock()

	value := e.leafValue(br)
	e.backup(path, value)
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

func (e *Engine) leafValue(b *Board) float64 {
	hash := b.Hash()
	e.mu.Lock()
	if v, ok := e.TT.Get(hash); ok && v.Depth != 0 {
		e.mu.Unlock()
		return v.Value
	}
	e.mu.Unlock()

	res := e.Eval.Evaluate(b)
	if res.HasValue {
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
