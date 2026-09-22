package main

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

const (
	defaultCPUCT      = 1.1
	defaultFPU        = 0.2
	rootTemperature   = 1.03
	virtualLoss       = 3
	defaultForcedRoot = 2
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
	ForcedRootPlayouts int           // Wu 2020 k: visited root children get sqrt(k*P*N) visits; 0 disables
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
	Rules Ruleset
	Eval  Evaluator
	cfg   SearchConfig
	rng   *rand.Rand
	arena *Arena
	root  int
	mu    sync.Mutex
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
	Move   Move
	Visits uint32
	// Value is from the parent side to move's perspective.
	Value float64
	Share float64
	Prior float64
	PUCT  float64
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
	// Fast searches only choose the move played; exploration there just weakens it.
	e.cfg.RootNoise = fullSearch
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
	e.prepareRoot(b)
	if e.cfg.ThinkTime > 0 {
		deadline := time.Now().Add(e.cfg.ThinkTime)
		n := 0
		for time.Now().Before(deadline) {
			e.runPlayout(b)
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
			e.runPlayout(b)
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
				e.runPlayout(b)
			}
		}(n)
	}
	wg.Wait()
}

// prepareRoot expands the root, or re-derives its priors when an earlier search
// expanded it: a root reused through AdvanceTree was expanded as an ordinary node,
// without the temperature and noise every search's root is meant to have.
func (e *Engine) prepareRoot(b *Board) {
	e.mu.Lock()
	defer e.mu.Unlock()
	root := e.arena.Get(e.root)
	if !root.Expanded {
		e.expandLocked(e.root, b)
		return
	}
	if len(root.Children) == 0 || (e.cfg.RootTemperature == 1 && !e.cfg.RootNoise) {
		return
	}
	moves := make([]Move, len(root.Children))
	for i, cidx := range root.Children {
		moves[i] = e.arena.Get(cidx).Move
	}
	priors := e.childPriorsLocked(e.root, b, moves, e.Eval.Evaluate(b).Policy)
	for i, cidx := range root.Children {
		e.arena.Get(cidx).Prior = priors[i]
	}
}

// runPlayout walks from the root to a leaf, expands it, and backs the result up.
func (e *Engine) runPlayout(b *Board) {
	br := b.Clone()
	node := e.root
	// Room for a typical 9x9 descent, so the hot loop does not reallocate.
	path := append(make([]int, 0, 24), node)
	var res Result
	// expandLocked adds no moves to a finished game, so a childless node's result
	// is known exactly and must not be estimated.
	terminal := false
	for {
		e.mu.Lock()
		n := e.arena.Get(node)
		if !n.Expanded {
			res = e.expandLocked(node, br)
			terminal = len(e.arena.Get(node).Children) == 0
			e.mu.Unlock()
			break
		}
		if len(n.Children) == 0 {
			terminal = true
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

	value := res.Value
	if terminal {
		value = terminalValue(e.Rules, br)
	}
	e.backup(path, value)
}

func (e *Engine) backup(path []int, value float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
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

// applyMove plays a move the search took from LegalMoves; searching on after a
// refusal would score the wrong position.
func (e *Engine) applyMove(br *Board, m Move) {
	if err := playChosen(e.Rules, br, m); err != nil {
		panic(err)
	}
}

// expandLocked adds node's children and returns the evaluation it made them from.
func (e *Engine) expandLocked(node int, b *Board) Result {
	n := e.arena.Get(node)
	if n.Expanded {
		return Result{}
	}
	if e.gameOver(b) {
		n.Expanded = true
		return Result{}
	}
	moves := searchMoves(e.Rules, b)
	res := e.Eval.Evaluate(b)
	priors := e.childPriorsLocked(node, b, moves, res.Policy)
	for i, m := range moves {
		e.arena.AddChild(node, m, priors[i])
	}
	// Re-fetch after AddChild; arena storage can grow.
	e.arena.Get(node).Expanded = true
	return res
}

// searchMoves is what the search may play: the legal moves less obvious own-eye
// fills (ownEye). Those only lose a liberty, and a side that never runs out of
// them never passes, so games did not end. The rules still allow them.
func searchMoves(r Ruleset, b *Board) []Move {
	moves := r.LegalMoves(b)
	player := b.Player()
	out := moves[:0]
	for _, m := range moves {
		if m.Pass || !ownEye(b, m.Point.Idx(b.Size()), player) {
			out = append(out, m)
		}
	}
	return out
}

// childPriorsLocked turns policy into priors over moves. At the root it also
// applies the policy temperature and then exploration noise.
func (e *Engine) childPriorsLocked(node int, b *Board, moves []Move, policy []float32) []float64 {
	priors := uniformPriors(len(moves))
	if len(policy) > 0 {
		priors = policyPriors(b, moves, policy)
	}
	if node != e.root {
		return priors
	}
	if e.cfg.RootTemperature != 1 {
		priors = temperPriors(priors, e.cfg.RootTemperature)
	}
	if e.cfg.RootNoise {
		priors = blendDirichlet(priors, e.rng)
	}
	return priors
}

// fpuFor returns an unvisited child's first-play urgency in the parent frame.
func fpuFor(a *Arena, n *Node, isRoot bool, cfg SearchConfig) float64 {
	fpu := n.Mean()
	if isRoot && cfg.RootNoise {
		return fpu // no reduction at a noised root, as in KataGo
	}
	explored := 0.0
	for _, cidx := range n.Children {
		if c := a.Get(cidx); c.Visits > 0 {
			explored += c.Prior
		}
	}
	return fpu - cfg.FPU*math.Sqrt(explored)
}

func (e *Engine) selectChildLocked(node int, isRoot bool) int {
	n := e.arena.Get(node)
	best := -1
	if isRoot {
		best = forcedChild(e.arena, n, e.cfg.ForcedRootPlayouts)
	}
	if best < 0 {
		parentVisits := float64(n.Visits)
		if parentVisits == 0 {
			parentVisits = 1
		}
		fpu := fpuFor(e.arena, n, isRoot, e.cfg)
		bestScore := math.Inf(-1)
		for _, cidx := range n.Children {
			score := puctScore(e.arena.Get(cidx), parentVisits, fpu, e.cfg)
			if score > bestScore {
				bestScore = score
				best = cidx
			}
		}
	}
	if best >= 0 {
		e.arena.Get(best).Visits += virtualLoss
	}
	return best
}

// forcedChild returns a visited root child still below its forced playouts, or
// -1. Wu 2020 §3.2 forces only children the search has already tried.
func forcedChild(a *Arena, root *Node, k int) int {
	if k <= 0 {
		return -1
	}
	total := float64(rootVisitTotal(a, root))
	for _, cidx := range root.Children {
		c := a.Get(cidx)
		if c.Visits > 0 && float64(c.Visits) < forcedPlayouts(k, c.Prior, total) {
			return cidx
		}
	}
	return -1
}

func forcedPlayouts(k int, prior, total float64) float64 {
	return math.Sqrt(float64(k) * prior * total)
}

func (e *Engine) isTerminal(b *Board) bool {
	for _, m := range e.Rules.LegalMoves(b) {
		if !m.Pass {
			return false
		}
	}
	return true
}

// gameOver reports whether play has ended under the same rule the game loops
// use: two consecutive passes, or no legal move other than pass.
func (e *Engine) gameOver(b *Board) bool {
	if twoPasses(b) {
		return true
	}
	return e.isTerminal(b)
}

func twoPasses(b *Board) bool {
	h := b.historyMoves(2)
	return len(h) == 2 && h[0].move.Pass && h[1].move.Pass
}

// terminalValue scores a finished board from the side to move's perspective.
func terminalValue(rs Ruleset, b *Board) float64 {
	black, white := rs.Score(b)
	diff := black - white
	if b.Player() == White {
		diff = -diff
	}
	switch {
	case diff > 0:
		return 1
	case diff < 0:
		return -1
	default:
		return 0
	}
}
