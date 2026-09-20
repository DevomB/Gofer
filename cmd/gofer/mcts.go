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
	if e.cfg.ForcedRootPlayouts > 0 {
		e.runForcedRootPlayouts(b)
	}
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
		need := 0
		// "for each child c of the root that has received any playouts" (Wu 2020,
		// SS3.2). Forcing a floor on every child instead floods the root: on 9x9
		// that is 82 children times k, which at a 200-playout budget outnumbers the
		// search itself and flattens the visit distribution -- the policy target.
		if c.Visits > 0 {
			target := k + int(math.Sqrt(c.Prior*float64(e.cfg.Playouts+1)))
			need = target - int(c.Visits)
		}
		move := c.Move
		e.mu.Unlock()
		for ; need > 0; need-- {
			br := b.Clone()
			e.applyMove(br, move)
			e.descend(br, append(newPath(), e.root, cidx), skipTT)
		}
	}
}

// Whether a playout may be served from the transposition table. Forced root
// playouts skip it: the visit is mandated to shape the policy target, and a
// table hit would credit the child without exploring anything beneath it.
const (
	probeTT = true
	skipTT  = false
)

// newPath returns an empty descent path with room for a typical 9x9 descent, so
// the hot loop does not reallocate on the way down.
func newPath() []int { return make([]int, 0, 24) }

// runPlayout is one ordinary playout from the root.
func (e *Engine) runPlayout(b *Board) {
	e.descend(b.Clone(), append(newPath(), e.root), probeTT)
}

// descend walks from the last node in path down to a leaf, expands it, and backs
// the result up. br must already be the position reached by the moves in path.
func (e *Engine) descend(br *Board, path []int, useTT bool) {
	node := path[len(path)-1]

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
	// An expanded node with no children is a finished game (expandLocked adds no
	// moves there); its result is known exactly and must not be re-estimated.
	terminal := n.Expanded && len(n.Children) == 0
	if !n.Expanded {
		// The transposition key covers stones, not pass history, so a finished
		// position can collide with a live one: never serve it from the table.
		if useTT {
			if v, ok := e.TT.Get(br.Hash()); ok && v.Depth != 0 && !twoPasses(br) {
				e.backupLocked(path, v.Value)
				e.mu.Unlock()
				return
			}
		}
		e.expandLocked(node, br)
		terminal = len(e.arena.Get(node).Children) == 0
	}
	e.mu.Unlock()

	value := e.leafValue(br)
	if terminal {
		value = terminalValue(e.Rules, br)
	}
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
	if e.gameOver(b) {
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
	// Re-fetch: AddChild appends to the arena's slice, which reallocates once the
	// children outgrow its capacity, and `n` would then point into the discarded
	// array. Writing the flag through that stale pointer left the node looking
	// unexpanded forever, so every playout stopped at it and no child was visited.
	e.arena.Get(node).Expanded = true
	e.TT.Store(b.Hash(), Entry{Depth: 1, Value: res.Value})
}

func (e *Engine) selectChildLocked(node int, isRoot bool) int {
	n := e.arena.Get(node)
	parentVisits := float64(n.Visits)
	if parentVisits == 0 {
		parentVisits = 1
	}
	// First-play urgency, relative to how much of the policy has been explored:
	// an unseen move is worth roughly what this node is worth, minus a reduction
	// that grows as the explored moves account for more of the prior. A flat
	// pessimistic constant instead made the first visited child unbeatable, so
	// the search put every playout down one line and the visit distribution --
	// which is the policy training target -- collapsed to a single move.
	fpu := n.Mean()
	if !(isRoot && e.cfg.RootNoise) { // no reduction at a noised root, as in KataGo
		explored := 0.0
		for _, cidx := range n.Children {
			if c := e.arena.Get(cidx); c.Visits > 0 {
				explored += c.Prior
			}
		}
		fpu -= e.cfg.FPU * math.Sqrt(explored)
	}
	best := -1
	bestScore := math.Inf(-1)
	for _, cidx := range n.Children {
		c := e.arena.Get(cidx)
		score := puctScore(c, parentVisits, isRoot, fpu, e.cfg)
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
	// A finished game has a known result; asking the evaluator for one would let
	// the search walk into a lost endgame it cannot see. Only the O(1) two-pass
	// test runs here: a board with no legal move is caught at expansion, and this
	// is the hot path. Checked before the transposition table, whose key covers
	// stones but not pass history.
	if twoPasses(b) {
		return terminalValue(e.Rules, b)
	}
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
