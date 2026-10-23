// Package mcts implements PUCT Monte Carlo tree search over a neural
// network evaluator (AlphaZero/KataGo style).
//
// Sign convention: every node stores W, the sum of backed-up values from the
// perspective of the player who made the move leading to that node. A parent
// therefore maximizes its children's Q = W/N directly. A network value v is
// from the side to move at the evaluated position, so the leaf receives -v,
// its parent +v, and so on up the path.
package mcts

import (
	"errors"
	"math"
	"math/rand"
	"sort"

	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/nn"
)

// Config controls the search.
type Config struct {
	CPuct            float64 // exploration constant
	FPUReduction     float64 // unvisited child value = parent value - reduction*sqrt(visited prior mass)
	RootFPUReduction float64
	RootNoise        bool    // mix Dirichlet noise into root priors (self-play only)
	NoiseAlpha       float64 // 0 = 10.83/legal moves (AlphaZero's 0.03 scaled to the board)
	NoiseFrac        float64
	Batch            int // leaves gathered (with virtual loss) per evaluator call
	MaxMoves         int // positions at this move number are scored as final; 0 = 3*points
	// ConservativePass only searches Pass when the opponent just passed (so
	// passing ends the game and is scored exactly) or when every other legal
	// move would fill one of our own eyes. Games are then played out until
	// dead stones are captured, which makes area scoring exact, and an
	// untrained network cannot end games by passing at random.
	ConservativePass bool
	Seed             int64
}

// DefaultConfig returns tuned defaults for 9x9.
func DefaultConfig() Config {
	return Config{
		CPuct:            1.4,
		FPUReduction:     0.25,
		RootFPUReduction: 0.1,
		NoiseFrac:        0.25,
		Batch:            8,
		ConservativePass: true,
		Seed:             1,
	}
}

type node struct {
	move     game.Point
	prior    float32
	n        int32   // completed visits
	vl       int32   // virtual losses in flight
	w        float64 // value sum for the player who played move
	children []node  // allocated once at expansion; never appended to, so pointers stay valid
	expanded bool
	terminal bool
	pending  bool // awaiting evaluation in the current batch
}

// Search owns a tree for one game. It is not safe for concurrent use; run
// one Search per goroutine and share the evaluator.
type Search struct {
	cfg   Config
	eval  nn.Evaluator
	rng   *rand.Rand
	root  *node
	board *game.Board // position at the root

	// scratch
	leafBoards []*game.Board
	leafPaths  [][]*node
	outs       []nn.Output
	legal      []game.Point
}

// New returns a search over eval with the given configuration.
func New(cfg Config, eval nn.Evaluator) *Search {
	if cfg.Batch < 1 {
		cfg.Batch = 1
	}
	return &Search{cfg: cfg, eval: eval, rng: rand.New(rand.NewSource(cfg.Seed))}
}

// Config returns the active configuration.
func (s *Search) Config() Config { return s.cfg }

// SetRootNoise toggles Dirichlet noise for subsequent searches.
func (s *Search) SetRootNoise(on bool) { s.cfg.RootNoise = on }

// Reset discards the tree and sets a new root position.
func (s *Search) Reset(b *game.Board) {
	s.board = b.Clone()
	s.root = &node{move: game.Pass}
}

// Advance moves the root to the child reached by m (keeping its subtree) and
// syncs to board b, which must be the position after m. Falls back to Reset.
func (s *Search) Advance(m game.Point, b *game.Board) {
	if s.root != nil && s.root.expanded {
		for i := range s.root.children {
			c := &s.root.children[i]
			if c.move == m {
				s.root = c
				s.board = b.Clone()
				return
			}
		}
	}
	s.Reset(b)
}

// Board returns the root position (do not modify).
func (s *Search) Board() *game.Board { return s.board }

func (s *Search) maxMoves() int {
	if s.cfg.MaxMoves > 0 {
		return s.cfg.MaxMoves
	}
	return 3 * s.board.NumPoints()
}

// Run performs visits playouts from the root.
func (s *Search) Run(visits int) error {
	if s.root == nil {
		return errors.New("mcts: Run before Reset")
	}
	if s.isTerminal(s.board) {
		return nil
	}
	if !s.root.expanded {
		out := make([]nn.Output, 1)
		if err := s.eval.Eval([]*game.Board{s.board}, out); err != nil {
			return err
		}
		s.expand(s.root, s.board, out[0].Policy)
		s.backup([]*node{s.root}, float64(out[0].Value))
		visits--
	}
	if s.cfg.RootNoise {
		s.applyNoise()
	}
	for len(s.leafBoards) < s.cfg.Batch {
		s.leafBoards = append(s.leafBoards, s.board.Clone())
		s.leafPaths = append(s.leafPaths, nil)
	}
	if len(s.outs) < s.cfg.Batch {
		s.outs = make([]nn.Output, s.cfg.Batch)
	}
	for done := 0; done < visits; {
		k := 0
		for k < s.cfg.Batch && done+k < visits {
			b := s.leafBoards[k]
			path := s.descend(b, s.leafPaths[k][:0])
			s.leafPaths[k] = path
			leaf := path[len(path)-1]
			if leaf.terminal {
				s.backup(path, terminalValue(b))
				done++
				continue
			}
			if leaf.pending {
				s.revert(path)
				break
			}
			leaf.pending = true
			k++
		}
		if k == 0 {
			continue
		}
		if err := s.eval.Eval(s.leafBoards[:k], s.outs[:k]); err != nil {
			for i := 0; i < k; i++ {
				s.leafPaths[i][len(s.leafPaths[i])-1].pending = false
				s.revert(s.leafPaths[i])
			}
			return err
		}
		for i := 0; i < k; i++ {
			path := s.leafPaths[i]
			leaf := path[len(path)-1]
			leaf.pending = false
			s.expand(leaf, s.leafBoards[i], s.outs[i].Policy)
			s.backup(path, float64(s.outs[i].Value))
		}
		done += k
	}
	return nil
}

func (s *Search) isTerminal(b *game.Board) bool {
	return b.GameOver() || b.MoveNum() >= s.maxMoves()
}

// terminalValue scores a finished game for the side to move.
func terminalValue(b *game.Board) float64 {
	switch b.Winner() {
	case b.ToPlay():
		return 1
	case game.Empty:
		return 0
	}
	return -1
}

// descend walks from the root to a leaf, applying virtual loss, and leaves b
// at the leaf position.
func (s *Search) descend(b *game.Board, path []*node) []*node {
	b.CopyFrom(s.board)
	nd := s.root
	path = append(path, nd)
	for nd.expanded && !nd.terminal {
		nd = s.selectChild(nd, nd == s.root)
		b.PlayUnchecked(nd.move)
		nd.vl++
		path = append(path, nd)
		if !nd.expanded && s.isTerminal(b) {
			nd.terminal = true
			nd.expanded = true
		}
	}
	return path
}

func (s *Search) selectChild(nd *node, isRoot bool) *node {
	parentN := float64(nd.n + nd.vl)
	sqrtN := math.Sqrt(math.Max(parentN, 1))
	parentQ := 0.0
	if nd.n > 0 {
		parentQ = -nd.w / float64(nd.n) // nd.w is from the opponent's view
	}
	visitedPrior := 0.0
	for i := range nd.children {
		if c := &nd.children[i]; c.n+c.vl > 0 {
			visitedPrior += float64(c.prior)
		}
	}
	red := s.cfg.FPUReduction
	if isRoot {
		red = s.cfg.RootFPUReduction
	}
	fpu := parentQ - red*math.Sqrt(visitedPrior)
	best, bestScore := 0, math.Inf(-1)
	for i := range nd.children {
		c := &nd.children[i]
		nv := float64(c.n + c.vl)
		q := fpu
		if nv > 0 {
			q = (c.w - float64(c.vl)) / nv // in-flight visits count as losses
		}
		score := q + s.cfg.CPuct*float64(c.prior)*sqrtN/(1+nv)
		if score > bestScore {
			best, bestScore = i, score
		}
	}
	return &nd.children[best]
}

func (s *Search) expand(nd *node, b *game.Board, policy []float32) {
	if nd.expanded {
		return
	}
	s.legal = b.LegalMoves(s.legal[:0])
	if s.cfg.ConservativePass {
		s.legal = filterMoves(b, s.legal)
	}
	n := b.NumPoints()
	children := make([]node, len(s.legal))
	sum := float32(0)
	for i, m := range s.legal {
		idx := int(m)
		if m == game.Pass {
			idx = n
		}
		children[i] = node{move: m, prior: policy[idx]}
		sum += policy[idx]
	}
	for i := range children {
		if sum > 0 {
			children[i].prior /= sum
		} else {
			children[i].prior = 1 / float32(len(children))
		}
	}
	nd.children = children
	nd.expanded = true
}

// filterMoves applies ConservativePass: drop own-eye fills, and drop Pass
// unless the opponent just passed or nothing else is left.
func filterMoves(b *game.Board, legal []game.Point) []game.Point {
	last, ok := b.LastMove(0)
	oppPassed := ok && last == game.Pass
	out := legal[:0]
	for _, m := range legal {
		if m == game.Pass || b.FillsOwnEye(m) {
			continue
		}
		out = append(out, m)
	}
	if len(out) == 0 || oppPassed {
		out = append(out, game.Pass)
	}
	return out
}

func (s *Search) backup(path []*node, v float64) {
	x := -v
	for i := len(path) - 1; i >= 0; i-- {
		nd := path[i]
		nd.n++
		nd.w += x
		if i > 0 {
			nd.vl--
		}
		x = -x
	}
}

func (s *Search) revert(path []*node) {
	for _, nd := range path[1:] {
		nd.vl--
	}
}

func (s *Search) applyNoise() {
	ch := s.root.children
	if len(ch) < 2 {
		return
	}
	alpha := s.cfg.NoiseAlpha
	if alpha <= 0 {
		alpha = 10.83 / float64(len(ch))
	}
	noise := make([]float64, len(ch))
	sum := 0.0
	for i := range noise {
		noise[i] = gamma(alpha, s.rng)
		sum += noise[i]
	}
	if sum == 0 {
		return
	}
	f := s.cfg.NoiseFrac
	for i := range ch {
		ch[i].prior = float32((1-f)*float64(ch[i].prior) + f*noise[i]/sum)
	}
}

// gamma samples Gamma(shape, 1) (Marsaglia-Tsang; boosted for shape < 1).
func gamma(shape float64, rng *rand.Rand) float64 {
	if shape < 1 {
		return gamma(shape+1, rng) * math.Pow(rng.Float64(), 1/shape)
	}
	d := shape - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x || math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}

// Child summarizes one root move.
type Child struct {
	Move   game.Point
	Visits int
	Q      float64 // mean value for the root player
	Prior  float32
}

// Children returns root moves sorted by visits (then Q), best first.
func (s *Search) Children() []Child {
	if s.root == nil {
		return nil
	}
	out := make([]Child, 0, len(s.root.children))
	for i := range s.root.children {
		c := &s.root.children[i]
		q := 0.0
		if c.n > 0 {
			q = c.w / float64(c.n)
		}
		out = append(out, Child{Move: c.move, Visits: int(c.n), Q: q, Prior: c.prior})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Visits != out[j].Visits {
			return out[i].Visits > out[j].Visits
		}
		return out[i].Q > out[j].Q
	})
	return out
}

// RootValue is the search's value estimate for the side to move at the root.
func (s *Search) RootValue() float64 {
	if s.root == nil || s.root.n == 0 {
		return 0
	}
	return -s.root.w / float64(s.root.n)
}

// Visits returns the root visit count.
func (s *Search) Visits() int {
	if s.root == nil {
		return 0
	}
	return int(s.root.n)
}

// PassWins reports whether the opponent just passed and passing now would end
// the game as a win for the side to move under area scoring. Players use it to
// pass instead of filling their own territory once the game is decided; the
// network rarely proposes pass because self-play only allows it at the end.
func (s *Search) PassWins() bool {
	b := s.board
	if b == nil {
		return false
	}
	if m, ok := b.LastMove(0); !ok || m != game.Pass {
		return false
	}
	t := b.Clone()
	t.PlayUnchecked(game.Pass)
	return t.Winner() == b.ToPlay()
}

// BestMove returns the most visited root move (Pass if the game is over).
func (s *Search) BestMove() game.Point {
	ch := s.Children()
	if len(ch) == 0 {
		return game.Pass
	}
	return ch[0].Move
}

// SampleMove picks a root move with probability proportional to visits^(1/temp).
func (s *Search) SampleMove(temp float64) game.Point {
	ch := s.Children()
	if len(ch) == 0 {
		return game.Pass
	}
	if temp <= 0 {
		return ch[0].Move
	}
	weights := make([]float64, len(ch))
	sum := 0.0
	for i, c := range ch {
		if c.Visits > 0 {
			weights[i] = math.Pow(float64(c.Visits), 1/temp)
			sum += weights[i]
		}
	}
	if sum == 0 {
		return ch[0].Move
	}
	r := s.rng.Float64() * sum
	for i, w := range weights {
		r -= w
		if r <= 0 {
			return ch[i].Move
		}
	}
	return ch[0].Move
}

// PolicyTarget returns root visit counts normalized over points+pass.
func (s *Search) PolicyTarget() []float32 {
	n := s.board.NumPoints()
	pi := make([]float32, n+1)
	total := 0
	for i := range s.root.children {
		total += int(s.root.children[i].n)
	}
	if total == 0 {
		return pi
	}
	for i := range s.root.children {
		c := &s.root.children[i]
		idx := int(c.move)
		if c.move == game.Pass {
			idx = n
		}
		pi[idx] = float32(c.n) / float32(total)
	}
	return pi
}

// PV returns the principal variation (most visited line) up to maxLen moves.
func (s *Search) PV(maxLen int) []game.Point {
	return s.pvFrom(s.root, nil, maxLen)
}

// PVFrom returns the most visited line that starts with root move m.
func (s *Search) PVFrom(m game.Point, maxLen int) []game.Point {
	if s.root == nil {
		return nil
	}
	for i := range s.root.children {
		if c := &s.root.children[i]; c.move == m {
			return s.pvFrom(c, []game.Point{m}, maxLen)
		}
	}
	return nil
}

func (s *Search) pvFrom(nd *node, pv []game.Point, maxLen int) []game.Point {
	for nd != nil && nd.expanded && len(pv) < maxLen {
		var best *node
		for i := range nd.children {
			c := &nd.children[i]
			if c.n > 0 && (best == nil || c.n > best.n) {
				best = c
			}
		}
		if best == nil {
			break
		}
		pv = append(pv, best.move)
		nd = best
	}
	return pv
}
