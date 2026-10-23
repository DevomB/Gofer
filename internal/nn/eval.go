package nn

import (
	"math"
	"math/rand"
	"sync"

	"github.com/DevomB/gofer/internal/game"
)

// Output is a network evaluation of one position.
type Output struct {
	Policy []float32 // move probabilities over n points + pass (index n), sums to 1
	Value  float32   // expected result in [-1, 1] for the side to move
}

// Evaluator scores positions. Implementations must be safe for concurrent use.
type Evaluator interface {
	// Eval evaluates every board; out has len(boards) entries.
	Eval(boards []*game.Board, out []Output) error
}

// Backend runs the network on an encoded batch. spatial is
// n*NumPlanes*points floats, global is n*NumGlobals. It returns policy
// logits (n*(points+1)) and values (n).
type Backend interface {
	Run(spatial, global []float32, n int) (logits, values []float32, err error)
	Close() error
}

// Softmax converts logits to probabilities in place.
func Softmax(x []float32) {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - max))
		x[i] = float32(e)
		sum += e
	}
	inv := float32(1 / sum)
	for i := range x {
		x[i] *= inv
	}
}

// Uniform is a network-free evaluator: flat policy, value 0.
type Uniform struct{}

func (Uniform) Eval(boards []*game.Board, out []Output) error {
	for i, b := range boards {
		p := make([]float32, b.NumPoints()+1)
		for j := range p {
			p[j] = 1 / float32(len(p))
		}
		out[i] = Output{Policy: p}
	}
	return nil
}

// Rollout is the classic network-free baseline: flat policy and a value
// from random playouts that never fill their own single-point eyes.
type Rollout struct {
	Playouts int
	mu       sync.Mutex
	rng      *rand.Rand
}

// NewRollout returns a rollout evaluator using k playouts per position.
func NewRollout(k int, seed int64) *Rollout {
	return &Rollout{Playouts: k, rng: rand.New(rand.NewSource(seed))}
}

func (r *Rollout) Eval(boards []*game.Board, out []Output) error {
	r.mu.Lock()
	seed := r.rng.Int63()
	r.mu.Unlock()
	rng := rand.New(rand.NewSource(seed))
	for i, b := range boards {
		Uniform{}.Eval(boards[i:i+1], out[i:i+1])
		me := b.ToPlay()
		var sum float32
		k := r.Playouts
		if k < 1 {
			k = 1
		}
		for j := 0; j < k; j++ {
			switch randomPlayout(b, rng) {
			case me:
				sum++
			case me.Opp():
				sum--
			}
		}
		out[i].Value = sum / float32(k)
	}
	return nil
}

func randomPlayout(start *game.Board, rng *rand.Rand) game.Color {
	b := start.Clone()
	cand := make([]game.Point, 0, b.NumPoints())
	limit := b.MoveNum() + 3*b.NumPoints()
	for !b.GameOver() && b.MoveNum() < limit {
		cand = cand[:0]
		for p := game.Point(0); int(p) < b.NumPoints(); p++ {
			if b.At(p) == game.Empty && !isOwnEye(b, p) && b.IsLegal(p) {
				cand = append(cand, p)
			}
		}
		if len(cand) == 0 {
			b.PlayUnchecked(game.Pass)
			continue
		}
		b.PlayUnchecked(cand[rng.Intn(len(cand))])
	}
	return b.Winner()
}

// isOwnEye reports a point whose neighbors are all the mover's stones, none in atari.
func isOwnEye(b *game.Board, p game.Point) bool {
	me := b.ToPlay()
	for _, q := range b.Neighbors(p) {
		if b.At(q) != me || b.Libs(q) == 1 {
			return false
		}
	}
	return true
}
