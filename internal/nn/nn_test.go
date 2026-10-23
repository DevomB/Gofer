package nn

import (
	"math"
	"sync"
	"testing"

	"github.com/DevomB/gofer/internal/game"
)

func TestSymmetriesArePermutations(t *testing.T) {
	for _, size := range []int{5, 9, 19} {
		for s := 0; s < 8; s++ {
			seen := map[game.Point]bool{}
			for _, q := range Symmetry(size, s) {
				seen[q] = true
			}
			if len(seen) != size*size {
				t.Fatalf("size %d sym %d is not a permutation", size, s)
			}
		}
	}
}

// fakeBackend puts logit 10 on every own stone, so decoded policy must peak
// on own stones in the original orientation whatever symmetry was used.
type fakeBackend struct {
	points int
	mu     sync.Mutex
	calls  int
}

func (f *fakeBackend) Run(spatial, global []float32, n int) ([]float32, []float32, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()
	per := NumPlanes * f.points
	logits := make([]float32, n*(f.points+1))
	values := make([]float32, n)
	for i := 0; i < n; i++ {
		for p := 0; p < f.points; p++ {
			logits[i*(f.points+1)+p] = 10 * spatial[i*per+PlaneOwn*f.points+p]
		}
		values[i] = global[i*NumGlobals+GlobalKomi]
	}
	return logits, values, nil
}

func (f *fakeBackend) Close() error { return nil }

func TestNetUndoesSymmetry(t *testing.T) {
	fb := &fakeBackend{points: 81}
	net, err := NewNet(NetConfig{Size: 9, Workers: 2, MaxBatch: 8, RandomSymmetry: true, Seed: 1},
		func() (Backend, error) { return fb, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer net.Close()
	b := game.NewBoard(game.Options{Size: 9, Komi: 7.5})
	b.Play(b.P(2, 3)) // black
	b.Play(b.P(6, 6)) // white
	b.Play(b.P(1, 7)) // black; white to move: own stone is (6,6)
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out := make([]Output, 4)
			if err := net.Eval([]*game.Board{b, b, b, b}, out); err != nil {
				t.Error(err)
				return
			}
			for _, o := range out {
				best := 0
				for p := range o.Policy {
					if o.Policy[p] > o.Policy[best] {
						best = p
					}
				}
				if best != int(b.P(6, 6)) {
					t.Errorf("policy peak at %d, want own stone %d", best, b.P(6, 6))
				}
				if math.Abs(float64(o.Value)-7.5/15) > 1e-6 {
					t.Errorf("value %v: komi should be +7.5/15 for white to move", o.Value)
				}
			}
		}()
	}
	wg.Wait()
}

func TestCacheHits(t *testing.T) {
	fb := &fakeBackend{points: 81}
	net, _ := NewNet(NetConfig{Size: 9, Workers: 1, CacheEntries: 1000}, func() (Backend, error) { return fb, nil })
	defer net.Close()
	b := game.NewBoard(game.Options{Size: 9})
	out := make([]Output, 1)
	for i := 0; i < 5; i++ {
		net.Eval([]*game.Board{b}, out)
	}
	if fb.calls != 1 || net.Stats.CacheHits.Load() != 4 {
		t.Fatalf("calls=%d hits=%d", fb.calls, net.Stats.CacheHits.Load())
	}
}

func TestEncodeLiberties(t *testing.T) {
	b := game.NewBoard(game.Options{Size: 9})
	b.Play(b.P(0, 0)) // black corner stone: 2 libs
	b.Play(b.P(1, 0)) // white: black now in atari
	sp := make([]float32, NumPlanes*81)
	gl := make([]float32, NumGlobals)
	Encode(b, 0, sp, gl) // black to move
	if sp[PlaneOwnLib1*81+0] != 1 || sp[PlaneOpp*81+1] != 1 || sp[(PlaneHistory)*81+1] != 1 || sp[(PlaneHistory+1)*81+0] != 1 {
		t.Fatal("liberty/history planes wrong")
	}
	if gl[GlobalKomi] != 0 {
		t.Fatal("zero komi expected")
	}
}

func TestRolloutPrefersWinningSide(t *testing.T) {
	b := game.NewBoard(game.Options{Size: 5, Komi: 0.5})
	// Black owns everything but a corner: black to move should evaluate positive.
	for _, p := range []game.Point{6, 7, 8, 11, 13, 16, 17, 18} {
		b.SetToPlay(game.Black)
		b.Play(p)
	}
	b.SetToPlay(game.Black)
	out := make([]Output, 1)
	NewRollout(64, 1).Eval([]*game.Board{b}, out)
	if out[0].Value < 0.5 {
		t.Fatalf("rollout value %v, expected black clearly winning", out[0].Value)
	}
}
