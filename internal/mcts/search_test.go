package mcts

import (
	"math"
	"testing"

	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/nn"
)

// material is a toy evaluator: flat policy, value = tanh(stone difference / 4)
// for the side to move. It is enough for search to find captures.
type material struct{}

func (material) Eval(boards []*game.Board, out []nn.Output) error {
	for i, b := range boards {
		nn.Uniform{}.Eval(boards[i:i+1], out[i:i+1])
		d := 0
		for p := game.Point(0); int(p) < b.NumPoints(); p++ {
			switch b.At(p) {
			case b.ToPlay():
				d++
			case b.ToPlay().Opp():
				d--
			}
		}
		out[i].Value = float32(math.Tanh(float64(d) / 4))
	}
	return nil
}

func place(b *game.Board, c game.Color, pts ...game.Point) {
	for _, p := range pts {
		b.SetToPlay(c)
		if err := b.Play(p); err != nil {
			panic(err)
		}
	}
}

func captureBoard() (*game.Board, game.Point) {
	b := game.NewBoard(game.Options{Size: 9, Komi: 0.5})
	row := func(y int, xs ...int) []game.Point {
		var out []game.Point
		for _, x := range xs {
			out = append(out, b.P(x, y))
		}
		return out
	}
	place(b, game.White, row(4, 1, 2, 3, 4, 5, 6)...)
	place(b, game.Black, row(3, 1, 2, 3, 4, 5, 6)...)
	place(b, game.Black, row(5, 1, 2, 3, 4, 5, 6)...)
	place(b, game.Black, b.P(0, 4))
	b.SetToPlay(game.Black)
	return b, b.P(7, 4)
}

func TestFindsCapture(t *testing.T) {
	for _, batch := range []int{1, 8} {
		b, capture := captureBoard()
		cfg := DefaultConfig()
		cfg.Batch = batch
		s := New(cfg, material{})
		s.Reset(b)
		if err := s.Run(400); err != nil {
			t.Fatal(err)
		}
		if got := s.BestMove(); got != capture {
			t.Fatalf("batch %d: best move %s, want capture %s; children %+v",
				batch, b.FormatPoint(got), b.FormatPoint(capture), s.Children()[:3])
		}
		if s.RootValue() < 0.5 {
			t.Fatalf("root value %.2f should be clearly winning for black", s.RootValue())
		}
	}
}

func TestPassesToWinAfterOpponentPass(t *testing.T) {
	b := game.NewBoard(game.Options{Size: 5, Komi: 0.5})
	// Black wall on column B, white on D: black area 10, white 10 + 0.5 komi... give
	// black one extra stone so black leads by area.
	place(b, game.Black, b.P(1, 0), b.P(1, 1), b.P(1, 2), b.P(1, 3), b.P(1, 4), b.P(2, 2))
	place(b, game.White, b.P(3, 0), b.P(3, 1), b.P(3, 2), b.P(3, 3), b.P(3, 4))
	b.SetToPlay(game.White)
	b.Play(game.Pass)
	if b.Margin() <= 0 {
		t.Fatalf("setup should have black ahead, margin %v", b.Margin())
	}
	s := New(DefaultConfig(), nn.Uniform{})
	s.Reset(b)
	s.Run(300)
	if got := s.BestMove(); got != game.Pass {
		t.Fatalf("black should pass to end a won game, played %s", b.FormatPoint(got))
	}
}

func TestDoesNotPassWhenLosing(t *testing.T) {
	b := game.NewBoard(game.Options{Size: 5, Komi: 7.5})
	b.Play(game.Pass) // black passes; white passing now would win for white
	b.SetToPlay(game.Black)
	b.Play(game.Pass)
	if !b.GameOver() {
		t.Fatal("two passes should end the game")
	}
	b2 := game.NewBoard(game.Options{Size: 5, Komi: 7.5})
	b2.SetToPlay(game.White)
	b2.Play(game.Pass) // white passed; black to move is losing on komi if it passes
	s := New(DefaultConfig(), nn.Uniform{})
	s.Reset(b2)
	s.Run(300)
	for _, c := range s.Children() {
		if c.Move == game.Pass && c.Q > -0.99 {
			t.Fatalf("pass should be a certain loss for black, Q=%v", c.Q)
		}
	}
	if s.BestMove() == game.Pass {
		t.Fatal("black must not pass into a lost game")
	}
}

func TestVisitAccountingAndReuse(t *testing.T) {
	b, _ := captureBoard()
	cfg := DefaultConfig()
	cfg.Batch = 8
	s := New(cfg, material{})
	s.Reset(b)
	s.Run(200)
	if s.Visits() != 200 {
		t.Fatalf("root visits %d want 200", s.Visits())
	}
	sum := 0
	for _, c := range s.Children() {
		sum += c.Visits
	}
	if sum != 199 {
		t.Fatalf("child visits %d want 199 (root expansion is one visit)", sum)
	}
	m := s.BestMove()
	kept := s.Children()[0].Visits
	nb := b.Clone()
	nb.Play(m)
	s.Advance(m, nb)
	if s.Visits() != kept {
		t.Fatalf("reused root has %d visits, want %d", s.Visits(), kept)
	}
	s.Run(50)
	if s.Visits() != kept+50 {
		t.Fatalf("visits after reuse %d want %d", s.Visits(), kept+50)
	}
	pi := s.PolicyTarget()
	total := float32(0)
	for _, p := range pi {
		total += p
	}
	if math.Abs(float64(total-1)) > 1e-4 {
		t.Fatalf("policy target sums to %v", total)
	}
}

func TestRootNoiseKeepsPriorsNormalized(t *testing.T) {
	b := game.NewBoard(game.Options{Size: 9, Komi: 7})
	cfg := DefaultConfig()
	cfg.RootNoise = true
	s := New(cfg, nn.Uniform{})
	s.Reset(b)
	s.Run(10)
	sum := float32(0)
	for _, c := range s.Children() {
		sum += c.Prior
	}
	if math.Abs(float64(sum-1)) > 1e-4 {
		t.Fatalf("noisy priors sum to %v", sum)
	}
}

func BenchmarkSearch200(b *testing.B) {
	bd := game.NewBoard(game.Options{Size: 9, Komi: 7})
	cfg := DefaultConfig()
	for i := 0; i < b.N; i++ {
		s := New(cfg, nn.Uniform{})
		s.Reset(bd)
		s.Run(200)
	}
}

func TestConservativePass(t *testing.T) {
	b := game.NewBoard(game.Options{Size: 9, Komi: 7.5})
	s := New(DefaultConfig(), nn.Uniform{})
	s.Reset(b)
	s.Run(50)
	for _, c := range s.Children() {
		if c.Move == game.Pass {
			t.Fatal("pass must not be searched on an open board")
		}
	}
	b.Play(game.Pass)
	s.Reset(b)
	s.Run(50)
	found := false
	for _, c := range s.Children() {
		found = found || c.Move == game.Pass
	}
	if !found {
		t.Fatal("pass must be searchable after the opponent passed")
	}
}

func TestPassWins(t *testing.T) {
	b := game.NewBoard(game.Options{Size: 5, Komi: 0.5})
	place(b, game.Black, b.P(2, 2))
	b.SetToPlay(game.White)
	b.Play(game.Pass) // white passes; black owns the board
	s := New(DefaultConfig(), nn.Uniform{})
	s.Reset(b)
	if !s.PassWins() {
		t.Fatal("black should see that passing wins")
	}
	b2 := game.NewBoard(game.Options{Size: 5, Komi: 7.5})
	b2.Play(game.Pass) // black passes on an empty board; white wins on komi
	s.Reset(b2)
	if !s.PassWins() {
		t.Fatal("white should see that passing wins on komi")
	}
	b3 := game.NewBoard(game.Options{Size: 5, Komi: 7.5})
	b3.SetToPlay(game.White)
	b3.Play(game.Pass) // black to move would lose by passing
	s.Reset(b3)
	if s.PassWins() {
		t.Fatal("passing loses for black here")
	}
}
