package arena

import (
	"math"
	"os"
	"testing"

	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
)

func TestStrongerSearchWins(t *testing.T) {
	cfg := Config{Size: 5, Komi: 0.5, Games: 20, Parallel: 4, OpeningMoves: 1, Seed: 1}
	strong := func() (Player, error) {
		return &MCTSPlayer{Label: "rollout-200", Eval: nn.NewRollout(2, 1), Config: mcts.DefaultConfig(), Visits: 200}, nil
	}
	weak := func() (Player, error) {
		return &MCTSPlayer{Label: "uniform-2", Eval: nn.Uniform{}, Config: mcts.DefaultConfig(), Visits: 2}, nil
	}
	res, err := Run(cfg, strong, weak)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Games) != 20 || res.Score < 0.8 {
		t.Fatalf("expected the rollout player to dominate: %+v", res)
	}
	if res.ScoreLow > res.Score || res.ScoreHigh < res.Score || res.EloLow > res.Elo {
		t.Fatalf("bad confidence interval %+v", res)
	}
}

func TestSPRTStopsEarly(t *testing.T) {
	r := &Result{}
	for i := 0; i < 40; i++ {
		r.Games = append(r.Games, Game{Winner: "A", AColor: "B", Color: "B"})
	}
	r.Games = append(r.Games, Game{Winner: "B", AColor: "B", Color: "W"})
	r.summarize()
	_, d := (&SPRT{Elo0: 0, Elo1: 35, Alpha: 0.05, Beta: 0.05}).test(r)
	if d != "H1" {
		t.Fatalf("40-1 should accept H1, got %q", d)
	}
}

func TestWilson(t *testing.T) {
	lo, hi := wilson(0.5, 100, 1.96)
	if math.Abs(lo-0.404) > 0.002 || math.Abs(hi-0.596) > 0.002 {
		t.Fatalf("wilson(0.5,100) = %v %v", lo, hi)
	}
}

func TestGNUGo(t *testing.T) {
	if _, err := os.Stat("/usr/games/gnugo"); err != nil {
		t.Skip("gnugo not installed")
	}
	cfg := Config{Size: 9, Komi: 7.5, Games: 2, Parallel: 2, Seed: 1}
	gnugo := func() (Player, error) {
		return NewGTPPlayer("gnugo-l1", []string{"/usr/games/gnugo", "--mode", "gtp", "--level", "1"}), nil
	}
	weak := func() (Player, error) {
		return &MCTSPlayer{Label: "uniform-1", Eval: nn.Uniform{}, Config: mcts.DefaultConfig(), Visits: 1}, nil
	}
	res, err := Run(cfg, gnugo, weak)
	if err != nil {
		t.Fatal(err)
	}
	if res.Wins != 2 {
		t.Fatalf("gnugo should beat a 1-visit uniform player: %+v", res.Games)
	}
}
