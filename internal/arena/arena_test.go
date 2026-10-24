package arena

import (
	"math"
	"math/rand"
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

// pairs builds a match from opening-pair outcomes: "AA" A swept, "BB" B
// swept, "AB"/"BA" split.
func pairs(outcomes ...string) *Result {
	r := &Result{}
	for k, o := range outcomes {
		for j := 0; j < 2; j++ {
			r.Games = append(r.Games, Game{Index: 2*k + j, Winner: string(o[j])})
		}
	}
	return r
}

func repeat(o string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = o
	}
	return out
}

func TestSPRTNeedsRealEvidence(t *testing.T) {
	s := &SPRT{Elo0: 0, Elo1: 50, Alpha: 0.05, Beta: 0.05}
	// The previous implementation accepted H1 here: a 2-0 start gave zero
	// variance, and the floor it used made the LLR about 66.
	r := pairs("AA")
	s.test(r)
	if r.SPRTDecision != "" {
		t.Fatalf("one sweep decided %q (llr %.2f)", r.SPRTDecision, r.LLR)
	}
	// split pairs carry no evidence
	r = pairs(repeat("AB", 50)...)
	s.test(r)
	if r.SPRTDecision != "" || r.LLR != 0 || r.SPRTPairs != 50 {
		t.Fatalf("split pairs moved the test: %+v", r)
	}
	// ln(19) / ln(q1/q0) = 2.944 / 0.247: twelve straight sweeps are needed
	r = pairs(repeat("AA", 11)...)
	if s.test(r); r.SPRTDecision != "" {
		t.Fatalf("11 sweeps accepted %q", r.SPRTDecision)
	}
	r = pairs(repeat("AA", 12)...)
	if s.test(r); r.SPRTDecision != "H1" || r.SweepsA != 12 {
		t.Fatalf("12 sweeps: %+v", r)
	}
	r = pairs(repeat("BB", 9)...)
	if s.test(r); r.SPRTDecision != "H0" {
		t.Fatalf("9 losses: %+v", r)
	}
}

func TestSPRTUsesPairsInOrder(t *testing.T) {
	s := &SPRT{Elo0: 0, Elo1: 50, Alpha: 0.05, Beta: 0.05}
	r := pairs(repeat("AA", 20)...)
	// the first pair is missing a game: nothing after it may be counted
	r.Games = r.Games[1:]
	if s.test(r); r.SPRTDecision != "" || r.SPRTPairs != 0 {
		t.Fatalf("counted pairs past an incomplete one: %+v", r)
	}
	// completion order does not matter
	r = pairs(append(repeat("AA", 8), repeat("BB", 8)...)...)
	want := *r
	s.test(&want)
	rand.New(rand.NewSource(1)).Shuffle(len(r.Games), func(i, j int) { r.Games[i], r.Games[j] = r.Games[j], r.Games[i] })
	if s.test(r); r.SPRTDecision != want.SPRTDecision || r.LLR != want.LLR {
		t.Fatalf("shuffled games changed the test: %+v vs %+v", r, want)
	}
}

// simulate plays a match of up to maxPairs opening pairs between players d Elo
// apart, with a color advantage and a random advantage per opening. test scans
// pairs in order and stops at the first crossing, so one call on the whole
// match decides exactly as testing after every pair (as Run does) would.
func simulate(rng *rand.Rand, s *SPRT, d float64, maxPairs int) string {
	sig := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	dn := d * math.Ln10 / 400
	const color = -0.2 // Black wins about 45% between equals
	r := &Result{Games: make([]Game, 0, 2*maxPairs)}
	for k := 0; k < maxPairs; k++ {
		opening := rng.NormFloat64() * 0.5
		for j, adv := range []float64{color + opening, -color - opening} { // A as Black, then as White
			w := "B"
			if rng.Float64() < sig(dn+adv) {
				w = "A"
			}
			r.Games = append(r.Games, Game{Index: 2*k + j, Winner: w})
		}
	}
	s.test(r)
	return r.SPRTDecision
}

// TestSPRTErrorRates checks Wald's bounds by simulation: with alpha = beta =
// 0.05, P(accept H1 | d <= Elo0) and P(accept H0 | d >= Elo1) must stay under
// alpha/(1-beta) = 0.0526, despite the color and opening advantages.
func TestSPRTErrorRates(t *testing.T) {
	if testing.Short() {
		t.Skip("simulation")
	}
	s := &SPRT{Elo0: 0, Elo1: 50, Alpha: 0.05, Beta: 0.05}
	rng := rand.New(rand.NewSource(7))
	const n = 4000
	rate := func(d float64, maxPairs int, want string) float64 {
		hits := 0
		for i := 0; i < n; i++ {
			if simulate(rng, s, d, maxPairs) == want {
				hits++
			}
		}
		return float64(hits) / n
	}
	bound := s.Alpha/(1-s.Beta) + 3*math.Sqrt(0.0526*0.9474/n) // 3 standard errors of slack
	for _, c := range []struct {
		d        float64
		maxPairs int
		want     string
		max      float64
	}{
		{0, 2000, "H1", bound},   // equal strength: false promotion
		{-50, 2000, "H1", bound}, // weaker challenger
		{0, 50, "H1", bound},     // truncated at 100 games
		{50, 2000, "H0", bound},  // 50 Elo stronger: false rejection
	} {
		got := rate(c.d, c.maxPairs, c.want)
		t.Logf("d=%+.0f, up to %d games: P(%s) = %.4f (bound %.4f)", c.d, 2*c.maxPairs, c.want, got, c.max)
		if got > c.max {
			t.Errorf("d=%+.0f: P(%s) = %.4f exceeds %.4f", c.d, c.want, got, c.max)
		}
	}
	if p := rate(150, 100, "H1"); p < 0.95 {
		t.Errorf("a 150 Elo stronger challenger should be promoted within 200 games: P = %.3f", p)
	} else {
		t.Logf("d=+150, up to 200 games: P(H1) = %.4f", p)
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
