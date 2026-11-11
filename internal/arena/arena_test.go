package arena

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/DevomB/gofer/internal/game"
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

func TestSPRTPassesOverSkippedGames(t *testing.T) {
	s := &SPRT{Elo0: 0, Elo1: 50, Alpha: 0.05, Beta: 0.05}
	full := pairs(repeat("AA", 60)...)
	s.test(full)
	if full.SPRTDecision != "H1" {
		t.Fatalf("60 sweeps should accept H1, got %q", full.SPRTDecision)
	}
	// The same match, except that the first game could not be played. Its
	// pair carries no evidence; the test must go on to the pairs after it
	// instead of waiting for a game that will never arrive.
	r := pairs(repeat("AA", 61)...)
	r.Games = r.Games[1:]
	r.Skipped = []int{0}
	s.test(r)
	if r.SPRTDecision != "H1" || r.SPRTPairs != full.SPRTPairs || r.SweepsA != full.SweepsA {
		t.Fatalf("skipped game: decision %q after %d pairs (%d sweeps), want H1 after %d (%d)",
			r.SPRTDecision, r.SPRTPairs, r.SweepsA, full.SPRTPairs, full.SweepsA)
	}
}

// illegalMover plays on an occupied point: a move the arena's rules reject,
// as an external engine that plays by other rules can.
type illegalMover struct{ MCTSPlayer }

func (p *illegalMover) GenMove(b *game.Board) (game.Point, bool, error) {
	for q := 0; q < b.NumPoints(); q++ {
		if b.At(game.Point(q)) != game.Empty {
			return game.Point(q), false, nil
		}
	}
	return p.MCTSPlayer.GenMove(b)
}

func TestUnplayableGameIsLeftOutOfTheResult(t *testing.T) {
	uniform := func(label string) MCTSPlayer {
		return MCTSPlayer{Label: label, Eval: nn.Uniform{}, Config: mcts.DefaultConfig(), Visits: 4}
	}
	a := func() (Player, error) { p := uniform("a"); return &p, nil }
	var logged []string
	cfg := Config{Size: 5, Komi: 0.5, Games: 10, Parallel: 1, OpeningMoves: 1, Seed: 3,
		Log: func(f string, args ...any) { logged = append(logged, fmt.Sprintf(f, args...)) }}

	// One game at a time, so the first three players made for B are the
	// three attempts at game 0.
	made := 0
	b := func() (Player, error) {
		made++
		if made <= 3 {
			return &illegalMover{uniform("b")}, nil
		}
		p := uniform("b")
		return &p, nil
	}
	res, err := Run(cfg, a, b)
	if err != nil {
		t.Fatalf("one unplayable game failed the match: %v", err)
	}
	if len(res.Games) != 9 || !reflect.DeepEqual(res.Skipped, []int{0}) || res.Wins+res.Losses+res.Draws != 9 {
		t.Fatalf("want 9 games counted and game 0 skipped, got %d games, skipped %v, %d-%d-%d",
			len(res.Games), res.Skipped, res.Wins, res.Losses, res.Draws)
	}
	if text := strings.Join(logged, "\n"); !strings.Contains(text, "game 0 left out") || !strings.Contains(text, "illegal move") {
		t.Fatalf("the log does not say which game was left out and why:\n%s", text)
	}

	// An engine that cannot play at all still fails the match.
	broken := func() (Player, error) { return &illegalMover{uniform("b")}, nil }
	if _, err := Run(cfg, a, broken); err == nil || !strings.Contains(err.Error(), "could not be played") {
		t.Fatalf("a match in which no game can be played returned %v", err)
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
		return NewGTPPlayer("gnugo-l1", []string{"/usr/games/gnugo", "--mode", "gtp", "--level", "1", "--chinese-rules", "--positional-superko"}), nil
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
