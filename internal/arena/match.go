package arena

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/sgf"
)

// Config controls a match between players A and B.
type Config struct {
	Size, MaxMoves int
	Komi           float64
	Superko        bool
	Games          int // rounded up to an even number: each opening is played with both colors
	Parallel       int
	OpeningMoves   int // random opening plies shared by each pair of games
	Seed           int64
	SGFDir         string
	SPRT           *SPRT // optional early stop for gating
	Log            func(format string, args ...any)
}

// SPRT is Wald's sequential probability ratio test on A's Elo advantage d,
// with H0: d <= Elo0 and H1: d >= Elo1.
//
// The unit of evidence is an opening pair: games 2k and 2k+1 share an opening
// with colors swapped. A pair one player wins twice is a sweep; a split pair
// says nothing about who is stronger and is skipped. If game results are
// logistic in strength plus any color and opening advantage, and the two games
// of a pair are independent given the opening, then
//
//	q = P(A sweeps | the pair is decisive) = 1 / (1 + 10^(-2d/400))
//
// for every opening and either color advantage, because those terms cancel in
// P(A sweeps) / P(B sweeps) = e^(2d). Decisive pairs are therefore independent
// Bernoulli(q) trials, and the test is Wald's SPRT of q(Elo0) against q(Elo1)
// over them, in pair order (so games finishing early, such as resignations,
// cannot bias it). Wald's inequality bounds the probability of accepting H1
// when d <= Elo0 by Alpha/(1-Beta), and of accepting H0 when d >= Elo1 by
// Beta/(1-Alpha), at any game limit: a match that ends undecided accepts
// neither.
type SPRT struct {
	Elo0, Elo1  float64
	Alpha, Beta float64
}

// Game is the outcome of one game.
type Game struct {
	Index    int     `json:"index"`
	AColor   string  `json:"a_color"`
	Winner   string  `json:"winner"`       // "A", "B" or "draw"
	Color    string  `json:"winner_color"` // "B", "W" or "." for a draw
	Margin   float64 `json:"margin"`       // black minus white, after dead-stone removal
	Moves    int     `json:"moves"`
	Reason   string  `json:"reason"` // score, resign, max_moves
	Duration float64 `json:"seconds"`
}

// Result summarizes a match from A's point of view.
type Result struct {
	NameA         string  `json:"a"`
	NameB         string  `json:"b"`
	Games         []Game  `json:"games"`
	Wins          int     `json:"wins"`
	Losses        int     `json:"losses"`
	Draws         int     `json:"draws"`
	Score         float64 `json:"score"` // (wins + draws/2) / games
	ScoreLow      float64 `json:"score_ci_low"`
	ScoreHigh     float64 `json:"score_ci_high"`
	Elo           float64 `json:"elo"`
	EloLow        float64 `json:"elo_ci_low"`
	EloHigh       float64 `json:"elo_ci_high"`
	WinsAsBlack   int     `json:"wins_as_black"`
	WinsAsWhite   int     `json:"wins_as_white"`
	BlackWinRate  float64 `json:"black_win_rate"`
	LLR           float64 `json:"llr,omitempty"`
	SPRTDecision  string  `json:"sprt,omitempty"`          // "H1" (A stronger), "H0", or "" (undecided)
	SPRTPairs     int     `json:"sprt_pairs,omitempty"`    // complete opening pairs the test used, in order
	SweepsA       int     `json:"sprt_sweeps_a,omitempty"` // pairs A won twice
	SweepsB       int     `json:"sprt_sweeps_b,omitempty"` // pairs B won twice
	ElapsedSecond float64 `json:"elapsed_seconds"`
}

// Run plays the match. newA/newB build a fresh player for each game.
func Run(cfg Config, newA, newB func() (Player, error)) (*Result, error) {
	if cfg.Games%2 == 1 {
		cfg.Games++
	}
	if cfg.Parallel < 1 {
		cfg.Parallel = 1
	}
	if cfg.MaxMoves <= 0 {
		cfg.MaxMoves = 3 * cfg.Size * cfg.Size
	}
	if cfg.Log == nil {
		cfg.Log = func(string, ...any) {}
	}
	if cfg.SGFDir != "" {
		os.MkdirAll(cfg.SGFDir, 0o755)
	}
	start := time.Now()
	res := &Result{}
	var (
		mu      sync.Mutex
		next    int
		stopped bool
		errOut  error
		wg      sync.WaitGroup
	)
	openings := make([][]game.Point, cfg.Games/2)
	rng := rand.New(rand.NewSource(cfg.Seed))
	for i := range openings {
		openings[i] = randomOpening(cfg, rng)
	}
	for w := 0; w < cfg.Parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				if next >= cfg.Games || stopped || errOut != nil {
					mu.Unlock()
					return
				}
				g := next
				next++
				mu.Unlock()

				// External engines occasionally crash (GNU Go 3.8 has known
				// assertion failures); replay the game with fresh players
				// before giving up on the match.
				var (
					out          Game
					nameA, nameB string
					err          error
				)
				for attempt := 0; attempt < 3; attempt++ {
					if out, nameA, nameB, err = playOne(cfg, g, openings[g/2], newA, newB); err == nil {
						break
					}
					cfg.Log("arena: game %d attempt %d failed: %v", g, attempt+1, err)
				}
				mu.Lock()
				if err != nil {
					errOut = err
					mu.Unlock()
					return
				}
				res.NameA, res.NameB = nameA, nameB
				res.Games = append(res.Games, out)
				res.summarize()
				if cfg.SPRT != nil && res.SPRTDecision == "" {
					// the decision is made once, at the first crossing; games
					// still in flight afterwards are reported but do not revisit it
					cfg.SPRT.test(res)
					if res.SPRTDecision != "" {
						stopped = true
					}
				}
				if len(res.Games)%10 == 0 || stopped {
					cfg.Log("arena: %d/%d games  %s %d-%d-%d vs %s  score %.3f  elo %+.0f [%+.0f, %+.0f]",
						len(res.Games), cfg.Games, nameA, res.Wins, res.Losses, res.Draws, nameB, res.Score, res.Elo, res.EloLow, res.EloHigh)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	res.ElapsedSecond = time.Since(start).Seconds()
	return res, errOut
}

// randomOpening plays a few random legal moves away from the first line.
func randomOpening(cfg Config, rng *rand.Rand) []game.Point {
	b := game.NewBoard(game.Options{Size: cfg.Size, Komi: cfg.Komi, Superko: cfg.Superko})
	var moves []game.Point
	for len(moves) < cfg.OpeningMoves {
		var cand []game.Point
		for p := game.Point(0); int(p) < b.NumPoints(); p++ {
			x, y := b.XY(p)
			if x > 0 && y > 0 && x < cfg.Size-1 && y < cfg.Size-1 && b.IsLegal(p) {
				cand = append(cand, p)
			}
		}
		if len(cand) == 0 {
			break
		}
		m := cand[rng.Intn(len(cand))]
		b.Play(m)
		moves = append(moves, m)
	}
	return moves
}

func playOne(cfg Config, idx int, opening []game.Point, newA, newB func() (Player, error)) (Game, string, string, error) {
	t0 := time.Now()
	a, err := newA()
	if err != nil {
		return Game{}, "", "", err
	}
	defer a.Close()
	bp, err := newB()
	if err != nil {
		return Game{}, "", "", err
	}
	defer bp.Close()
	aBlack := idx%2 == 0
	black, white := a, bp
	if !aBlack {
		black, white = bp, a
	}
	b := game.NewBoard(game.Options{Size: cfg.Size, Komi: cfg.Komi, Superko: cfg.Superko})
	for _, p := range []Player{black, white} {
		if err := p.NewGame(b.Clone()); err != nil {
			return Game{}, "", "", err
		}
	}
	var record []sgf.Move
	play := func(m game.Point) error {
		c := b.ToPlay()
		if err := b.Play(m); err != nil {
			return err
		}
		record = append(record, sgf.Move{Color: c, Point: m})
		for _, p := range []Player{black, white} {
			if err := p.Observe(c, m, b); err != nil {
				return err
			}
		}
		return nil
	}
	for _, m := range opening {
		if err := play(m); err != nil {
			return Game{}, "", "", err
		}
	}
	out := Game{Index: idx, AColor: "B", Reason: "score"}
	if !aBlack {
		out.AColor = "W"
	}
	var winner game.Color
	for {
		if b.GameOver() {
			break
		}
		if b.MoveNum() >= cfg.MaxMoves {
			out.Reason = "max_moves"
			break
		}
		mover := black
		if b.ToPlay() == game.White {
			mover = white
		}
		m, resign, err := mover.GenMove(b)
		if err != nil {
			return Game{}, "", "", err
		}
		if resign {
			out.Reason = "resign"
			winner = b.ToPlay().Opp()
			break
		}
		if err := play(m); err != nil {
			return Game{}, "", "", fmt.Errorf("arena: %s: %w", mover.Name(), err)
		}
	}
	if out.Reason != "resign" {
		final := b
		// The first player able to judge dead stones (e.g. GNU Go) adjudicates.
		for _, p := range []Player{black, white} {
			if j, ok := p.(DeadStoneJudge); ok {
				if dead, err := j.DeadStones(); err == nil {
					final = b.WithoutStones(dead)
					break
				}
			}
		}
		out.Margin = final.Margin()
		winner = final.Winner()
	}
	out.Color = winner.String()
	switch {
	case winner == game.Empty:
		out.Winner = "draw"
	case (winner == game.Black) == aBlack:
		out.Winner = "A"
	default:
		out.Winner = "B"
	}
	out.Moves = b.MoveNum()
	out.Duration = time.Since(t0).Seconds()
	if cfg.SGFDir != "" {
		result := "0"
		if winner != game.Empty {
			if out.Reason == "resign" {
				result = winner.String() + "+R"
			} else {
				result = fmt.Sprintf("%s+%g", winner, math.Abs(out.Margin))
			}
		}
		rec := sgf.Record{Size: cfg.Size, Komi: cfg.Komi, Black: black.Name(), White: white.Name(), Result: result, Moves: record}
		os.WriteFile(filepath.Join(cfg.SGFDir, fmt.Sprintf("game-%04d.sgf", idx)), []byte(sgf.Write(rec)), 0o644)
	}
	return out, a.Name(), bp.Name(), nil
}

func (r *Result) summarize() {
	r.Wins, r.Losses, r.Draws, r.WinsAsBlack, r.WinsAsWhite = 0, 0, 0, 0, 0
	blackWins := 0
	for _, g := range r.Games {
		switch g.Winner {
		case "A":
			r.Wins++
			if g.AColor == "B" {
				r.WinsAsBlack++
			} else {
				r.WinsAsWhite++
			}
		case "B":
			r.Losses++
		default:
			r.Draws++
		}
		if g.Color == "B" {
			blackWins++
		}
	}
	n := float64(len(r.Games))
	r.Score = (float64(r.Wins) + 0.5*float64(r.Draws)) / n
	r.ScoreLow, r.ScoreHigh = wilson(r.Score, n, 1.96)
	r.Elo, r.EloLow, r.EloHigh = elo(r.Score), elo(r.ScoreLow), elo(r.ScoreHigh)
	r.BlackWinRate = float64(blackWins) / n
}

// wilson returns the Wilson score interval for proportion p over n trials.
func wilson(p, n, z float64) (lo, hi float64) {
	if n == 0 {
		return 0, 1
	}
	den := 1 + z*z/n
	c := (p + z*z/(2*n)) / den
	h := z * math.Sqrt(p*(1-p)/n+z*z/(4*n*n)) / den
	return math.Max(0, c-h), math.Min(1, c+h)
}

// elo converts an expected score to an Elo difference (clamped at +-800).
func elo(s float64) float64 {
	s = math.Min(math.Max(s, 0.01), 0.99)
	return -400 * math.Log10(1/s-1)
}

// sweepProb is q(d): the probability that a decisive pair is A's sweep when A
// is d Elo stronger.
func sweepProb(d float64) float64 { return 1 / (1 + math.Pow(10, -2*d/400)) }

// test runs the SPRT over the complete opening pairs at the start of the
// match (pairs 0, 1, 2, ... while both games of each are in), stopping at the
// first pair whose evidence crosses a bound, and records the outcome in r.
func (t *SPRT) test(r *Result) {
	byIndex := make(map[int]Game, len(r.Games))
	for _, g := range r.Games {
		byIndex[g.Index] = g
	}
	q0, q1 := sweepProb(t.Elo0), sweepProb(t.Elo1)
	winA, winB := math.Log(q1/q0), math.Log((1-q1)/(1-q0))
	lower, upper := math.Log(t.Beta/(1-t.Alpha)), math.Log((1-t.Beta)/t.Alpha)
	r.LLR, r.SPRTPairs, r.SweepsA, r.SweepsB = 0, 0, 0, 0
	for k := 0; ; k++ {
		g0, ok0 := byIndex[2*k]
		g1, ok1 := byIndex[2*k+1]
		if !ok0 || !ok1 {
			return
		}
		r.SPRTPairs++
		switch {
		case g0.Winner == "A" && g1.Winner == "A":
			r.SweepsA++
			r.LLR += winA
		case g0.Winner == "B" && g1.Winner == "B":
			r.SweepsB++
			r.LLR += winB
		}
		switch {
		case r.LLR >= upper:
			r.SPRTDecision = "H1"
			return
		case r.LLR <= lower:
			r.SPRTDecision = "H0"
			return
		}
	}
}
