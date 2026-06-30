package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"runtime/debug"
	"time"
)

// MatchConfig configures champion/challenger arena matches.
type MatchConfig struct {
	Games           int
	Size            int
	Komi            float64
	Playouts        int
	BlackPlayouts   int // 0 = Playouts
	WhitePlayouts   int // 0 = Playouts
	ThinkTime       time.Duration
	BlackEval       string
	WhiteEval       string
	Seed            int64
	SwapColors      bool
}

// GameSummary is one arena game outcome.
type GameSummary struct {
	Game      int    `json:"game"`
	BlackEval string `json:"black_eval"`
	WhiteEval string `json:"white_eval"`
	BlackWins bool   `json:"black_wins"`
	WhiteWins bool   `json:"white_wins"`
	Draw      bool   `json:"draw"`
	Moves     int    `json:"moves"`
}

// MatchResult is JSON output for arena runs.
type MatchResult struct {
	WinsBlack      int           `json:"wins_black"`
	WinsWhite      int           `json:"wins_white"`
	Draws          int           `json:"draws"`
	WinsBaseline   int           `json:"wins_baseline"`
	WinsChallenger int           `json:"wins_challenger"`
	WinRateBlack   float64       `json:"win_rate_black"`
	WinRateBaseline float64      `json:"win_rate_baseline"`
	WinRateChallenger float64    `json:"win_rate_challenger"`
	WilsonCILow    float64       `json:"wilson_ci_low"`
	WilsonCIHigh   float64       `json:"wilson_ci_high"`
	BaselineWilsonLow  float64   `json:"baseline_wilson_ci_low"`
	BaselineWilsonHigh float64   `json:"baseline_wilson_ci_high"`
	ConfigHash     string        `json:"config_hash"`
	Games          int           `json:"game_count"`
	BaselineEval   string        `json:"baseline_eval"`
	ChallengerEval string        `json:"challenger_eval"`
	Promoted       bool          `json:"promoted"`
	GameSummaries  []GameSummary `json:"games,omitempty"`
}

// WilsonCI returns Wilson score interval for binomial proportion (z=1.96 ~ 95%).
func WilsonCI(wins, n int, z float64) (low, high float64) {
	if n == 0 {
		return 0, 1
	}
	p := float64(wins) / float64(n)
	z2 := z * z
	denom := 1 + z2/float64(n)
	center := (p + z2/(2*float64(n))) / denom
	margin := z * math.Sqrt((p*(1-p)/float64(n)+z2/(4*float64(n)*float64(n)))) / denom
	low = center - margin
	high = center + margin
	if low < 0 {
		low = 0
	}
	if high > 1 {
		high = 1
	}
	return low, high
}

func matchConfigHash(cfg MatchConfig) string {
	h := sha256.New()
	fmt.Fprintf(h, "games=%d size=%d komi=%.2f playouts=%d bpl=%d wpl=%d think=%d black=%s white=%s seed=%d swap=%v",
		cfg.Games, cfg.Size, cfg.Komi, cfg.Playouts, cfg.BlackPlayouts, cfg.WhitePlayouts,
		cfg.ThinkTime, cfg.BlackEval, cfg.WhiteEval, cfg.Seed, cfg.SwapColors)
	if info, ok := debug.ReadBuildInfo(); ok {
		fmt.Fprintf(h, " mod=%s", info.Main.Version)
	}
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:8])
}

// RunMatch plays cfg.Games between BlackEval (baseline) and WhiteEval (challenger).
func RunMatch(cfg MatchConfig) MatchResult {
	if cfg.Games <= 0 {
		cfg.Games = 1
	}
	if cfg.Playouts <= 0 && cfg.ThinkTime <= 0 {
		cfg.Playouts = defaultPlayoutsForSize(cfg.Size)
	}
	if !cfg.SwapColors {
		cfg.SwapColors = true
	}
	r := Chinese()
	out := MatchResult{
		Games:          cfg.Games,
		ConfigHash:     matchConfigHash(cfg),
		BaselineEval:   cfg.BlackEval,
		ChallengerEval: cfg.WhiteEval,
	}
	baselineWins := 0
	challengerWins := 0

	for g := 0; g < cfg.Games; g++ {
		blackEval, whiteEval := cfg.BlackEval, cfg.WhiteEval
		if cfg.SwapColors && g%2 == 1 {
			blackEval, whiteEval = whiteEval, blackEval
		}
		bp, wp := cfg.Playouts, cfg.Playouts
		if cfg.BlackPlayouts > 0 {
			bp = cfg.BlackPlayouts
		}
		if cfg.WhitePlayouts > 0 {
			wp = cfg.WhitePlayouts
		}
		blackEng := newArenaEngine(r, bp, cfg.ThinkTime, blackEval, cfg.Seed+int64(g)*2, evalIsBaseline(blackEval, cfg.BlackEval))
		whiteEng := newArenaEngine(r, wp, cfg.ThinkTime, whiteEval, cfg.Seed+int64(g)*2+1, evalIsBaseline(whiteEval, cfg.BlackEval))

		b := NewBoard(cfg.Size, cfg.Komi)
		moves := playArenaGame(r, b, blackEng, whiteEng, cfg.Size)
		bl, wl := r.Score(b)

		summary := GameSummary{
			Game:      g + 1,
			BlackEval: blackEval,
			WhiteEval: whiteEval,
			Moves:     moves,
		}
		switch {
		case bl > wl:
			out.WinsBlack++
			summary.BlackWins = true
			if blackEval == cfg.BlackEval {
				baselineWins++
			} else {
				challengerWins++
			}
		case wl > bl:
			out.WinsWhite++
			summary.WhiteWins = true
			if whiteEval == cfg.BlackEval {
				baselineWins++
			} else {
				challengerWins++
			}
		default:
			out.Draws++
			summary.Draw = true
		}
		out.GameSummaries = append(out.GameSummaries, summary)
	}

	out.WinsBaseline = baselineWins
	out.WinsChallenger = challengerWins
	out.WinRateBlack = float64(out.WinsBlack) / float64(cfg.Games)
	out.WinRateBaseline = float64(baselineWins) / float64(cfg.Games)
	out.WinRateChallenger = float64(challengerWins) / float64(cfg.Games)
	out.WilsonCILow, out.WilsonCIHigh = WilsonCI(challengerWins, cfg.Games, 1.96)
	out.BaselineWilsonLow, out.BaselineWilsonHigh = WilsonCI(baselineWins, cfg.Games, 1.96)

	harness := GatingHarness{Games: cfg.Games, MinWinRateMargin: 0.55}
	out.Promoted = harness.Pass(out.WinRateBaseline, out.WinRateChallenger)
	return out
}

func playArenaGame(r Ruleset, b *Board, blackEng, whiteEng *Engine, size int) int {
	passes := 0
	moves := 0
	for moveNum := 0; moveNum < size*size+2; moveNum++ {
		if onlyPass(r.LegalMoves(b)) {
			break
		}
		eng := whiteEng
		if b.Player() == Black {
			eng = blackEng
		}
		m := eng.BestMove(b)
		if !r.Play(b, m) {
			m = PassMove
			r.Play(b, m)
		}
		eng.AdvanceTree(m)
		moves++
		if m.Pass {
			passes++
		} else {
			passes = 0
		}
		if passes >= 2 {
			break
		}
	}
	return moves
}

func newArenaEngine(r Ruleset, playouts int, think time.Duration, evalName string, seed int64, enhanced bool) *Engine {
	scfg := DefaultConfig()
	scfg.Playouts = playouts
	scfg.ThinkTime = think
	scfg.Seed = seed
	if enhanced {
		scfg.ForcedRootPlayouts = defaultForcedRoot
	}
	return NewEngine(r, parseEvaluator(evalName), scfg)
}

func evalIsBaseline(evalName, baselineName string) bool {
	return evalName == baselineName
}
