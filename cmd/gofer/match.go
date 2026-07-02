package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"runtime/debug"
	"strings"
	"sync"
	"time"
)

// MatchConfig configures champion/challenger arena matches.
type MatchConfig struct {
	Games         int
	Size          int
	Komi          float64
	Playouts      int
	BlackPlayouts int // 0 = Playouts
	WhitePlayouts int // 0 = Playouts
	ThinkTime     time.Duration
	BlackEval     string
	WhiteEval     string
	Seed          int64
	SwapColors    bool
	ArenaEnhanced string // none, baseline, both — forced root playouts
	Parallel      int    // concurrent games sharing one evaluator per role
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
	WinsBlack          int           `json:"wins_black"`
	WinsWhite          int           `json:"wins_white"`
	Draws              int           `json:"draws"`
	WinsBaseline       int           `json:"wins_baseline"`
	WinsChallenger     int           `json:"wins_challenger"`
	WinRateBlack       float64       `json:"win_rate_black"`
	WinRateBaseline    float64       `json:"win_rate_baseline"`
	WinRateChallenger  float64       `json:"win_rate_challenger"`
	WilsonCILow        float64       `json:"wilson_ci_low"`
	WilsonCIHigh       float64       `json:"wilson_ci_high"`
	BaselineWilsonLow  float64       `json:"baseline_wilson_ci_low"`
	BaselineWilsonHigh float64       `json:"baseline_wilson_ci_high"`
	ConfigHash         string        `json:"config_hash"`
	Games              int           `json:"game_count"`
	BaselineEval       string        `json:"baseline_eval"`
	ChallengerEval     string        `json:"challenger_eval"`
	Promoted           bool          `json:"promoted"`
	GameSummaries      []GameSummary `json:"games,omitempty"`
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
	margin := z * math.Sqrt((p*(1-p)/float64(n) + z2/(4*float64(n)*float64(n)))) / denom
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
	fmt.Fprintf(h, "games=%d size=%d komi=%.2f playouts=%d bpl=%d wpl=%d think=%d black=%s white=%s seed=%d swap=%v enhanced=%s",
		cfg.Games, cfg.Size, cfg.Komi, cfg.Playouts, cfg.BlackPlayouts, cfg.WhitePlayouts,
		cfg.ThinkTime, cfg.BlackEval, cfg.WhiteEval, cfg.Seed, cfg.SwapColors, cfg.ArenaEnhanced)
	if info, ok := debug.ReadBuildInfo(); ok {
		fmt.Fprintf(h, " mod=%s", info.Main.Version)
	}
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:8])
}

func normalizeMatchConfig(cfg MatchConfig) MatchConfig {
	if cfg.Games <= 0 {
		cfg.Games = 1
	}
	if cfg.Playouts <= 0 && cfg.ThinkTime <= 0 {
		cfg.Playouts = defaultPlayoutsForSize(cfg.Size)
	}
	if !cfg.SwapColors {
		cfg.SwapColors = true
	}
	return cfg
}

// RunMatch plays cfg.Games between BlackEval (baseline) and WhiteEval (challenger).
// Games run concurrently (cfg.Parallel) over one shared evaluator per role so the
// inference sidecars receive real batches. Per-game seeds keep results deterministic.
func RunMatch(cfg MatchConfig) MatchResult {
	cfg = normalizeMatchConfig(cfg)
	r := Chinese()
	out := MatchResult{
		Games:          cfg.Games,
		ConfigHash:     matchConfigHash(cfg),
		BaselineEval:   cfg.BlackEval,
		ChallengerEval: cfg.WhiteEval,
	}

	evals, closeEvals := buildArenaEvaluators(cfg)
	defer closeEvals()

	par := cfg.Parallel
	if par < 1 {
		par = 1
	}
	if par > cfg.Games {
		par = cfg.Games
	}
	summaries := make([]GameSummary, cfg.Games)
	sem := make(chan struct{}, par)
	var wg sync.WaitGroup
	for g := 0; g < cfg.Games; g++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()
			summaries[idx] = runArenaSummary(r, cfg, idx, evals)
		}(g)
	}
	wg.Wait()

	baselineWins := 0
	challengerWins := 0
	for _, summary := range summaries {
		accumulateArenaSummary(&out, summary)
		baselineWins, challengerWins = countRoleWin(cfg, summary, baselineWins, challengerWins)
		out.GameSummaries = append(out.GameSummaries, summary)
	}

	out.WinsBaseline = baselineWins
	out.WinsChallenger = challengerWins
	out.WinRateBlack = float64(out.WinsBlack) / float64(cfg.Games)
	out.WinRateBaseline = float64(baselineWins) / float64(cfg.Games)
	out.WinRateChallenger = float64(challengerWins) / float64(cfg.Games)
	out.WilsonCILow, out.WilsonCIHigh = WilsonCI(challengerWins, cfg.Games, 1.96)
	out.BaselineWilsonLow, out.BaselineWilsonHigh = WilsonCI(baselineWins, cfg.Games, 1.96)

	harness := GatingHarness{Games: cfg.Games, MinWinRateMargin: PromoteMin}
	out.Promoted = harness.Pass(out.WinRateBaseline, out.WinRateChallenger)
	return out
}

func arenaEvalsForGame(cfg MatchConfig, gameIdx int) (string, string) {
	if cfg.SwapColors && gameIdx%2 == 1 {
		return cfg.WhiteEval, cfg.BlackEval
	}
	return cfg.BlackEval, cfg.WhiteEval
}

func arenaPlayouts(cfg MatchConfig) (int, int) {
	bp, wp := cfg.Playouts, cfg.Playouts
	if cfg.BlackPlayouts > 0 {
		bp = cfg.BlackPlayouts
	}
	if cfg.WhitePlayouts > 0 {
		wp = cfg.WhitePlayouts
	}
	return bp, wp
}

func runArenaSummary(r Ruleset, cfg MatchConfig, gameIdx int, evals map[string]Evaluator) GameSummary {
	blackEval, whiteEval := arenaEvalsForGame(cfg, gameIdx)
	bp, wp := arenaPlayouts(cfg)
	// Evaluators are shared across games; do not Close() the per-game engines.
	blackEng := newArenaEngineEval(r, bp, cfg.ThinkTime, evals[blackEval], cfg.Seed+int64(gameIdx)*2, arenaEnhancedFor(blackEval, cfg))
	whiteEng := newArenaEngineEval(r, wp, cfg.ThinkTime, evals[whiteEval], cfg.Seed+int64(gameIdx)*2+1, arenaEnhancedFor(whiteEval, cfg))

	b := NewBoard(cfg.Size, cfg.Komi)
	moves := playArenaGame(r, b, blackEng, whiteEng, cfg.Size)
	bl, wl := r.Score(b)
	summary := GameSummary{Game: gameIdx + 1, BlackEval: blackEval, WhiteEval: whiteEval, Moves: moves}
	switch {
	case bl > wl:
		summary.BlackWins = true
	case wl > bl:
		summary.WhiteWins = true
	default:
		summary.Draw = true
	}
	return summary
}

func accumulateArenaSummary(out *MatchResult, summary GameSummary) {
	switch {
	case summary.BlackWins:
		out.WinsBlack++
	case summary.WhiteWins:
		out.WinsWhite++
	default:
		out.Draws++
	}
}

func countRoleWin(cfg MatchConfig, summary GameSummary, baselineWins, challengerWins int) (int, int) {
	if summary.BlackWins && summary.BlackEval == cfg.BlackEval {
		baselineWins++
	} else if summary.WhiteWins && summary.WhiteEval == cfg.BlackEval {
		baselineWins++
	} else if summary.BlackWins || summary.WhiteWins {
		challengerWins++
	}
	return baselineWins, challengerWins
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

func arenaEnhancedFor(evalName string, cfg MatchConfig) bool {
	switch strings.ToLower(cfg.ArenaEnhanced) {
	case "both":
		return true
	case "baseline":
		return evalIsBaseline(evalName, cfg.BlackEval)
	default:
		return false
	}
}

// buildArenaEvaluators constructs one shared evaluator per distinct eval role.
// Stateless evaluators (heuristic/uniform) and the concurrency-safe batched ONNX
// evaluator are all safe to share across concurrent games.
func buildArenaEvaluators(cfg MatchConfig) (map[string]Evaluator, func()) {
	minBatch := cfg.Parallel
	if minBatch < 1 {
		minBatch = 1
	}
	evals := make(map[string]Evaluator)
	var closers []func()
	for _, name := range []string{cfg.BlackEval, cfg.WhiteEval} {
		if _, ok := evals[name]; ok {
			continue
		}
		ev := buildSharedEvaluator(name, minBatch)
		evals[name] = ev
		if c, ok := ev.(*BatchedEvaluator); ok {
			closers = append(closers, c.Close)
		}
	}
	return evals, func() {
		for _, c := range closers {
			c()
		}
	}
}

func buildSharedEvaluator(name string, minBatch int) Evaluator {
	switch {
	case strings.EqualFold(name, "onnx"), strings.EqualFold(name, "onnx-batch"):
		return newONNXEvaluator(minBatch)
	case strings.EqualFold(name, "onnx2"):
		return newONNXEvaluatorURL(evalConfig.ONNXURL2, minBatch)
	default:
		return parseEvaluator(name)
	}
}

func newArenaEngineEval(r Ruleset, playouts int, think time.Duration, ev Evaluator, seed int64, enhanced bool) *Engine {
	scfg := DefaultConfig()
	scfg.Playouts = playouts
	scfg.ThinkTime = think
	scfg.Seed = seed
	scfg.Workers = 1 // concurrency comes from parallel games feeding the shared batcher
	if enhanced {
		scfg.ForcedRootPlayouts = defaultForcedRoot
	}
	return NewEngine(r, ev, scfg)
}

func evalIsBaseline(evalName, baselineName string) bool {
	return evalName == baselineName
}
