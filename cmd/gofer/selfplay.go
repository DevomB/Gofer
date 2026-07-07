package main

import (
	"math/rand"
	"strings"
	"sync"
)

// SelfplayConfig holds self-play parameters.
type SelfplayConfig struct {
	Games            int
	BoardSize        int
	Komi             float64
	Playouts         int
	FastPlayouts     int
	FullPlayouts     int
	CapRandomizeP    float64
	Seed             int64
	RulesRandomize   bool
	FullOnlyExport   bool    // export only full-search positions (paper training set)
	EvalMode         string  // heuristic, onnx, mix
	ONNXFraction     float64 // used by mix mode (odd/even when 0.5 default)
	Parallel         int     // concurrent games; >1 lets the batched evaluator fill real batches
	TemperatureMoves int     // opening plies sampled ~ visits (>0 = enabled) for game diversity
}

// DefaultSelfplayConfig returns reasonable defaults.
func DefaultSelfplayConfig() SelfplayConfig {
	return SelfplayConfig{
		Games:            1,
		BoardSize:        9,
		Komi:             6.5,
		Playouts:         30,
		FastPlayouts:     0,
		FullPlayouts:     0,
		CapRandomizeP:    0.25,
		Seed:             1,
		RulesRandomize:   false,
		FullOnlyExport:   true,
		EvalMode:         "heuristic",
		ONNXFraction:     0.7,
		Parallel:         1,
		TemperatureMoves: 0,
	}
}

// RunSelfplay plays games and returns training samples with visit-weighted policy.
func RunSelfplay(cfg SelfplayConfig) []Sample {
	samples, _ := RunSelfplayWithLogs(cfg)
	return samples
}

// RunSelfplayWithLogs returns samples and SGF-ready game logs.
// Games run concurrently (cfg.Parallel) over one shared batched evaluator so the
// inference sidecar sees real batches instead of one position at a time. Per-game
// RNG is derived from the seed alone, so output is deterministic regardless of Parallel.
func RunSelfplayWithLogs(cfg SelfplayConfig) ([]Sample, []*GameLog) {
	normalizeSelfplayPlayouts(&cfg)
	pool, closePool := newSelfplayEvalPool(cfg)
	defer closePool()

	par := cfg.Parallel
	if par < 1 {
		par = 1
	}
	if par > cfg.Games {
		par = cfg.Games
	}

	type gameResult struct {
		samples []Sample
		log     *GameLog
	}
	results := make([]gameResult, cfg.Games)
	sem := make(chan struct{}, par)
	var wg sync.WaitGroup
	for g := 0; g < cfg.Games; g++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()
			s, log := playSelfplayGameWithLog(cfg, idx, pool)
			results[idx] = gameResult{s, log}
		}(g)
	}
	wg.Wait()

	var samples []Sample
	logs := make([]*GameLog, 0, cfg.Games)
	for _, r := range results {
		samples = append(samples, r.samples...)
		logs = append(logs, r.log)
	}
	if cfg.FullOnlyExport {
		samples = FilterFullSearchSamples(samples)
	}
	return samples, logs
}

// evalPool holds the shared evaluators reused across concurrent self-play games.
type evalPool struct {
	onnx      Evaluator // shared batched ONNX evaluator, or nil when unused
	heuristic Evaluator
}

// newSelfplayEvalPool builds one shared evaluator set and a closer for its worker.
func newSelfplayEvalPool(cfg SelfplayConfig) (evalPool, func()) {
	pool := evalPool{heuristic: Heuristic{}}
	mode := strings.ToLower(cfg.EvalMode)
	needONNX := mode == "onnx" || (mode == "mix" && cfg.ONNXFraction > 0)
	if !needONNX {
		return pool, func() {}
	}
	minBatch := cfg.Parallel
	if minBatch < 1 {
		minBatch = 1
	}
	onnx := newONNXEvaluator(minBatch)
	pool.onnx = onnx
	return pool, func() {
		if c, ok := onnx.(*BatchedEvaluator); ok {
			c.Close()
		}
	}
}

func normalizeSelfplayPlayouts(cfg *SelfplayConfig) {
	if cfg.FullPlayouts <= 0 {
		cfg.FullPlayouts = cfg.Playouts
	}
	if cfg.FastPlayouts <= 0 {
		cfg.FastPlayouts = cfg.FullPlayouts / 4
		if cfg.FastPlayouts < 8 {
			cfg.FastPlayouts = 8
		}
	}
}

func playSelfplayGameWithLog(cfg SelfplayConfig, gameIdx int, pool evalPool) ([]Sample, *GameLog) {
	// Per-game RNG keyed only by gameIdx keeps output deterministic under Parallel.
	rng := rand.New(rand.NewSource(cfg.Seed + int64(gameIdx)))
	rs, size := selfplayRuleset(cfg, rng)
	b := NewBoard(size, cfg.Komi)
	log := NewGameLog(size, cfg.Komi)
	scfg := DefaultConfig()
	scfg.Seed = cfg.Seed + int64(gameIdx)
	scfg.Workers = 1 // concurrency comes from parallel games feeding the shared batcher
	eval := selfplayEvaluator(cfg, gameIdx, rng, pool)
	eng := NewEngine(rs, eval, scfg)
	// Do not eng.Close(): the ONNX evaluator is shared and closed once by the pool.
	game, _ := collectSelfplaySamples(cfg, rs, b, eng, size, log, rng)
	bl, wl := rs.Score(b)
	ownership := OwnershipLabel(b)
	labelGameSamples(game, bl, wl, ownership)
	return game, log
}

func selfplayEvaluator(cfg SelfplayConfig, gameIdx int, rng *rand.Rand, pool evalPool) Evaluator {
	_ = gameIdx
	switch strings.ToLower(cfg.EvalMode) {
	case "onnx":
		return pool.onnx
	case "mix":
		frac := cfg.ONNXFraction
		if frac < 0 {
			frac = 0.7
		}
		if pool.onnx != nil && (frac >= 1 || rng.Float64() < frac) {
			return pool.onnx
		}
		return pool.heuristic
	default:
		return pool.heuristic
	}
}

func selfplayRuleset(cfg SelfplayConfig, rng *rand.Rand) (Ruleset, int) {
	rs := Chinese()
	size := cfg.BoardSize
	if cfg.RulesRandomize {
		if rng.Float64() < 0.5 {
			rs = TrompTaylor()
		}
		if rng.Float64() < 0.25 {
			rs = WithSuperko(rs)
		}
		sizes := []int{9, 13, 19}
		size = sizes[rng.Intn(len(sizes))]
	}
	return rs, size
}

func selfplayMovePlayouts(cfg SelfplayConfig, rng *rand.Rand) (playouts int, fullSearch bool) {
	fullSearch = rng.Float64() < cfg.CapRandomizeP
	playouts = cfg.FastPlayouts
	if fullSearch {
		playouts = cfg.FullPlayouts
	}
	return playouts, fullSearch
}

func collectSelfplaySamples(cfg SelfplayConfig, rs Ruleset, b *Board, eng *Engine, size int, log *GameLog, rng *rand.Rand) ([]Sample, int) {
	var game []Sample
	var prevPolicy []float32
	passes := 0
	for moveNum := 0; moveNum < size*size+2; moveNum++ {
		moves := rs.LegalMoves(b)
		if onlyPass(moves) {
			break
		}
		playouts, fullSearch := selfplayMovePlayouts(cfg, rng)
		eng.ConfigureSelfplayMove(playouts, fullSearch)
		color := b.Player()
		// Sample from the visit distribution during the opening (paper SE: tau=1
		// for the first plies) so games diverge; play argmax afterwards. The policy
		// TARGET is always the tau=1 visit distribution regardless of what we play.
		temp := 0.0
		if moveNum < cfg.TemperatureMoves {
			temp = 1.0
		}
		m := eng.SelectMove(b, rng, temp)
		policy := eng.RootPolicyBoard(b, moves)
		spatial, globals := BuildFeaturesV2(b)
		game = append(game, Sample{
			BoardHash:       b.Hash(),
			MoveNum:         moveNum,
			Policy:          policy,
			PolicyOpp:       append([]float32(nil), prevPolicy...),
			FeaturesSpatial: spatial,
			FeaturesGlobal:  globals,
			ToPlay:          color,
			Komi:            cfg.Komi,
			FullSearch:      fullSearch,
		})
		prevPolicy = policy
		rs.Play(b, m)
		log.Record(color, m)
		eng.AdvanceTree(m)
		if m.Pass {
			passes++
		} else {
			passes = 0
		}
		if passes >= 2 {
			break
		}
	}
	return game, passes
}

func labelGameSamples(game []Sample, bl, wl float64, ownership []float32) {
	diff := bl - wl
	for i := range game {
		if game[i].ToPlay == Black {
			game[i].Value = outcomeValue(diff)
		} else {
			game[i].Value = outcomeValue(-diff)
		}
		if len(ownership) > 0 {
			game[i].Ownership = append([]float32(nil), ownership...)
		}
	}
}

func outcomeValue(diff float64) float32 {
	if diff > 0 {
		return 1
	}
	if diff < 0 {
		return -1
	}
	return 0
}

func onlyPass(moves []Move) bool {
	for _, m := range moves {
		if !m.Pass {
			return false
		}
	}
	return true
}

// FilterFullSearchSamples returns only positions from full-search moves (paper policy training set).
func FilterFullSearchSamples(samples []Sample) []Sample {
	out := make([]Sample, 0, len(samples))
	for _, s := range samples {
		if s.FullSearch {
			out = append(out, s)
		}
	}
	return out
}

func isJSONLPath(path string) bool {
	return strings.HasSuffix(strings.ToLower(path), ".jsonl")
}
