package main

import (
	"math/rand"
	"strings"
)

// SelfplayConfig holds self-play parameters.
type SelfplayConfig struct {
	Games          int
	BoardSize      int
	Komi           float64
	Playouts       int
	FastPlayouts   int
	FullPlayouts   int
	CapRandomizeP  float64
	Seed           int64
	RulesRandomize bool
	FullOnlyExport bool // export only full-search positions (paper training set)
}

// DefaultSelfplayConfig returns reasonable defaults.
func DefaultSelfplayConfig() SelfplayConfig {
	return SelfplayConfig{
		Games:          1,
		BoardSize:      9,
		Komi:           6.5,
		Playouts:       30,
		FastPlayouts:   0,
		FullPlayouts:   0,
		CapRandomizeP:  0.25,
		Seed:           1,
		RulesRandomize: false,
		FullOnlyExport: true,
	}
}

// RunSelfplay plays games and returns training samples with visit-weighted policy.
func RunSelfplay(cfg SelfplayConfig) []Sample {
	samples, _ := RunSelfplayWithLogs(cfg)
	return samples
}

// RunSelfplayWithLogs returns samples and SGF-ready game logs.
func RunSelfplayWithLogs(cfg SelfplayConfig) ([]Sample, []*GameLog) {
	rng := rand.New(rand.NewSource(cfg.Seed))
	normalizeSelfplayPlayouts(&cfg)
	var samples []Sample
	var logs []*GameLog
	for g := 0; g < cfg.Games; g++ {
		gameSamples, log := playSelfplayGameWithLog(cfg, g, rng)
		samples = append(samples, gameSamples...)
		logs = append(logs, log)
	}
	if cfg.FullOnlyExport {
		samples = FilterFullSearchSamples(samples)
	}
	return samples, logs
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

func playSelfplayGameWithLog(cfg SelfplayConfig, gameIdx int, rng *rand.Rand) ([]Sample, *GameLog) {
	rs, size := selfplayRuleset(cfg, rng)
	b := NewBoard(size, cfg.Komi)
	log := NewGameLog(size, cfg.Komi)
	scfg := DefaultConfig()
	scfg.Seed = cfg.Seed + int64(gameIdx)
	eng := NewEngine(rs, Heuristic{}, scfg)
	game, _ := collectSelfplaySamples(cfg, rs, b, eng, size, log, rng)
	bl, wl := rs.Score(b)
	ownership := OwnershipLabel(b)
	labelGameSamples(game, bl, wl, ownership)
	return game, log
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
		m := eng.BestMove(b)
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
			Komi:       cfg.Komi,
			FullSearch: fullSearch,
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
