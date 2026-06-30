package main

import (
	"math/rand"
)

// SelfplayConfig holds self-play parameters.
type SelfplayConfig struct {
	Games          int
	BoardSize      int
	Komi           float64
	Playouts       int
	CapRandomizeP  float64
	Seed           int64
	RulesRandomize bool
}

// DefaultSelfplayConfig returns reasonable defaults.
func DefaultSelfplayConfig() SelfplayConfig {
	return SelfplayConfig{
		Games:          1,
		BoardSize:      9,
		Komi:           6.5,
		Playouts:       30,
		CapRandomizeP:  0.25,
		Seed:           1,
		RulesRandomize: false,
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
	var samples []Sample
	var logs []*GameLog
	for g := 0; g < cfg.Games; g++ {
		gameSamples, log := playSelfplayGameWithLog(cfg, g, rng)
		samples = append(samples, gameSamples...)
		logs = append(logs, log)
	}
	return samples, logs
}

func playSelfplayGameWithLog(cfg SelfplayConfig, gameIdx int, rng *rand.Rand) ([]Sample, *GameLog) {
	rs, size := selfplayRuleset(cfg, rng)
	b := NewBoard(size, cfg.Komi)
	log := NewGameLog(size, cfg.Komi)
	eng := NewEngine(rs, nil, selfplaySearchCfg(cfg, gameIdx, rng))
	game, _ := collectSelfplaySamples(cfg, rs, b, eng, size, log)
	bl, wl := rs.Score(b)
	labelGameSamples(game, bl, wl)
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

func selfplaySearchCfg(cfg SelfplayConfig, gameIdx int, rng *rand.Rand) SearchConfig {
	playouts := cfg.Playouts
	if rng.Float64() < cfg.CapRandomizeP {
		playouts = cfg.Playouts * 2
	}
	scfg := DefaultConfig()
	scfg.Playouts = playouts
	scfg.Seed = cfg.Seed + int64(gameIdx)
	return scfg
}

func collectSelfplaySamples(cfg SelfplayConfig, rs Ruleset, b *Board, eng *Engine, size int, log *GameLog) ([]Sample, int) {
	var game []Sample
	passes := 0
	for moveNum := 0; moveNum < size*size+2; moveNum++ {
		moves := rs.LegalMoves(b)
		if onlyPass(moves) {
			break
		}
		color := b.Player()
		m := eng.BestMove(b)
		game = append(game, Sample{
			BoardHash: b.Hash(),
			MoveNum:   moveNum,
			Policy:    eng.RootPolicy(moves),
			ToPlay:    color,
			Komi:      cfg.Komi,
		})
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

func labelGameSamples(game []Sample, bl, wl float64) {
	diff := bl - wl
	for i := range game {
		if game[i].ToPlay == Black {
			game[i].Value = outcomeValue(diff)
		} else {
			game[i].Value = outcomeValue(-diff)
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
