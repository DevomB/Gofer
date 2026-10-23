// Package selfplay generates training data: complete self-play games searched
// with MCTS, recorded as NumPy shards for the Python trainer.
package selfplay

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
	"github.com/DevomB/gofer/internal/sgf"
)

// Config controls a self-play run.
type Config struct {
	Size    int
	Komi    float64
	Superko bool

	Games    int
	Parallel int // concurrent games; each blocks on the shared evaluator, so this sets the batch size

	// Playout cap randomization (KataGo): a fraction FullProb of moves get a
	// full, noised search and become training rows; the rest get a cheap
	// search and are only played.
	FullVisits int
	FastVisits int
	FullProb   float64

	TempMoves   int     // opening plies sampled from the visit distribution
	Temperature float64 // sampling temperature for those plies

	ResignThreshold float64 // resign when the full-search root value falls below this (0 disables)
	NoResignProb    float64 // fraction of games that never resign (measures false resignations)

	Search mcts.Config
	Seed   int64

	OutDir     string // shard directory
	ShardGames int    // games per shard file
	Prefix     string // shard file prefix
	SGFDir     string // optional: write every game as SGF
	Log        func(format string, args ...any)
}

// Stats summarizes a run.
type Stats struct {
	Games, Moves, Rows  int
	BlackWins, Resigned int
	FalseResigns        int // no-resign games where the would-be resigner won
	NoResignWithTrigger int
	Elapsed             time.Duration
	Shards              []string
}

type row struct {
	spatial []uint8
	global  []float32
	policy  []float32
	q       float32
	toPlay  game.Color
}

type gameResult struct {
	rows      []row
	winner    game.Color
	margin    float64
	ownership []game.Color // nil when the game ended by resignation
	moves     []sgf.Move
	resigned  bool
	trigger   game.Color // side whose value crossed the resign threshold (no-resign games)
}

// Run plays cfg.Games games and writes shards to cfg.OutDir.
func Run(cfg Config, eval nn.Evaluator) (Stats, error) {
	if cfg.Parallel < 1 {
		cfg.Parallel = 1
	}
	if cfg.ShardGames < 1 {
		cfg.ShardGames = 250
	}
	if cfg.Temperature <= 0 {
		cfg.Temperature = 1
	}
	if cfg.Log == nil {
		cfg.Log = func(string, ...any) {}
	}
	if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
		return Stats{}, err
	}
	if cfg.SGFDir != "" {
		if err := os.MkdirAll(cfg.SGFDir, 0o755); err != nil {
			return Stats{}, err
		}
	}
	start := time.Now()
	var (
		next     atomic.Int64
		mu       sync.Mutex
		pending  []gameResult
		stats    Stats
		firstErr error
		shardNo  int
	)
	flush := func(force bool) error {
		if len(pending) == 0 || (!force && len(pending) < cfg.ShardGames) {
			return nil
		}
		path := filepath.Join(cfg.OutDir, fmt.Sprintf("%s-%04d.npz", cfg.Prefix, shardNo))
		shardNo++
		if err := writeShard(path, cfg.Size, pending); err != nil {
			return err
		}
		stats.Shards = append(stats.Shards, path)
		pending = pending[:0]
		return nil
	}

	var wg sync.WaitGroup
	for w := 0; w < cfg.Parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				g := int(next.Add(1)) - 1
				if g >= cfg.Games {
					return
				}
				res, err := playGame(cfg, eval, g)
				mu.Lock()
				if err != nil {
					if firstErr == nil {
						firstErr = err
					}
					mu.Unlock()
					return
				}
				stats.Games++
				stats.Moves += len(res.moves)
				stats.Rows += len(res.rows)
				if res.winner == game.Black {
					stats.BlackWins++
				}
				if res.resigned {
					stats.Resigned++
				}
				if res.trigger != game.Empty {
					stats.NoResignWithTrigger++
					if res.trigger == res.winner {
						stats.FalseResigns++
					}
				}
				if cfg.SGFDir != "" {
					writeSGF(cfg, g, res)
				}
				pending = append(pending, res)
				if err := flush(false); err != nil && firstErr == nil {
					firstErr = err
				}
				if stats.Games%50 == 0 {
					el := time.Since(start).Seconds()
					cfg.Log("selfplay: %d/%d games, %.1f games/s, %d rows, black wins %.1f%%",
						stats.Games, cfg.Games, float64(stats.Games)/el, stats.Rows, 100*float64(stats.BlackWins)/float64(stats.Games))
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return stats, firstErr
	}
	if err := flush(true); err != nil {
		return stats, err
	}
	stats.Elapsed = time.Since(start)
	return stats, nil
}

func playGame(cfg Config, eval nn.Evaluator, idx int) (gameResult, error) {
	rng := rand.New(rand.NewSource(cfg.Seed*1_000_003 + int64(idx)))
	b := game.NewBoard(game.Options{Size: cfg.Size, Komi: cfg.Komi, Superko: cfg.Superko})
	sc := cfg.Search
	sc.Seed = rng.Int63()
	s := mcts.New(sc, eval)
	s.Reset(b)
	allowResign := cfg.ResignThreshold != 0 && rng.Float64() >= cfg.NoResignProb
	maxMoves := sc.MaxMoves
	if maxMoves <= 0 {
		maxMoves = 3 * b.NumPoints()
	}
	n := b.NumPoints()
	var res gameResult
	for !b.GameOver() && b.MoveNum() < maxMoves {
		full := cfg.FastVisits <= 0 || rng.Float64() < cfg.FullProb
		visits := cfg.FastVisits
		if full {
			visits = cfg.FullVisits
		}
		s.SetRootNoise(full)
		if err := s.Run(visits); err != nil {
			return res, err
		}
		me := b.ToPlay()
		if full {
			r := row{
				spatial: make([]uint8, nn.NumPlanes*n),
				global:  make([]float32, nn.NumGlobals),
				policy:  s.PolicyTarget(),
				q:       float32(s.RootValue()),
				toPlay:  me,
			}
			nn.EncodeBytes(b, r.spatial, r.global)
			res.rows = append(res.rows, r)
			if cfg.ResignThreshold != 0 && s.RootValue() < cfg.ResignThreshold {
				if allowResign {
					res.resigned = true
					res.winner = me.Opp()
					break
				}
				if res.trigger == game.Empty {
					res.trigger = me
				}
			}
		}
		m := s.BestMove()
		if b.MoveNum() < cfg.TempMoves {
			m = s.SampleMove(cfg.Temperature)
		}
		if err := b.Play(m); err != nil {
			return res, fmt.Errorf("selfplay: search chose %w", err)
		}
		res.moves = append(res.moves, sgf.Move{Color: me, Point: m})
		s.Advance(m, b)
	}
	if !res.resigned {
		res.winner = b.Winner()
		res.margin = b.Margin()
		res.ownership = b.Ownership()
	}
	return res, nil
}

func writeShard(path string, size int, games []gameResult) error {
	n := size * size
	var rows int
	for _, g := range games {
		rows += len(g.rows)
	}
	spatial := make([]uint8, 0, rows*nn.NumPlanes*n)
	global := make([]float32, 0, rows*nn.NumGlobals)
	policy := make([]float32, 0, rows*(n+1))
	value := make([]float32, 0, rows)
	q := make([]float32, 0, rows)
	own := make([]int8, 0, rows*n)
	ownW := make([]float32, 0, rows)
	margin := make([]float32, 0, rows)
	for _, g := range games {
		for _, r := range g.rows {
			spatial = append(spatial, r.spatial...)
			global = append(global, r.global...)
			policy = append(policy, r.policy...)
			q = append(q, r.q)
			sign := float32(1)
			if r.toPlay == game.White {
				sign = -1
			}
			switch g.winner {
			case r.toPlay:
				value = append(value, 1)
			case game.Empty:
				value = append(value, 0)
			default:
				value = append(value, -1)
			}
			if g.ownership == nil {
				own = append(own, make([]int8, n)...)
				ownW = append(ownW, 0)
				margin = append(margin, 0)
				continue
			}
			for _, c := range g.ownership {
				switch c {
				case r.toPlay:
					own = append(own, 1)
				case r.toPlay.Opp():
					own = append(own, -1)
				default:
					own = append(own, 0)
				}
			}
			ownW = append(ownW, 1)
			margin = append(margin, sign*float32(g.margin))
		}
	}
	return writeNPZ(path, []npyArray{
		{"spatial", "|u1", []int{rows, nn.NumPlanes, size, size}, spatial},
		{"global", "<f4", []int{rows, nn.NumGlobals}, global},
		{"policy", "<f4", []int{rows, n + 1}, policy},
		{"value", "<f4", []int{rows}, value},
		{"q", "<f4", []int{rows}, q},
		{"ownership", "|i1", []int{rows, n}, own},
		{"ownership_weight", "<f4", []int{rows}, ownW},
		{"margin", "<f4", []int{rows}, margin},
	})
}

func writeSGF(cfg Config, idx int, res gameResult) {
	result := "0"
	switch {
	case res.resigned:
		result = fmt.Sprintf("%s+R", res.winner)
	case res.winner != game.Empty:
		m := res.margin
		if m < 0 {
			m = -m
		}
		result = fmt.Sprintf("%s+%g", res.winner, m)
	}
	rec := sgf.Record{Size: cfg.Size, Komi: cfg.Komi, Black: "gofer", White: "gofer", Result: result, Moves: res.moves}
	path := filepath.Join(cfg.SGFDir, fmt.Sprintf("%s-%05d.sgf", cfg.Prefix, idx))
	os.WriteFile(path, []byte(sgf.Write(rec)), 0o644)
}
