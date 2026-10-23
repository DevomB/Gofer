package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/DevomB/gofer/internal/arena"
	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/gtp"
	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
	"github.com/DevomB/gofer/internal/selfplay"
)

func cmdGTP(args []string) error {
	fs := flag.NewFlagSet("gtp", flag.ExitOnError)
	var nf netFlags
	var sf searchFlags
	nf.register(fs)
	sf.register(fs, 800)
	size := fs.Int("size", 9, "board size the model was trained for")
	resign := fs.Float64("resign", -0.95, "resign below this root value (0 disables)")
	verbose := fs.Bool("v", false, "log search info to stderr")
	fs.Parse(args)
	eval, closeEval, err := nf.evaluator(*size, 1)
	if err != nil {
		return err
	}
	defer closeEval()
	e := gtp.Engine{Name: "Gofer", Version: version, Eval: eval, Search: sf.config(1), Visits: sf.visits, Superko: true, ResignBelow: *resign}
	if *verbose {
		e.Log = os.Stderr
	}
	return gtp.NewEngine(e).Serve(os.Stdin, os.Stdout)
}

func cmdPlay(args []string) error {
	fs := flag.NewFlagSet("play", flag.ExitOnError)
	var nf netFlags
	var sf searchFlags
	nf.register(fs)
	sf.register(fs, 400)
	size := fs.Int("size", 9, "board size")
	komi := fs.Float64("komi", 7.5, "komi")
	color := fs.String("color", "b", "your color: b or w")
	fs.Parse(args)
	eval, closeEval, err := nf.evaluator(*size, 1)
	if err != nil {
		return err
	}
	defer closeEval()
	human := game.Black
	if strings.HasPrefix(strings.ToLower(*color), "w") {
		human = game.White
	}
	b := game.NewBoard(game.Options{Size: *size, Komi: *komi, Superko: true})
	s := mcts.New(sf.config(time.Now().UnixNano()), eval)
	s.Reset(b)
	in := bufio.NewScanner(os.Stdin)
	fmt.Println("Moves: D4, pass, resign, hint, quit")
	for !b.GameOver() {
		fmt.Printf("\n%s\n", b)
		var m game.Point
		if b.ToPlay() == human {
			fmt.Printf("%s to move> ", b.ToPlay())
			if !in.Scan() {
				return nil
			}
			cmd := strings.TrimSpace(in.Text())
			switch strings.ToLower(cmd) {
			case "quit", "resign":
				fmt.Println("You resigned.")
				return nil
			case "hint":
				s.Run(sf.visits)
				fmt.Printf("hint: %s (win %.0f%%)\n", b.FormatPoint(s.BestMove()), 50*(1+s.RootValue()))
				continue
			}
			p, err := game.ParsePoint(cmd, *size)
			if err != nil || !b.IsLegal(p) {
				fmt.Println("illegal move")
				continue
			}
			m = p
		} else {
			if err := s.Run(sf.visits); err != nil {
				return err
			}
			m = s.BestMove()
			fmt.Printf("Gofer plays %s (win %.0f%%, pv %s)\n", b.FormatPoint(m), 50*(1+s.RootValue()), pvString(s.PV(6), *size))
		}
		b.Play(m)
		s.Advance(m, b)
	}
	fmt.Printf("\n%s\nGame over: %s\n", b, scoreString(b.Margin()))
	return nil
}

func pvString(pv []game.Point, size int) string {
	parts := make([]string, len(pv))
	for i, p := range pv {
		parts[i] = game.FormatPoint(p, size)
	}
	return strings.Join(parts, " ")
}

func scoreString(m float64) string {
	switch {
	case m > 0:
		return fmt.Sprintf("B+%g", m)
	case m < 0:
		return fmt.Sprintf("W+%g", -m)
	}
	return "draw"
}

func cmdSelfplay(args []string) error {
	fs := flag.NewFlagSet("selfplay", flag.ExitOnError)
	var nf netFlags
	var sf searchFlags
	nf.register(fs)
	sf.register(fs, 0)
	var c selfplay.Config
	fs.IntVar(&c.Size, "size", 9, "board size")
	fs.Float64Var(&c.Komi, "komi", 7.5, "komi")
	fs.IntVar(&c.Games, "games", 100, "games to play")
	fs.IntVar(&c.Parallel, "parallel", 0, "concurrent games (0: 8 per inference worker)")
	fs.IntVar(&c.FullVisits, "full-visits", 200, "visits for recorded (full) searches")
	fs.IntVar(&c.FastVisits, "fast-visits", 40, "visits for cheap searches (0: every move is full)")
	fs.Float64Var(&c.FullProb, "full-prob", 0.25, "probability a move gets a full search")
	fs.IntVar(&c.TempMoves, "temp-moves", 10, "opening plies sampled from visit counts")
	fs.Float64Var(&c.ResignThreshold, "resign", -0.9, "resign below this value (0 disables)")
	fs.Float64Var(&c.NoResignProb, "no-resign-prob", 0.1, "fraction of games played out to measure false resignations")
	fs.Int64Var(&c.Seed, "seed", 1, "random seed")
	fs.StringVar(&c.OutDir, "out", "selfplay", "output directory for .npz shards")
	fs.StringVar(&c.Prefix, "prefix", "shard", "shard file prefix")
	fs.IntVar(&c.ShardGames, "shard-games", 250, "games per shard")
	fs.StringVar(&c.SGFDir, "sgf", "", "also write SGF records here")
	statsPath := fs.String("stats", "", "write run statistics as JSON")
	fs.Parse(args)
	c.Superko = true
	c.Search = sf.config(c.Seed)
	eval, closeEval, err := nf.evaluator(c.Size, c.Seed)
	if err != nil {
		return err
	}
	defer closeEval()
	if c.Parallel == 0 {
		w := nf.workers
		if w == 0 {
			w = runtime.NumCPU()
		}
		c.Parallel = 8 * w
	}
	c.Log = func(f string, a ...any) { fmt.Fprintf(os.Stderr, f+"\n", a...) }
	st, err := selfplay.Run(c, eval)
	if err != nil {
		return err
	}
	out := map[string]any{
		"games": st.Games, "moves": st.Moves, "rows": st.Rows, "black_wins": st.BlackWins,
		"resigned": st.Resigned, "false_resigns": st.FalseResigns, "no_resign_triggered": st.NoResignWithTrigger,
		"seconds": st.Elapsed.Seconds(), "games_per_second": float64(st.Games) / st.Elapsed.Seconds(),
		"shards": st.Shards,
	}
	if net, ok := eval.(*nn.Net); ok {
		out["nn_positions"] = net.Stats.Positions.Load()
		out["nn_batches"] = net.Stats.Batches.Load()
		out["nn_cache_hits"] = net.Stats.CacheHits.Load()
		out["nn_seconds"] = float64(net.Stats.NetNanos.Load()) / 1e9
	}
	fmt.Fprintf(os.Stderr, "selfplay: %d games, %d rows in %.0fs (%.2f games/s), black won %.1f%%, resigned %d, false resigns %d/%d\n",
		st.Games, st.Rows, st.Elapsed.Seconds(), out["games_per_second"], 100*float64(st.BlackWins)/float64(st.Games),
		st.Resigned, st.FalseResigns, st.NoResignWithTrigger)
	if *statsPath != "" {
		data, _ := json.MarshalIndent(out, "", "  ")
		return os.WriteFile(*statsPath, data, 0o644)
	}
	return nil
}

func cmdArena(args []string) error {
	fs := flag.NewFlagSet("arena", flag.ExitOnError)
	var nf netFlags
	var sf searchFlags
	nf.register(fs)
	sf.register(fs, 200)
	a := fs.String("a", "", "player A spec (nn:PATH | rollout | uniform | gnugo,level=N | gtp:CMD), options ,visits=N")
	bspec := fs.String("b", "", "player B spec")
	var c arena.Config
	fs.IntVar(&c.Size, "size", 9, "board size")
	fs.Float64Var(&c.Komi, "komi", 7.5, "komi")
	fs.IntVar(&c.Games, "games", 100, "games (even: each opening is played with both colors)")
	fs.IntVar(&c.Parallel, "parallel", 0, "concurrent games (0: 2 per CPU)")
	fs.IntVar(&c.OpeningMoves, "opening", 2, "random opening plies")
	fs.Int64Var(&c.Seed, "seed", 1, "random seed")
	fs.StringVar(&c.SGFDir, "sgf", "", "write game records here")
	sprt := fs.String("sprt", "", "stop early via SPRT: elo0,elo1 (e.g. 0,35)")
	jsonPath := fs.String("json", "", "write the result as JSON")
	fs.Parse(args)
	if *a == "" || *bspec == "" {
		return fmt.Errorf("arena needs -a and -b")
	}
	pa, err := parsePlayerSpec(*a, sf.visits)
	if err != nil {
		return err
	}
	pb, err := parsePlayerSpec(*bspec, sf.visits)
	if err != nil {
		return err
	}
	if *sprt != "" {
		var e0, e1 float64
		if _, err := fmt.Sscanf(*sprt, "%g,%g", &e0, &e1); err != nil {
			return fmt.Errorf("bad -sprt %q", *sprt)
		}
		c.SPRT = &arena.SPRT{Elo0: e0, Elo1: e1, Alpha: 0.05, Beta: 0.05}
	}
	if c.Parallel == 0 {
		c.Parallel = 2 * runtime.NumCPU()
	}
	c.Superko = true
	c.Log = func(f string, a ...any) { fmt.Fprintf(os.Stderr, f+"\n", a...) }
	pool := &nets{m: map[string]*nn.Net{}, f: &nf, size: c.Size}
	defer pool.close()
	var seed int64 = c.Seed * 7919
	res, err := arena.Run(c, pa.factory(pool, &sf, &seed), pb.factory(pool, &sf, &seed))
	if err != nil {
		return err
	}
	fmt.Printf("%s vs %s: %d-%d-%d  score %.3f [%.3f, %.3f]  elo %+.0f [%+.0f, %+.0f]  black wins %.0f%%  %s\n",
		res.NameA, res.NameB, res.Wins, res.Losses, res.Draws, res.Score, res.ScoreLow, res.ScoreHigh,
		res.Elo, res.EloLow, res.EloHigh, 100*res.BlackWinRate, res.SPRTDecision)
	if *jsonPath != "" {
		data, _ := json.MarshalIndent(res, "", "  ")
		return os.WriteFile(*jsonPath, data, 0o644)
	}
	return nil
}

func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	var nf netFlags
	nf.register(fs)
	size := fs.Int("size", 9, "board size")
	games := fs.Int("games", 64, "concurrent self-play-like searches")
	visits := fs.Int("visits", 100, "visits per move")
	dur := fs.Duration("time", 10*time.Second, "benchmark duration")
	fs.Parse(args)
	eval, closeEval, err := nf.evaluator(*size, 1)
	if err != nil {
		return err
	}
	defer closeEval()
	var playouts, moves int64
	done := make(chan struct{})
	results := make(chan [2]int64, *games)
	for g := 0; g < *games; g++ {
		go func(g int) {
			var p, m int64
			cfg := mcts.DefaultConfig()
			cfg.Seed = int64(g)
			s := mcts.New(cfg, eval)
			b := game.NewBoard(game.Options{Size: *size, Komi: 7.5, Superko: true})
			s.Reset(b)
			for {
				select {
				case <-done:
					results <- [2]int64{p, m}
					return
				default:
				}
				if b.GameOver() || b.MoveNum() > 2*b.NumPoints() {
					b = game.NewBoard(game.Options{Size: *size, Komi: 7.5, Superko: true})
					s.Reset(b)
				}
				if err := s.Run(*visits); err != nil {
					results <- [2]int64{p, m}
					return
				}
				mv := s.SampleMove(1)
				b.Play(mv)
				s.Advance(mv, b)
				p += int64(*visits)
				m++
			}
		}(g)
	}
	time.Sleep(*dur)
	close(done)
	for g := 0; g < *games; g++ {
		r := <-results
		playouts += r[0]
		moves += r[1]
	}
	sec := dur.Seconds()
	fmt.Printf("bench: %d searches, %.0f playouts/s, %.1f moves/s", *games, float64(playouts)/sec, float64(moves)/sec)
	if net, ok := eval.(*nn.Net); ok {
		pos := net.Stats.Positions.Load()
		fmt.Printf(", nn %.0f positions/s, mean batch %.1f, cache hits %.0f%%",
			float64(pos)/sec, float64(pos)/float64(max(net.Stats.Batches.Load(), 1)),
			100*float64(net.Stats.CacheHits.Load())/float64(max(pos+net.Stats.CacheHits.Load(), 1)))
	}
	fmt.Println()
	return nil
}
