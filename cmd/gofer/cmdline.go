package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

type cliFlags struct {
	gtp, play, analyze, selfplay, watch bool
	games                               int
	size                                int
	komi                                float64
	playouts, gtpPlayouts               int
	think                               time.Duration
	topN                                int
	eval, humanColor, out, sgfDir       string
	sgfPath, setup                        string
}

func parseCLIFlags() cliFlags {
	var f cliFlags
	flag.BoolVar(&f.gtp, "gtp", false, "run GTP protocol on stdin/stdout")
	flag.BoolVar(&f.play, "play", false, "interactive terminal game")
	flag.BoolVar(&f.analyze, "analyze", false, "analyze current position")
	flag.BoolVar(&f.selfplay, "selfplay", false, "run self-play and emit training samples")
	flag.BoolVar(&f.watch, "watch", false, "engine vs engine demo (mutually exclusive with other modes)")
	flag.IntVar(&f.games, "games", 1, "number of games (with -selfplay)")
	flag.IntVar(&f.size, "size", 9, "board size")
	flag.Float64Var(&f.komi, "komi", 6.5, "komi")
	flag.IntVar(&f.playouts, "playouts", 0, "MCTS playouts per move (0 = size default)")
	flag.IntVar(&f.gtpPlayouts, "gtp-playouts", 0, "MCTS playouts per GTP move (0 = size default)")
	flag.DurationVar(&f.think, "think-time", 0, "search time budget per move (overrides -playouts when set)")
	flag.IntVar(&f.topN, "top", 5, "top moves to show (with -analyze)")
	flag.StringVar(&f.eval, "eval", "heuristic", "evaluator: uniform or heuristic")
	flag.StringVar(&f.humanColor, "color", "b", "your color in -play: b or w")
	flag.StringVar(&f.out, "o", "", "output path (self-play JSON, game SGF, or GTP SGF on quit)")
	flag.StringVar(&f.sgfDir, "sgf-dir", "", "write self-play games as SGF files to directory")
	flag.StringVar(&f.sgfPath, "sgf", "", "replay SGF file and print score")
	flag.StringVar(&f.setup, "moves", "", "comma-separated setup moves for -analyze (e.g. D4,Q16)")
	flag.Parse()
	return f
}

func runMode(f cliFlags) bool {
	n := 0
	for _, on := range []bool{f.gtp, f.play, f.analyze, f.selfplay, f.watch} {
		if on {
			n++
		}
	}
	if n > 1 {
		fmt.Fprintln(os.Stderr, "only one mode flag allowed (-gtp, -play, -analyze, -selfplay, -watch)")
		os.Exit(1)
	}
	switch {
	case f.gtp:
		runGTP(f.gtpPlayouts, f.size, f.think, f.eval, f.out)
	case f.play:
		p := f.playouts
		if p <= 0 {
			p = defaultPlayoutsForSize(f.size)
		}
		runPlay(f.size, f.komi, p, f.think, f.eval, f.humanColor, f.out)
	case f.analyze:
		p := f.playouts
		if p <= 0 && f.think <= 0 {
			p = defaultPlayoutsForSize(f.size)
		}
		runAnalyze(f.size, f.komi, p, f.think, f.topN, f.eval, parseSetupMoves(f.setup))
	case f.selfplay:
		runSelfplayCLI(f.games, f.size, f.komi, f.playouts, f.out, f.sgfDir)
	case f.watch:
		p := f.playouts
		if p <= 0 && f.think <= 0 {
			p = defaultPlayoutsForSize(f.size)
		}
		runWatch(f.size, f.komi, p, f.think, f.eval)
	default:
		return false
	}
	return true
}

func parseSetupMoves(setup string) []string {
	if setup == "" {
		return nil
	}
	var moves []string
	for _, m := range strings.Split(setup, ",") {
		if s := strings.TrimSpace(m); s != "" {
			moves = append(moves, s)
		}
	}
	return moves
}

func printUsage(size int, komi float64) {
	b := NewBoard(size, komi)
	r := Chinese()
	fmt.Printf("gofer %dx%d komi=%.1f, %d legal moves\n", size, size, komi, len(r.LegalMoves(b)))
	fmt.Println("\nUsage:")
	fmt.Println("  gofer -play              interactive game")
	fmt.Println("  gofer -analyze           MCTS analysis")
	fmt.Println("  gofer -gtp                GTP engine (Sabaki/Lizzie)")
	fmt.Println("  gofer -watch              engine vs engine demo")
	fmt.Println("  gofer -selfplay           generate training games")
	fmt.Println("  gofer -sgf game.sgf       replay SGF")
}

func runSGFReplay(path string) {
	n, bl, wl, err := ReplaySGFFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("replayed %d moves: black=%.1f white=%.1f\n", n, bl, wl)
}

func runGTP(playouts, defaultSize int, think time.Duration, evalMode, sgfOut string) {
	if playouts <= 0 {
		playouts = defaultPlayoutsForSize(defaultSize)
	}
	s := NewSession(SessionConfig{
		Playouts:  playouts,
		ThinkTime: think,
		Eval:      evalMode,
		BoardSize: defaultSize,
	})
	in := bufio.NewScanner(os.Stdin)
	for in.Scan() {
		line := in.Text()
		if strings.TrimSpace(line) == "quit" {
			if sgfOut != "" {
				if err := s.log.WriteSGF(sgfOut); err != nil {
					fmt.Fprintln(os.Stderr, err)
				}
			}
			fmt.Print("= \n\n")
			break
		}
		out := s.Handle(line)
		if strings.HasPrefix(out, "?") {
			fmt.Printf("? %s\n\n", out)
		} else {
			fmt.Printf("= %s\n\n", out)
		}
	}
}

func runSelfplayCLI(games, size int, komi float64, playouts int, outPath, sgfDir string) {
	if playouts <= 0 {
		playouts = defaultPlayoutsForSize(size)
	}
	cfg := DefaultSelfplayConfig()
	cfg.Games = games
	cfg.BoardSize = size
	cfg.Komi = komi
	cfg.Playouts = playouts
	samples, logs := RunSelfplayWithLogs(cfg)
	if outPath != "" {
		writeSelfplayJSON(outPath, samples)
	} else if sgfDir == "" {
		data, _ := json.MarshalIndent(samples, "", "  ")
		fmt.Println(string(data))
	}
	if sgfDir != "" {
		writeSelfplaySGFs(sgfDir, logs)
	}
}

func writeSelfplayJSON(path string, samples []Sample) {
	data, err := json.MarshalIndent(samples, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func writeSelfplaySGFs(dir string, logs []*GameLog) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for i, log := range logs {
		path := fmt.Sprintf("%s/game-%03d.sgf", dir, i+1)
		if err := log.WriteSGF(path); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	fmt.Printf("wrote %d SGF files to %s\n", len(logs), dir)
}
