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

func main() {
	gtpMode := flag.Bool("gtp", false, "run GTP protocol on stdin/stdout")
	playMode := flag.Bool("play", false, "interactive terminal game")
	analyzeMode := flag.Bool("analyze", false, "analyze current position")
	selfplayMode := flag.Bool("selfplay", false, "run self-play and emit training samples")
	games := flag.Int("games", 1, "number of games (with -selfplay)")
	size := flag.Int("size", 9, "board size")
	komi := flag.Float64("komi", 6.5, "komi")
	playouts := flag.Int("playouts", 0, "MCTS playouts per move (0 = size default)")
	gtpPlayouts := flag.Int("gtp-playouts", 0, "MCTS playouts per GTP move (0 = size default)")
	think := flag.Duration("think-time", 0, "search time budget per move (overrides -playouts when set)")
	topN := flag.Int("top", 5, "top moves to show (with -analyze)")
	evalMode := flag.String("eval", "heuristic", "evaluator: uniform or heuristic")
	humanColor := flag.String("color", "b", "your color in -play: b or w")
	out := flag.String("o", "", "output path (self-play JSON or game SGF)")
	sgfDir := flag.String("sgf-dir", "", "write self-play games as SGF files to directory")
	sgfPath := flag.String("sgf", "", "replay SGF file and print score")
	setup := flag.String("moves", "", "comma-separated setup moves for -analyze (e.g. D4,Q16)")
	flag.Parse()

	if *sgfPath != "" {
		runSGFReplay(*sgfPath)
		return
	}
	if *gtpMode {
		runGTP(*gtpPlayouts, *size, *think, *evalMode)
		return
	}
	if *playMode {
		p := *playouts
		if p <= 0 {
			p = defaultPlayoutsForSize(*size)
		}
		runPlay(*size, *komi, p, *think, *evalMode, *humanColor, *out)
		return
	}
	if *analyzeMode {
		p := *playouts
		if p <= 0 && *think <= 0 {
			p = defaultPlayoutsForSize(*size)
		}
		var setupMoves []string
		if *setup != "" {
			for _, m := range strings.Split(*setup, ",") {
				if s := strings.TrimSpace(m); s != "" {
					setupMoves = append(setupMoves, s)
				}
			}
		}
		runAnalyze(*size, *komi, p, *think, *topN, *evalMode, setupMoves)
		return
	}
	if *selfplayMode {
		runSelfplayCLI(*games, *size, *komi, *playouts, *out, *sgfDir)
		return
	}

	b := NewBoard(*size, *komi)
	r := Chinese()
	moves := r.LegalMoves(b)
	fmt.Printf("gofer %dx%d komi=%.1f, %d legal moves\n", *size, *size, *komi, len(moves))
	fmt.Println("\nUsage:")
	fmt.Println("  gofer -play              interactive game")
	fmt.Println("  gofer -analyze           MCTS analysis")
	fmt.Println("  gofer -gtp                GTP engine (Sabaki/Lizzie)")
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

func runGTP(playouts, defaultSize int, think time.Duration, evalMode string) {
	if playouts <= 0 {
		playouts = defaultPlayoutsForSize(defaultSize)
	}
	s := NewSession(SessionConfig{
		Playouts:   playouts,
		ThinkTime:  think,
		Eval:       evalMode,
		BoardSize:  defaultSize,
	})
	in := bufio.NewScanner(os.Stdin)
	for in.Scan() {
		line := in.Text()
		if strings.TrimSpace(line) == "quit" {
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
		data, err := json.MarshalIndent(samples, "", "  ")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := os.WriteFile(outPath, data, 0644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	} else if sgfDir == "" {
		data, _ := json.MarshalIndent(samples, "", "  ")
		fmt.Println(string(data))
	}
	if sgfDir != "" {
		if err := os.MkdirAll(sgfDir, 0755); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		for i, log := range logs {
			path := fmt.Sprintf("%s/game-%03d.sgf", sgfDir, i+1)
			if err := log.WriteSGF(path); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		}
		fmt.Printf("wrote %d SGF files to %s\n", len(logs), sgfDir)
	}
}
