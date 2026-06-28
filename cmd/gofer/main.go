package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	gtpMode := flag.Bool("gtp", false, "run GTP protocol on stdin/stdout")
	selfplayMode := flag.Bool("selfplay", false, "run self-play and emit training samples")
	games := flag.Int("games", 1, "self-play games (with -selfplay)")
	size := flag.Int("size", 9, "board size")
	komi := flag.Float64("komi", 6.5, "komi")
	playouts := flag.Int("playouts", 20, "MCTS playouts per move (with -selfplay)")
	out := flag.String("o", "", "write self-play JSON to path (stdout if empty)")
	flag.Parse()

	if *gtpMode {
		runGTP()
		return
	}
	if *selfplayMode {
		runSelfplayCLI(*games, *size, *playouts, *out)
		return
	}

	b := NewBoard(*size, *komi)
	r := Chinese()
	moves := r.LegalMoves(b)
	fmt.Printf("gofer chinese: %dx%d komi=%.1f, %d legal moves\n",
		*size, *size, *komi, len(moves))
}

func runGTP() {
	s := NewSession()
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

func runSelfplayCLI(games, size, playouts int, outPath string) {
	cfg := DefaultSelfplayConfig()
	cfg.Games = games
	cfg.BoardSize = size
	cfg.Playouts = playouts
	samples := RunSelfplay(cfg)
	data, err := json.MarshalIndent(samples, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if outPath == "" {
		fmt.Println(string(data))
		return
	}
	if err := os.WriteFile(outPath, data, 0644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
