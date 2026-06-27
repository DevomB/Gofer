package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/DevomB/gofer/internal/gofer"
)

func main() {
	games := flag.Int("games", 1, "number of games")
	size := flag.Int("size", 9, "board size")
	playouts := flag.Int("playouts", 20, "MCTS playouts per move")
	out := flag.String("o", "", "output JSON file (stdout if empty)")
	flag.Parse()

	cfg := gofer.DefaultSelfplayConfig()
	cfg.Games = *games
	cfg.BoardSize = *size
	cfg.Playouts = *playouts

	samples := gofer.RunSelfplay(cfg)
	data, err := json.MarshalIndent(samples, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *out == "" {
		fmt.Println(string(data))
		return
	}
	if err := os.WriteFile(*out, data, 0644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
