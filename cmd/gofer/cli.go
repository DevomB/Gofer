package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func parseEvaluator(name string) Evaluator {
	if strings.EqualFold(name, "uniform") {
		return Uniform{}
	}
	return Heuristic{}
}

func newSearchEngine(r Ruleset, playouts int, think time.Duration, evalName string) *Engine {
	cfg := DefaultConfig()
	cfg.Playouts = playouts
	cfg.ThinkTime = think
	return NewEngine(r, parseEvaluator(evalName), cfg)
}

func defaultPlayoutsForSize(size int) int {
	switch {
	case size <= 9:
		return 400
	case size <= 13:
		return 800
	default:
		return 1600
	}
}

func runAnalyze(size int, komi float64, playouts int, think time.Duration, topN int, evalName string, setup []string) {
	r := Chinese()
	b := NewBoard(size, komi)
	if err := applySetupMoves(r, b, setup); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if playouts <= 0 && think <= 0 {
		playouts = defaultPlayoutsForSize(size)
	}
	eng := newSearchEngine(r, playouts, think, evalName)
	a := eng.Analyze(b, topN)
	printBoard(b, size)
	fmt.Printf("\n%s to move | %d playouts | root value %.3f\n\n",
		colorName(b.Player()), a.Playouts, a.RootValue)
	fmt.Println("  move      visits   share   winrate")
	for _, c := range a.Candidates {
		fmt.Printf("  %-8s %7d %6.1f%% %7.3f\n",
			moveToGTPVertex(c.Move, size), c.Visits, c.Share*100, c.WinRate)
	}
	if len(a.PV) > 0 {
		parts := make([]string, len(a.PV))
		for i, m := range a.PV {
			parts[i] = moveToGTPVertex(m, size)
		}
		fmt.Printf("\nPV: %s\n", strings.Join(parts, " "))
	}
	fmt.Printf("\nBest: %s\n", moveToGTPVertex(a.Best, size))
}

func runPlay(size int, komi float64, playouts int, think time.Duration, evalName, humanColor string, sgfOut string) {
	r := Chinese()
	b := NewBoard(size, komi)
	log := NewGameLog(size, komi)
	eng := newSearchEngine(r, playouts, think, evalName)
	human := Black
	if strings.EqualFold(humanColor, "w") || strings.EqualFold(humanColor, "white") {
		human = White
	}
	fmt.Printf("Gofer %dx%d komi=%.1f — you are %s\n", size, size, komi, colorName(human))
	fmt.Println("Commands: board | play <vertex> | genmove | analyze | score | undo | quit")
	printBoard(b, size)

	in := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !in.Scan() {
			break
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		switch strings.ToLower(parts[0]) {
		case "quit", "exit", "q":
			if sgfOut != "" {
				if err := log.WriteSGF(sgfOut); err != nil {
					fmt.Fprintln(os.Stderr, err)
				} else {
					fmt.Printf("saved %s\n", sgfOut)
				}
			}
			return
		case "board", "show":
			printBoard(b, size)
		case "score":
			fmt.Println(formatGTPScore(b, r))
		case "undo":
			if !b.CanUndo() {
				fmt.Println("nothing to undo")
				continue
			}
			b.Undo()
			if len(log.Moves) > 0 {
				log.Moves = log.Moves[:len(log.Moves)-1]
			}
			eng.ResetArena()
			printBoard(b, size)
		case "analyze":
			n := 5
			if len(parts) >= 2 {
				if v, err := strconv.Atoi(parts[1]); err == nil {
					n = v
				}
			}
			a := eng.Analyze(b, n)
			fmt.Printf("%d playouts | root %.3f\n", a.Playouts, a.RootValue)
			for _, c := range a.Candidates {
				fmt.Printf("  %-8s %7d %6.1f%% %7.3f\n",
					moveToGTPVertex(c.Move, size), c.Visits, c.Share*100, c.WinRate)
			}
		case "genmove":
			if b.Player() == human {
				fmt.Println("your turn — use play <vertex>")
				continue
			}
			m := eng.BestMove(b)
			color := b.Player()
			if !r.Play(b, m) {
				m = PassMove
				r.Play(b, m)
			}
			log.Record(color, m)
			eng.AdvanceTree(m)
			fmt.Printf("engine %s\n", moveToGTPVertex(m, size))
			printBoard(b, size)
			if onlyPass(r.LegalMoves(b)) {
				fmt.Println("game over (only pass left)")
				fmt.Println(formatGTPScore(b, r))
			}
		case "play":
			if len(parts) < 2 {
				fmt.Println("usage: play <vertex>  (e.g. D4 or pass)")
				continue
			}
			if b.Player() != human {
				fmt.Println("engine turn — use genmove")
				continue
			}
			m, err := parseGTPVertex(parts[1], size)
			if err != nil {
				fmt.Println(err.Error())
				continue
			}
			if !r.Play(b, m) {
				fmt.Println("illegal move")
				continue
			}
			log.Record(human, m)
			eng.AdvanceTree(m)
			fmt.Printf("played %s\n", moveToGTPVertex(m, size))
			printBoard(b, size)
		default:
			fmt.Println("unknown command")
		}
	}
}

func applySetupMoves(r Ruleset, b *Board, moves []string) error {
	for i, raw := range moves {
		m, err := parseGTPVertex(raw, b.Size())
		if err != nil {
			return fmt.Errorf("setup move %d: %w", i, err)
		}
		if !r.Play(b, m) {
			return fmt.Errorf("illegal setup move %d: %s", i, raw)
		}
	}
	return nil
}

func printBoard(b *Board, size int) {
	fmt.Println(formatGTPBoard(b, size))
	fmt.Printf("%s to move\n", colorName(b.Player()))
}

func colorName(c Color) string {
	switch c {
	case Black:
		return "Black"
	case White:
		return "White"
	default:
		return "?"
	}
}
