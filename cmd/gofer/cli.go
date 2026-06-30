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

func printAnalyzeTable(a Analysis, size int, header string) {
	if header != "" {
		fmt.Println(header)
	}
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
	header := fmt.Sprintf("\n%s to move | %d playouts | root value %.3f\n",
		colorName(b.Player()), a.Playouts, a.RootValue)
	printAnalyzeTable(a, size, header)
}

type playSession struct {
	r     Ruleset
	b     *Board
	log   *GameLog
	eng   *Engine
	human Color
	size  int
}

func newPlaySession(size int, komi float64, playouts int, think time.Duration, evalName, humanColor string) *playSession {
	r := Chinese()
	human := Black
	if strings.EqualFold(humanColor, "w") || strings.EqualFold(humanColor, "white") {
		human = White
	}
	return &playSession{
		r:     r,
		b:     NewBoard(size, komi),
		log:   NewGameLog(size, komi),
		eng:   newSearchEngine(r, playouts, think, evalName),
		human: human,
		size:  size,
	}
}

func (ps *playSession) printTurnHint() {
	n := len(ps.r.LegalMoves(ps.b))
	if ps.b.Player() == ps.human {
		fmt.Printf("%d legal moves | your turn — play <vertex>\n", n)
	} else {
		fmt.Printf("%d legal moves | engine turn — genmove\n", n)
	}
}

func (ps *playSession) cmdQuit(sgfOut string) bool {
	if sgfOut != "" {
		if err := ps.log.WriteSGF(sgfOut); err != nil {
			fmt.Fprintln(os.Stderr, err)
		} else {
			fmt.Printf("saved %s\n", sgfOut)
		}
	}
	return true
}

func (ps *playSession) cmdUndo() {
	if !ps.b.CanUndo() {
		fmt.Println("nothing to undo")
		return
	}
	ps.b.Undo()
	if len(ps.log.Moves) > 0 {
		ps.log.Moves = ps.log.Moves[:len(ps.log.Moves)-1]
	}
	ps.eng.ResetArena()
	printBoard(ps.b, ps.size)
	ps.printTurnHint()
}

func (ps *playSession) cmdAnalyze(parts []string) {
	n := 5
	if len(parts) >= 2 {
		if v, err := strconv.Atoi(parts[1]); err == nil {
			n = v
		}
	}
	a := ps.eng.Analyze(ps.b, n)
	header := fmt.Sprintf("%d playouts | root %.3f", a.Playouts, a.RootValue)
	printAnalyzeTable(a, ps.size, header)
}

func (ps *playSession) cmdGenmove() {
	if ps.b.Player() == ps.human {
		fmt.Println("your turn — use play <vertex>")
		return
	}
	m := ps.eng.BestMove(ps.b)
	color := ps.b.Player()
	if !ps.r.Play(ps.b, m) {
		m = PassMove
		ps.r.Play(ps.b, m)
	}
	ps.log.Record(color, m)
	ps.eng.AdvanceTree(m)
	fmt.Printf("engine %s\n", moveToGTPVertex(m, ps.size))
	printBoard(ps.b, ps.size)
	if onlyPass(ps.r.LegalMoves(ps.b)) {
		fmt.Println("game over (only pass left)")
		fmt.Println(formatGTPScore(ps.b, ps.r))
	} else {
		ps.printTurnHint()
	}
}

func (ps *playSession) cmdPlay(parts []string) {
	if len(parts) < 2 {
		fmt.Println("usage: play <vertex>  (e.g. D4 or pass)")
		return
	}
	if ps.b.Player() != ps.human {
		fmt.Println("engine turn — use genmove")
		return
	}
	m, err := parseGTPVertex(parts[1], ps.size)
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	if !ps.r.Play(ps.b, m) {
		fmt.Println("illegal move")
		return
	}
	ps.log.Record(ps.human, m)
	ps.eng.AdvanceTree(m)
	fmt.Printf("played %s\n", moveToGTPVertex(m, ps.size))
	printBoard(ps.b, ps.size)
	ps.printTurnHint()
}

func (ps *playSession) handleCommand(parts []string, sgfOut string) bool {
	switch strings.ToLower(parts[0]) {
	case "quit", "exit", "q":
		return ps.cmdQuit(sgfOut)
	case "board", "show":
		printBoard(ps.b, ps.size)
	case "score":
		fmt.Println(formatGTPScore(ps.b, ps.r))
	case "undo":
		ps.cmdUndo()
	case "analyze":
		ps.cmdAnalyze(parts)
	case "genmove":
		ps.cmdGenmove()
	case "play":
		ps.cmdPlay(parts)
	default:
		fmt.Println("unknown command — board | play | genmove | analyze | score | undo | quit")
	}
	return false
}

func runPlay(size int, komi float64, playouts int, think time.Duration, evalName, humanColor string, sgfOut string) {
	ps := newPlaySession(size, komi, playouts, think, evalName, humanColor)
	fmt.Printf("Gofer %dx%d komi=%.1f — you are %s\n", size, size, komi, colorName(ps.human))
	fmt.Println("Commands: board | play <vertex> | genmove | analyze | score | undo | quit")
	printBoard(ps.b, size)
	ps.printTurnHint()

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
		if ps.handleCommand(strings.Fields(line), sgfOut) {
			return
		}
	}
}

func watchUntilEnd(r Ruleset, b *Board, eng *Engine, size int, onMove func(Color, Move)) int {
	passes := 0
	for moveNum := 0; moveNum < size*size+2; moveNum++ {
		if onlyPass(r.LegalMoves(b)) {
			break
		}
		color := b.Player()
		m := eng.BestMove(b)
		if !r.Play(b, m) {
			m = PassMove
			r.Play(b, m)
		}
		if onMove != nil {
			onMove(color, m)
		}
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
	return passes
}

func runWatch(size int, komi float64, playouts int, think time.Duration, evalName string) {
	if playouts <= 0 && think <= 0 {
		playouts = defaultPlayoutsForSize(size)
	}
	r := Chinese()
	b := NewBoard(size, komi)
	eng := newSearchEngine(r, playouts, think, evalName)
	fmt.Printf("Gofer watch %dx%d komi=%.1f\n", size, size, komi)
	moveNum := 0
	watchUntilEnd(r, b, eng, size, func(color Color, m Move) {
		moveNum++
		fmt.Printf("\n%d. %s %s\n", moveNum, colorName(color), moveToGTPVertex(m, size))
		printBoard(b, size)
	})
	fmt.Println("\n" + formatGTPScore(b, r))
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
