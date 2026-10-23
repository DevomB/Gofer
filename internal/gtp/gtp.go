// Package gtp serves the Go Text Protocol (version 2) so Gofer can be used
// from Sabaki, Lizzie, GoGui, KGS clients or tournament managers.
package gtp

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
)

// Engine is a GTP-controlled player.
type Engine struct {
	Name, Version string
	Eval          nn.Evaluator
	Search        mcts.Config
	Visits        int
	Superko       bool
	ResignBelow   float64 // resign when the root value is below this (0 = never)
	Log           io.Writer

	size  int
	komi  float64
	board *game.Board
	moves []move
	s     *mcts.Search
	clock [3]clock // per color, from time_settings / time_left
}

// clock is a player's remaining time as reported by the controller.
type clock struct {
	set    bool
	main   time.Duration // main time left, or byo-yomi period left when stones > 0
	stones int           // stones to play in the current byo-yomi period (0: main time)
	byo    time.Duration // byo-yomi period length
}

// budget returns how long to think for the next move (0: no time control).
func (c clock) budget(movesLeft int) time.Duration {
	if !c.set {
		return 0
	}
	if c.stones > 0 {
		return c.main * 9 / (10 * time.Duration(c.stones))
	}
	t := c.main / time.Duration(max(movesLeft, 8))
	if c.byo > 0 {
		t += c.byo * 8 / 10
	}
	return t
}

type move struct {
	c game.Color
	p game.Point
}

// NewEngine returns an engine with a 9x9 board.
func NewEngine(e Engine) *Engine {
	if e.Name == "" {
		e.Name = "Gofer"
	}
	if e.Log == nil {
		e.Log = io.Discard
	}
	e.size, e.komi = 9, 7.5
	e.reset()
	return &e
}

func (e *Engine) reset() {
	e.board = game.NewBoard(game.Options{Size: e.size, Komi: e.komi, Superko: e.Superko})
	e.moves = nil
	e.s = mcts.New(e.Search, e.Eval)
	e.s.Reset(e.board)
}

// replay rebuilds the board from the move list (after undo or komi change).
func (e *Engine) replay() error {
	moves := e.moves
	e.reset()
	for _, m := range moves {
		if err := e.apply(m.c, m.p); err != nil {
			return err
		}
	}
	return nil
}

func (e *Engine) apply(c game.Color, p game.Point) error {
	e.board.SetToPlay(c)
	if err := e.board.Play(p); err != nil {
		return err
	}
	e.moves = append(e.moves, move{c, p})
	e.s.Advance(p, e.board)
	return nil
}

var commands = []string{
	"protocol_version", "name", "version", "known_command", "list_commands", "quit",
	"boardsize", "clear_board", "komi", "play", "genmove", "undo", "showboard",
	"final_score", "time_settings", "time_left", "gofer-analyze", "lz-analyze",
}

// maxAnalysisVisits bounds tree growth during open-ended analysis.
const maxAnalysisVisits = 2_000_000

// Serve runs the GTP loop until "quit" or EOF.
func (e *Engine) Serve(r io.Reader, w io.Writer) error {
	lines := make(chan string)
	scanErr := make(chan error, 1)
	go func() {
		sc := bufio.NewScanner(r)
		for sc.Scan() {
			lines <- sc.Text()
		}
		scanErr <- sc.Err()
		close(lines)
	}()
	out := bufio.NewWriter(w)
	defer out.Flush()
	var pending *string
	for {
		var raw string
		if pending != nil {
			raw, pending = *pending, nil
		} else {
			l, ok := <-lines
			if !ok {
				return <-scanErr
			}
			raw = l
		}
		line := strings.TrimSpace(strings.SplitN(raw, "#", 2)[0])
		if line == "" {
			continue
		}
		f := strings.Fields(line)
		id := ""
		if _, err := strconv.Atoi(f[0]); err == nil {
			id, f = f[0], f[1:]
			if len(f) == 0 {
				continue
			}
		}
		cmd := strings.ToLower(f[0])
		if cmd == "lz-analyze" {
			next, err := e.lzAnalyze(id, f[1:], out, lines)
			if err != nil {
				fmt.Fprintf(out, "?%s %s\n\n", id, err)
				out.Flush()
			}
			if next == nil {
				return <-scanErr // input closed during analysis
			}
			if *next != "" {
				pending = next
			}
			continue
		}
		resp, err := e.handle(cmd, f[1:])
		if err != nil {
			fmt.Fprintf(out, "?%s %s\n\n", id, err)
		} else {
			fmt.Fprintf(out, "=%s %s\n\n", id, resp)
		}
		out.Flush()
		if cmd == "quit" {
			return nil
		}
	}
}

// lzAnalyze implements Leela Zero's streaming analysis (used by Sabaki and
// Lizzie): "lz-analyze [color] [interval-centiseconds]". It prints "info"
// lines until the next command arrives, which it returns (nil on EOF).
func (e *Engine) lzAnalyze(id string, args []string, out *bufio.Writer, lines <-chan string) (*string, error) {
	interval := 100 * time.Millisecond
	for _, a := range args {
		if c, err := parseColor(a); err == nil {
			if e.board.ToPlay() != c {
				e.board.SetToPlay(c)
				e.s.Reset(e.board)
			}
			continue
		}
		cs, err := strconv.Atoi(a)
		if err != nil {
			return new(string), fmt.Errorf("syntax error")
		}
		if cs > 0 {
			interval = time.Duration(cs) * 10 * time.Millisecond
		}
	}
	if e.board.GameOver() {
		e.s.Reset(e.board)
	}
	fmt.Fprintf(out, "=%s\n", id)
	out.Flush()
	last := time.Now()
	for {
		select {
		case l, ok := <-lines:
			fmt.Fprint(out, "\n")
			out.Flush()
			if !ok {
				return nil, nil
			}
			return &l, nil
		default:
		}
		if e.s.Visits() < maxAnalysisVisits {
			if err := e.s.Run(32); err != nil {
				return new(string), err
			}
		} else {
			time.Sleep(interval)
		}
		if time.Since(last) >= interval {
			fmt.Fprintln(out, e.lzInfo())
			out.Flush()
			last = time.Now()
		}
	}
}

func (e *Engine) lzInfo() string {
	var sb strings.Builder
	for i, c := range e.s.Children() {
		if c.Visits == 0 {
			break
		}
		if i > 0 {
			sb.WriteByte(' ')
		}
		fmt.Fprintf(&sb, "info move %s visits %d winrate %d prior %d order %d pv %s",
			game.FormatPoint(c.Move, e.size), c.Visits, int(5000*(1+c.Q)), int(10000*c.Prior), i,
			formatPV(e.childPV(c.Move), e.size))
	}
	return sb.String()
}

// childPV is the principal variation starting with root move m.
func (e *Engine) childPV(m game.Point) []game.Point {
	pv := e.s.PVFrom(m, 10)
	if len(pv) == 0 {
		return []game.Point{m}
	}
	return pv
}

func (e *Engine) handle(cmd string, args []string) (string, error) {
	switch cmd {
	case "protocol_version":
		return "2", nil
	case "name":
		return e.Name, nil
	case "version":
		return e.Version, nil
	case "known_command":
		if len(args) == 1 {
			for _, c := range commands {
				if c == args[0] {
					return "true", nil
				}
			}
		}
		return "false", nil
	case "list_commands":
		return strings.Join(commands, "\n"), nil
	case "quit":
		return "", nil
	case "boardsize":
		if len(args) != 1 {
			return "", fmt.Errorf("syntax error")
		}
		n, err := strconv.Atoi(args[0])
		if err != nil || n < 2 || n > game.MaxSize {
			return "", fmt.Errorf("unacceptable size")
		}
		e.size = n
		e.reset()
		return "", nil
	case "clear_board":
		e.reset()
		return "", nil
	case "komi":
		if len(args) != 1 {
			return "", fmt.Errorf("syntax error")
		}
		k, err := strconv.ParseFloat(args[0], 64)
		if err != nil {
			return "", fmt.Errorf("syntax error")
		}
		e.komi = k
		return "", e.replay()
	case "play":
		if len(args) != 2 {
			return "", fmt.Errorf("syntax error")
		}
		c, err := parseColor(args[0])
		if err != nil {
			return "", err
		}
		p, err := game.ParsePoint(args[1], e.size)
		if err != nil {
			return "", err
		}
		if err := e.apply(c, p); err != nil {
			return "", fmt.Errorf("illegal move")
		}
		return "", nil
	case "genmove":
		if len(args) != 1 {
			return "", fmt.Errorf("syntax error")
		}
		c, err := parseColor(args[0])
		if err != nil {
			return "", err
		}
		return e.genmove(c)
	case "undo":
		if len(e.moves) == 0 {
			return "", fmt.Errorf("cannot undo")
		}
		e.moves = e.moves[:len(e.moves)-1]
		return "", e.replay()
	case "showboard":
		return "\n" + e.board.String(), nil
	case "final_score":
		m := e.board.Margin()
		switch {
		case m > 0:
			return fmt.Sprintf("B+%g", m), nil
		case m < 0:
			return fmt.Sprintf("W+%g", -m), nil
		}
		return "0", nil
	case "time_settings":
		if len(args) != 3 {
			return "", fmt.Errorf("syntax error")
		}
		mainT, err1 := strconv.Atoi(args[0])
		byo, err2 := strconv.Atoi(args[1])
		stones, err3 := strconv.Atoi(args[2])
		if err1 != nil || err2 != nil || err3 != nil {
			return "", fmt.Errorf("syntax error")
		}
		for _, c := range []game.Color{game.Black, game.White} {
			e.clock[c] = clock{set: mainT > 0 || byo > 0, main: time.Duration(mainT) * time.Second, byo: time.Duration(byo) * time.Second}
			if mainT == 0 && stones > 0 {
				e.clock[c].main, e.clock[c].stones = e.clock[c].byo, stones
			}
		}
		return "", nil
	case "time_left":
		if len(args) != 3 {
			return "", fmt.Errorf("syntax error")
		}
		c, err := parseColor(args[0])
		if err != nil {
			return "", err
		}
		secs, err1 := strconv.Atoi(args[1])
		stones, err2 := strconv.Atoi(args[2])
		if err1 != nil || err2 != nil {
			return "", fmt.Errorf("syntax error")
		}
		e.clock[c].set = true
		e.clock[c].main, e.clock[c].stones = time.Duration(secs)*time.Second, stones
		return "", nil
	case "gofer-analyze":
		return e.analyze(args)
	}
	return "", fmt.Errorf("unknown command")
}

func (e *Engine) genmove(c game.Color) (string, error) {
	if e.board.ToPlay() != c {
		e.board.SetToPlay(c)
		e.s.Reset(e.board)
	}
	if e.board.GameOver() {
		e.s.Reset(e.board)
	}
	movesLeft := (e.board.NumPoints() - e.board.MoveNum()/2) / 2
	if budget := e.clock[c].budget(movesLeft); budget > 0 {
		// Under time control, search until the budget is spent (at least one
		// batch, at most maxAnalysisVisits).
		deadline := time.Now().Add(budget)
		for done := 0; done == 0 || (time.Now().Before(deadline) && e.s.Visits() < maxAnalysisVisits); done++ {
			if err := e.s.Run(64); err != nil {
				return "", err
			}
		}
	} else if err := e.s.Run(e.Visits); err != nil {
		return "", err
	}
	v := e.s.RootValue()
	m := e.s.BestMove()
	fmt.Fprintf(e.Log, "genmove %s: %s value %+.3f visits %d pv %s\n",
		c, e.board.FormatPoint(m), v, e.s.Visits(), formatPV(e.s.PV(8), e.size))
	if e.ResignBelow != 0 && v < e.ResignBelow {
		return "resign", nil
	}
	if err := e.apply(c, m); err != nil {
		return "", err
	}
	return game.FormatPoint(m, e.size), nil
}

// analyze runs a search and reports the top moves: gofer-analyze [visits].
func (e *Engine) analyze(args []string) (string, error) {
	visits := e.Visits
	if len(args) > 0 {
		v, err := strconv.Atoi(args[0])
		if err != nil {
			return "", fmt.Errorf("syntax error")
		}
		visits = v
	}
	if err := e.s.Run(visits); err != nil {
		return "", err
	}
	ch := e.s.Children()
	sort.SliceStable(ch, func(i, j int) bool { return ch[i].Visits > ch[j].Visits })
	var sb strings.Builder
	fmt.Fprintf(&sb, "value %+.3f winrate %.1f%% visits %d", e.s.RootValue(), 50*(1+e.s.RootValue()), e.s.Visits())
	for i, c := range ch {
		if i == 8 || c.Visits == 0 {
			break
		}
		fmt.Fprintf(&sb, "\n%-5s visits %5d winrate %5.1f%% prior %5.1f%%", game.FormatPoint(c.Move, e.size), c.Visits, 50*(1+c.Q), 100*c.Prior)
	}
	fmt.Fprintf(&sb, "\npv %s", formatPV(e.s.PV(12), e.size))
	return sb.String(), nil
}

func formatPV(pv []game.Point, size int) string {
	parts := make([]string, len(pv))
	for i, p := range pv {
		parts[i] = game.FormatPoint(p, size)
	}
	return strings.Join(parts, " ")
}

func parseColor(s string) (game.Color, error) {
	switch strings.ToLower(s) {
	case "b", "black":
		return game.Black, nil
	case "w", "white":
		return game.White, nil
	}
	return game.Empty, fmt.Errorf("invalid color")
}
