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
	"final_score", "time_settings", "time_left", "gofer-analyze",
}

// Serve runs the GTP loop until "quit" or EOF.
func (e *Engine) Serve(r io.Reader, w io.Writer) error {
	sc := bufio.NewScanner(r)
	out := bufio.NewWriter(w)
	defer out.Flush()
	for sc.Scan() {
		line := strings.TrimSpace(strings.SplitN(sc.Text(), "#", 2)[0])
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
		resp, err := e.handle(strings.ToLower(f[0]), f[1:])
		if err != nil {
			fmt.Fprintf(out, "?%s %s\n\n", id, err)
		} else {
			fmt.Fprintf(out, "=%s %s\n\n", id, resp)
		}
		out.Flush()
		if f[0] == "quit" {
			return nil
		}
	}
	return sc.Err()
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
	case "time_settings", "time_left":
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
	if err := e.s.Run(e.Visits); err != nil {
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
