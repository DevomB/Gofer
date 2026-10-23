// Package arena plays matches between engines (Gofer networks, network-free
// baselines, or external GTP programs such as GNU Go) and reports results
// with confidence intervals, Elo estimates and an optional SPRT stop.
package arena

import (
	"bufio"
	"fmt"
	"io"
	"os/exec"
	"strings"

	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
)

// Player is one side of a game. A fresh Player is created for every game.
type Player interface {
	Name() string
	// NewGame starts a game on the given empty board; opening moves arrive via Observe.
	NewGame(b *game.Board) error
	// GenMove chooses a move for the side to move in b; resign reports resignation.
	GenMove(b *game.Board) (m game.Point, resign bool, err error)
	// Observe is called after every move by either side with the new position.
	Observe(c game.Color, m game.Point, after *game.Board) error
	Close() error
}

// DeadStoneJudge is implemented by players that can mark dead stones at game end.
type DeadStoneJudge interface {
	DeadStones() ([]game.Point, error)
}

// MCTSPlayer searches with a network (or network-free) evaluator.
type MCTSPlayer struct {
	Label  string
	Eval   nn.Evaluator
	Config mcts.Config
	Visits int
	// ResignBelow resigns when the root value stays below it (0 = never).
	ResignBelow float64

	s *mcts.Search
}

func (p *MCTSPlayer) Name() string { return p.Label }

func (p *MCTSPlayer) NewGame(b *game.Board) error {
	p.s = mcts.New(p.Config, p.Eval)
	p.s.Reset(b)
	return nil
}

func (p *MCTSPlayer) GenMove(b *game.Board) (game.Point, bool, error) {
	if p.s.Board().Key() != b.Key() {
		p.s.Reset(b)
	}
	if err := p.s.Run(p.Visits); err != nil {
		return game.Pass, false, err
	}
	if p.ResignBelow != 0 && p.s.RootValue() < p.ResignBelow {
		return game.Pass, true, nil
	}
	return p.s.BestMove(), false, nil
}

func (p *MCTSPlayer) Observe(c game.Color, m game.Point, after *game.Board) error {
	p.s.Advance(m, after)
	return nil
}

func (p *MCTSPlayer) Close() error { return nil }

// GTPPlayer drives an external engine over the Go Text Protocol.
type GTPPlayer struct {
	Label   string
	Command []string

	cmd       *exec.Cmd
	in        io.WriteCloser
	out       *bufio.Reader
	size      int
	generated bool // the engine already knows the last move (it generated it)
}

// NewGTPPlayer prepares an engine; the process starts on NewGame.
func NewGTPPlayer(label string, command []string) *GTPPlayer {
	return &GTPPlayer{Label: label, Command: command}
}

func (p *GTPPlayer) Name() string { return p.Label }

func (p *GTPPlayer) start() error {
	p.cmd = exec.Command(p.Command[0], p.Command[1:]...)
	in, err := p.cmd.StdinPipe()
	if err != nil {
		return err
	}
	out, err := p.cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := p.cmd.Start(); err != nil {
		return fmt.Errorf("arena: start %v: %w", p.Command, err)
	}
	p.in, p.out = in, bufio.NewReader(out)
	return nil
}

// send issues one GTP command and returns the response payload.
func (p *GTPPlayer) send(cmd string) (string, error) {
	if _, err := fmt.Fprintln(p.in, cmd); err != nil {
		return "", err
	}
	var lines []string
	for {
		line, err := p.out.ReadString('\n')
		if err != nil {
			return "", fmt.Errorf("arena: %s: %q: %w", p.Label, cmd, err)
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" && len(lines) > 0 {
			break
		}
		if line != "" {
			lines = append(lines, line)
		}
	}
	resp := strings.Join(lines, "\n")
	if strings.HasPrefix(resp, "?") {
		return "", fmt.Errorf("arena: %s rejected %q: %s", p.Label, cmd, resp)
	}
	return strings.TrimSpace(strings.TrimPrefix(resp, "=")), nil
}

func (p *GTPPlayer) NewGame(b *game.Board) error {
	if p.cmd == nil {
		if err := p.start(); err != nil {
			return err
		}
	}
	p.size = b.Size()
	for _, c := range []string{
		fmt.Sprintf("boardsize %d", b.Size()),
		"clear_board",
		fmt.Sprintf("komi %g", b.Komi()),
	} {
		if _, err := p.send(c); err != nil {
			return err
		}
	}
	return nil
}

func (p *GTPPlayer) GenMove(b *game.Board) (game.Point, bool, error) {
	resp, err := p.send("genmove " + b.ToPlay().String())
	if err != nil {
		return game.Pass, false, err
	}
	p.generated = true
	if strings.EqualFold(resp, "resign") {
		return game.Pass, true, nil
	}
	m, err := game.ParsePoint(resp, p.size)
	return m, false, err
}

func (p *GTPPlayer) Observe(c game.Color, m game.Point, after *game.Board) error {
	if p.generated {
		p.generated = false
		return nil
	}
	_, err := p.send(fmt.Sprintf("play %s %s", c, game.FormatPoint(m, p.size)))
	return err
}

func (p *GTPPlayer) DeadStones() ([]game.Point, error) {
	resp, err := p.send("final_status_list dead")
	if err != nil {
		return nil, err
	}
	var out []game.Point
	for _, f := range strings.Fields(resp) {
		pt, err := game.ParsePoint(f, p.size)
		if err != nil {
			return nil, err
		}
		out = append(out, pt)
	}
	return out, nil
}

func (p *GTPPlayer) Close() error {
	if p.cmd == nil {
		return nil
	}
	p.send("quit")
	p.in.Close()
	return p.cmd.Wait()
}
