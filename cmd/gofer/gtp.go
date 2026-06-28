package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Session holds GTP engine state.
type Session struct {
	Board  *Board
	Rules  Ruleset
	Search *Engine
	Size   int
	Komi   float64
}

// NewSession creates a default 19x19 session.
func NewSession() *Session {
	size, komi := 19, 7.5
	b := NewBoard(size, komi)
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 8
	cfg.Seed = 1
	return &Session{
		Board:  b,
		Rules:  r,
		Search: NewEngine(r, nil, cfg),
		Size:   size,
		Komi:   komi,
	}
}

// Handle processes one GTP command line and returns the response body (without =/? prefix).
func (s *Session) Handle(line string) string {
	line = strings.TrimSpace(line)
	if line == "" {
		return ""
	}
	parts := strings.Fields(line)
	cmd := strings.ToLower(parts[0])
	switch cmd {
	case "protocol_version":
		return "2"
	case "name":
		return "Gofer"
	case "version":
		return "0.1"
	case "known_command":
		return "true"
	case "list_commands":
		return "boardsize clear_board komi play genmove quit"
	case "boardsize":
		if len(parts) < 2 {
			return "boardsize not an integer"
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil || n < 2 || n > 19 {
			return "unacceptable size"
		}
		s.Size = n
		s.Board = NewBoard(n, s.Komi)
		s.Search.ResetArena()
		return ""
	case "clear_board":
		s.Board = NewBoard(s.Size, s.Komi)
		s.Search.ResetArena()
		return ""
	case "komi":
		if len(parts) < 2 {
			return "komi not a float"
		}
		k, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			return "komi not a float"
		}
		s.Komi = k
		s.Board = NewBoard(s.Size, k)
		s.Search.ResetArena()
		return ""
	case "play":
		if len(parts) < 3 {
			return "syntax error"
		}
		color, err := parseColor(parts[1])
		if err != nil {
			return err.Error()
		}
		if s.Board.Player() != color {
			return "wrong color to move"
		}
		m, err := parseVertex(parts[2], s.Size)
		if err != nil {
			return err.Error()
		}
		if !s.Rules.Play(s.Board, m) {
			return "illegal move"
		}
		s.Search.ResetArena()
		return ""
	case "genmove":
		if len(parts) < 2 {
			return "syntax error"
		}
		color, err := parseColor(parts[1])
		if err != nil {
			return err.Error()
		}
		if s.Board.Player() != color {
			return "wrong color to move"
		}
		m := s.Search.BestMove(s.Board)
		if !s.Rules.Play(s.Board, m) {
			return "pass"
		}
		return moveToVertex(m, s.Size)
	case "quit":
		return ""
	default:
		return "? unknown command"
	}
}

func parseColor(s string) (Color, error) {
	switch strings.ToUpper(s) {
	case "B", "BLACK":
		return Black, nil
	case "W", "WHITE":
		return White, nil
	default:
		return Empty, fmt.Errorf("invalid color")
	}
}

func parseVertex(v string, size int) (Move, error) {
	if strings.ToLower(v) == "pass" {
		return PassMove, nil
	}
	v = strings.ToUpper(strings.TrimSpace(v))
	if len(v) < 2 {
		return Move{}, fmt.Errorf("invalid vertex")
	}
	x := int(v[0] - 'A')
	if v[0] >= 'I' {
		x--
	}
	row, err := strconv.Atoi(v[1:])
	if err != nil || row < 1 || row > size {
		return Move{}, fmt.Errorf("invalid vertex")
	}
	y := size - row
	if x < 0 || y < 0 || x >= size || y >= size {
		return Move{}, fmt.Errorf("invalid vertex")
	}
	return StoneMove(At(x, y)), nil
}

func moveToVertex(m Move, size int) string {
	if m.Pass {
		return "pass"
	}
	col := 'A' + m.Point.X
	if col >= 'I' {
		col++
	}
	row := size - m.Point.Y
	return fmt.Sprintf("%c%d", col, row)
}
