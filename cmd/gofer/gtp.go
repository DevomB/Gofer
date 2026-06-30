package main

import (
	"strconv"
	"strings"
)

var gtpCommands = []string{
	"boardsize", "clear_board", "final_score", "genmove", "komi",
	"known_command", "list_commands", "name", "play", "protocol_version",
	"quit", "showboard", "time_left", "time_settings", "version",
}

// SessionConfig holds GTP session parameters.
type SessionConfig struct {
	Playouts int
	Eval     string // "uniform" or "heuristic"
}

// Session holds GTP engine state.
type Session struct {
	Board  *Board
	Rules  Ruleset
	Search *Engine
	Size   int
	Komi   float64
}

// NewSession creates a GTP session with the given config.
func NewSession(cfg SessionConfig) *Session {
	size, komi := 19, 7.5
	b := NewBoard(size, komi)
	r := Chinese()
	playouts := cfg.Playouts
	if playouts <= 0 {
		playouts = 200
	}
	ev := Evaluator(Heuristic{})
	if cfg.Eval == "uniform" {
		ev = Uniform{}
	}
	scfg := DefaultConfig()
	scfg.Playouts = playouts
	scfg.Seed = 1
	return &Session{
		Board:  b,
		Rules:  r,
		Search: NewEngine(r, ev, scfg),
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
		return s.handleKnownCommand(parts)
	case "list_commands":
		return strings.Join(gtpCommands, " ")
	case "boardsize":
		return s.handleBoardsize(parts)
	case "clear_board":
		return s.handleClearBoard()
	case "komi":
		return s.handleKomi(parts)
	case "play":
		return s.handlePlay(parts)
	case "genmove":
		return s.handleGenmove(parts)
	case "showboard":
		return formatGTPBoard(s.Board, s.Size)
	case "final_score":
		return formatGTPScore(s.Board, s.Rules)
	case "time_settings", "time_left":
		return ""
	case "quit":
		return ""
	default:
		return "? unknown command"
	}
}

func (s *Session) handleKnownCommand(parts []string) string {
	if len(parts) < 2 {
		return "false"
	}
	cmd := strings.ToLower(parts[1])
	for _, c := range gtpCommands {
		if c == cmd {
			return "true"
		}
	}
	return "false"
}

func (s *Session) handleBoardsize(parts []string) string {
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
}

func (s *Session) handleClearBoard() string {
	s.Board = NewBoard(s.Size, s.Komi)
	s.Search.ResetArena()
	return ""
}

func (s *Session) handleKomi(parts []string) string {
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
}

func (s *Session) handlePlay(parts []string) string {
	if len(parts) < 3 {
		return "syntax error"
	}
	color, err := parseGTPColor(parts[1])
	if err != nil {
		return err.Error()
	}
	if s.Board.Player() != color {
		return "wrong color to move"
	}
	m, err := parseGTPVertex(parts[2], s.Size)
	if err != nil {
		return err.Error()
	}
	if !s.Rules.Play(s.Board, m) {
		return "illegal move"
	}
	s.Search.AdvanceTree(m)
	return ""
}

func (s *Session) handleGenmove(parts []string) string {
	if len(parts) < 2 {
		return "syntax error"
	}
	color, err := parseGTPColor(parts[1])
	if err != nil {
		return err.Error()
	}
	if s.Board.Player() != color {
		return "wrong color to move"
	}
	m := s.Search.BestMove(s.Board)
	if !s.Rules.Play(s.Board, m) {
		s.Search.AdvanceTree(m)
		return "pass"
	}
	s.Search.AdvanceTree(m)
	return moveToGTPVertex(m, s.Size)
}
