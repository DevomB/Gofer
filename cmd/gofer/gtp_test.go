package main

import (
	"strings"
	"testing"
	"time"
)

func TestGTPVersion(t *testing.T) {
	s := NewSession(SessionConfig{})
	if out := s.Handle("version"); out != "1.0" {
		t.Fatalf("version: %q", out)
	}
}

func TestGTPSGFLog(t *testing.T) {
	s := NewSession(SessionConfig{Playouts: 4, BoardSize: 5})
	s.Handle("boardsize 5")
	s.Handle("play black D3")
	s.Handle("genmove white")
	if len(s.log.Moves) != 2 {
		t.Fatalf("moves %d want 2", len(s.log.Moves))
	}
	out := s.log.ExportSGF()
	if !strings.Contains(out, ";B[") || !strings.Contains(out, ";W[") {
		t.Fatalf("sgf missing moves: %s", out)
	}
}

func TestGTPClearResetsLog(t *testing.T) {
	s := NewSession(SessionConfig{BoardSize: 5})
	s.Handle("boardsize 5")
	s.Handle("play black D3")
	s.Handle("clear_board")
	if len(s.log.Moves) != 0 {
		t.Fatalf("clear_board should reset log, got %d moves", len(s.log.Moves))
	}
}

func TestWatchGameEnds(t *testing.T) {
	r := Chinese()
	b := NewBoard(5, 6.5)
	cfg := DefaultConfig()
	cfg.Playouts = 4
	eng := NewEngine(r, Uniform{}, cfg)
	moves := 0
	passes := watchUntilEnd(r, b, eng, 5, func(_ Color, _ Move) { moves++ })
	if moves == 0 {
		t.Fatal("watch game produced no moves")
	}
	if passes < 2 && !onlyPass(r.LegalMoves(b)) {
		t.Fatalf("watch game did not end (passes=%d moves=%d)", passes, moves)
	}
}

func TestGTPBoardsize(t *testing.T) {
	s := NewSession(SessionConfig{})
	if out := s.Handle("boardsize 9"); out != "" {
		t.Fatalf("boardsize: %q", out)
	}
}

func TestGTPKnownCommand(t *testing.T) {
	s := NewSession(SessionConfig{})
	if out := s.Handle("known_command genmove"); out != "true" {
		t.Fatalf("known genmove: %q", out)
	}
	if out := s.Handle("known_command not_a_command"); out != "false" {
		t.Fatalf("unknown command: %q", out)
	}
}

func TestGTPShowboard(t *testing.T) {
	s := NewSession(SessionConfig{})
	s.Handle("boardsize 9")
	out := s.Handle("showboard")
	if out == "" || !strings.Contains(out, ".") {
		t.Fatalf("showboard empty or missing dots: %q", out)
	}
}

func TestGTPFinalScore(t *testing.T) {
	s := NewSession(SessionConfig{})
	s.Handle("boardsize 9")
	s.Handle("play black pass")
	s.Handle("play white pass")
	out := s.Handle("final_score")
	if out == "" {
		t.Fatal("final_score empty")
	}
}

func TestGTPTimeLeft(t *testing.T) {
	s := NewSession(SessionConfig{Playouts: 10})
	s.Handle("boardsize 9")
	if out := s.Handle("time_left black 30 0"); out != "" {
		t.Fatalf("time_left: %q", out)
	}
	if s.nextThink != 30*time.Second {
		t.Fatalf("nextThink=%v", s.nextThink)
	}
}
