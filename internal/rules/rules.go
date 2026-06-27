package rules

import "github.com/DevomB/gofer/internal/board"

// Ruleset applies rule-specific legality, scoring, and move application.
type Ruleset interface {
	LegalMoves(b *board.Board) []board.Move
	Play(b *board.Board, m board.Move) bool
	Score(b *board.Board) (black, white float64)
}

// Chinese returns the v1 Chinese rules implementation.
func Chinese() Ruleset {
	return &chineseRules{}
}

// TrompTaylor returns Tromp-Taylor rules (M2 — not implemented).
func TrompTaylor() Ruleset {
	return &trompRules{}
}

// NewBoard creates an empty board.
func NewBoard(size int, komi float64) *board.Board {
	return board.New(size, komi)
}

type trompRules struct{}

func (r *trompRules) LegalMoves(b *board.Board) []board.Move {
	panic("tromp-taylor: not implemented (M2)")
}

func (r *trompRules) Play(b *board.Board, m board.Move) bool {
	panic("tromp-taylor: not implemented (M2)")
}

func (r *trompRules) Score(b *board.Board) (black, white float64) {
	panic("tromp-taylor: not implemented (M2)")
}
