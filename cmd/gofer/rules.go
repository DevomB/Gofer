package main

import "fmt"

// Ruleset applies rule-specific legality, scoring, and move application.
type Ruleset interface {
	LegalMoves(b *Board) []Move
	Play(b *Board, m Move) bool
	Score(b *Board) (black, white float64)
}

// Chinese returns the v1 Chinese rules implementation.
func Chinese() Ruleset {
	return &chineseRules{}
}

// playChosen plays a move an engine chose. The rules refusing it means the
// engine's tree and the board disagree: a bug, never a reason to pass instead,
// which would hide it and hand the opponent a free move.
func playChosen(r Ruleset, b *Board, m Move) error {
	if r.Play(b, m) {
		return nil
	}
	return fmt.Errorf("the rules refused %s for %s at move %d (board hash %016x):\n%s",
		moveToGTPVertex(m, b.Size()), colorName(b.Player()), b.MoveNum()+1, b.Hash(), formatGTPBoard(b, b.Size()))
}
