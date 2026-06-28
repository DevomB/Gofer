package main

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

// TrompTaylor returns Tromp-Taylor rules with positional superko.
func TrompTaylor() Ruleset {
	return newTrompRules()
}
