package main

import "os"

// GameEnd says why a game loop stopped. Only a finished game has a result: one
// stopped at the move cap is scored mid-fight, so its score describes nothing.
type GameEnd string

const (
	EndTwoPasses GameEnd = "two_passes"
	EndOnlyPass  GameEnd = "only_pass"
	EndMoveCap   GameEnd = "move_cap"
)

// maxGameMoves bounds a game loop on a size×size board. Captures reopen points,
// so finished games run past the board's area; the cap only stops a game that
// never ends, and a game that reaches it has no result.
func maxGameMoves(size int) int { return 3*size*size + 2 }

// GameLog records moves for SGF export, and why the game stopped.
type GameLog struct {
	Size  int
	Komi  float64
	Moves []SGFMove
	End   GameEnd
}

// countEndings tallies why each game stopped.
func countEndings(logs []*GameLog) map[GameEnd]int {
	out := make(map[GameEnd]int)
	for _, l := range logs {
		out[l.End]++
	}
	return out
}

// NewGameLog starts an empty game record.
func NewGameLog(size int, komi float64) *GameLog {
	return &GameLog{Size: size, Komi: komi}
}

// Record appends a move by the given color.
func (g *GameLog) Record(color Color, m Move) {
	sm := SGFMove{Color: color}
	if !m.Pass {
		pt := m.Point
		sm.Point = &pt
	}
	g.Moves = append(g.Moves, sm)
}

// ExportSGF returns the game as an SGF string.
func (g *GameLog) ExportSGF() string {
	meta := &SGFGame{Size: g.Size, Komi: g.Komi}
	return ExportSGF(meta, g.Moves)
}

// WriteSGF writes the game to path.
func (g *GameLog) WriteSGF(path string) error {
	return os.WriteFile(path, []byte(g.ExportSGF()), 0644)
}
