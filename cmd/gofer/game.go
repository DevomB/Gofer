package main

import "os"

// GameLog records moves for SGF export.
type GameLog struct {
	Size  int
	Komi  float64
	Moves []SGFMove
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
