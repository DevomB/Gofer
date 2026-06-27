package sgf

import (
	"fmt"

	"github.com/DevomB/gofer/internal/board"
)

// Move is a color and optional board point (pass when Point is nil).
type Move struct {
	Color board.Color
	Point *board.Point
}

// ParseCoord decodes SGF coords (aa = upper-left) for any board size.
func ParseCoord(size int, s string) (board.Point, error) {
	if len(s) != 2 {
		return board.Point{}, fmt.Errorf("sgf coord %q: want 2 letters", s)
	}
	x := int(s[0] - 'a')
	y := int(s[1] - 'a')
	if x < 0 || y < 0 || x >= size || y >= size {
		return board.Point{}, fmt.Errorf("sgf coord %q off %dx%d board", s, size, size)
	}
	return board.At(x, y), nil
}

// ponytail: scans ;B[aa] / ;W[] tokens only — no variations or setup stones.
// Ceiling: not a full FF[4] parser.
// Upgrade: tree + AB/AW setup (backlog-core-engine M3).
func ParseMoves(data string) ([]Move, error) {
	var moves []Move
	i := 0
	for i < len(data) {
		if data[i] != ';' {
			i++
			continue
		}
		i++
		if i >= len(data) {
			break
		}
		tag := data[i]
		i++
		if i >= len(data) || data[i] != '[' {
			continue
		}
		i++
		start := i
		for i < len(data) && data[i] != ']' {
			i++
		}
		val := data[start:i]
		if i < len(data) {
			i++
		}
		var c board.Color
		switch tag {
		case 'B':
			c = board.Black
		case 'W':
			c = board.White
		default:
			continue
		}
		if val == "" {
			moves = append(moves, Move{Color: c})
			continue
		}
		if len(val) != 2 {
			return nil, fmt.Errorf("bad coord %q", val)
		}
		pt := board.At(int(val[0]-'a'), int(val[1]-'a'))
		moves = append(moves, Move{Color: c, Point: &pt})
	}
	return moves, nil
}
