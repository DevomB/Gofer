package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Point is a board intersection.
type Point struct {
	X, Y int
}

// Idx returns linear index or -1 if off
func (p Point) Idx(size int) int {
	if p.X < 0 || p.Y < 0 || p.X >= size || p.Y >= size {
		return -1
	}
	return p.Y*size + p.X
}

// At builds a Point from coordinates.
func At(x, y int) Point {
	return Point{X: x, Y: y}
}

// PointFromIdx converts linear index to Point.
func PointFromIdx(size, idx int) Point {
	return Point{X: idx % size, Y: idx / size}
}

func parseGTPVertex(v string, size int) (Move, error) {
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

func moveToGTPVertex(m Move, size int) string {
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
