package main

import (
	"fmt"
	"strings"
)

// Color is stone color or empty.
type Color uint8

const (
	Empty Color = iota
	Black
	White
)

// Stone is an alias for Color on the grid.
type Stone = Color

func (c Color) Opposite() Color {
	switch c {
	case Black:
		return White
	case White:
		return Black
	default:
		return Empty
	}
}

func parseGTPColor(s string) (Color, error) {
	switch strings.ToUpper(s) {
	case "B", "BLACK":
		return Black, nil
	case "W", "WHITE":
		return White, nil
	default:
		return Empty, fmt.Errorf("invalid color")
	}
}
