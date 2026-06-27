package board

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
