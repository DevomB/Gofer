package board

// Point is a board intersection.
type Point struct {
	X, Y int
}

// Idx returns linear index or -1 if off board.
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
