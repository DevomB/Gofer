package main

// Move is a stone play or pass.
type Move struct {
	Point Point
	Pass  bool
}

// PassMove is the pass move sentinel.
var PassMove = Move{Pass: true}

// StoneMove returns a move at p.
func StoneMove(p Point) Move {
	return Move{Point: p}
}
