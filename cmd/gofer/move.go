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

func movesEqual(a, b Move) bool {
	if a.Pass != b.Pass {
		return false
	}
	if a.Pass {
		return true
	}
	return a.Point == b.Point
}
