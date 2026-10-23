// Package nn encodes positions for the network, defines the evaluator
// interface the search uses, and batches evaluation requests across goroutines.
package nn

import "github.com/DevomB/gofer/internal/game"

// Feature layout (schema 3). Keep in sync with training/gofer/features.py.
const (
	SchemaVersion = 3

	PlaneOwn       = 0
	PlaneOpp       = 1
	PlaneOnBoard   = 2 // all ones: lets convolutions see the edge through zero padding
	PlaneOwnLib1   = 3
	PlaneOwnLib2   = 4
	PlaneOppLib1   = 5
	PlaneOppLib2   = 6
	PlaneKo        = 7
	PlaneHistory   = 8 // 8..12: last 5 moves, most recent first
	NumPlanes      = PlaneHistory + game.HistoryLen
	GlobalKomi     = 0 // komi from the side to move's perspective / 15
	GlobalPass1    = 1 // opponent's last move was a pass
	GlobalPass2    = 2 // our previous move was a pass
	GlobalProgress = 3 // move number / board area (capped at 2)
	NumGlobals     = 4
)

// Encode writes the features of b into spatial (NumPlanes*n) and global
// (NumGlobals), viewed through symmetry sym. Both slices must be zeroed.
func Encode(b *game.Board, sym int, spatial, global []float32) {
	n := b.NumPoints()
	t := symmetry(b.Size(), sym)
	me, opp := b.ToPlay(), b.ToPlay().Opp()
	for p := 0; p < n; p++ {
		q := int(t[p])
		spatial[PlaneOnBoard*n+q] = 1
		c := b.At(game.Point(p))
		if c == game.Empty {
			continue
		}
		libs := b.Libs(game.Point(p))
		if c == me {
			spatial[PlaneOwn*n+q] = 1
			if libs == 1 {
				spatial[PlaneOwnLib1*n+q] = 1
			} else if libs == 2 {
				spatial[PlaneOwnLib2*n+q] = 1
			}
		} else if c == opp {
			spatial[PlaneOpp*n+q] = 1
			if libs == 1 {
				spatial[PlaneOppLib1*n+q] = 1
			} else if libs == 2 {
				spatial[PlaneOppLib2*n+q] = 1
			}
		}
	}
	if ko := b.Ko(); ko != game.Pass {
		spatial[PlaneKo*n+int(t[ko])] = 1
	}
	for i := 0; i < game.HistoryLen; i++ {
		if m, ok := b.LastMove(i); ok && m != game.Pass {
			spatial[(PlaneHistory+i)*n+int(t[m])] = 1
		}
	}
	komi := b.Komi()
	if me == game.Black {
		komi = -komi
	}
	global[GlobalKomi] = float32(komi / 15)
	if m, ok := b.LastMove(0); ok && m == game.Pass {
		global[GlobalPass1] = 1
	}
	if m, ok := b.LastMove(1); ok && m == game.Pass {
		global[GlobalPass2] = 1
	}
	progress := float32(b.MoveNum()) / float32(n)
	if progress > 2 {
		progress = 2
	}
	global[GlobalProgress] = progress
}

// EncodeBytes writes identity-symmetry spatial features as 0/1 bytes
// (the on-disk training format).
func EncodeBytes(b *game.Board, spatial []uint8, global []float32) {
	n := b.NumPoints()
	tmp := make([]float32, NumPlanes*n)
	Encode(b, 0, tmp, global)
	for i, v := range tmp {
		spatial[i] = uint8(v)
	}
}

// symTables[size][s][p] maps point p to its image under dihedral symmetry s.
var symTables [game.MaxSize + 1]*[8][]game.Point

func init() {
	for size := 2; size <= game.MaxSize; size++ {
		var tabs [8][]game.Point
		n := size * size
		for s := 0; s < 8; s++ {
			tabs[s] = make([]game.Point, n)
			for p := 0; p < n; p++ {
				x, y := p%size, p/size
				if s&4 != 0 {
					x, y = y, x
				}
				if s&1 != 0 {
					x = size - 1 - x
				}
				if s&2 != 0 {
					y = size - 1 - y
				}
				tabs[s][p] = game.Point(y*size + x)
			}
		}
		symTables[size] = &tabs
	}
}

func symmetry(size, s int) []game.Point { return symTables[size][s] }

// Symmetry exposes the point mapping for symmetry s (used by tests and tools).
func Symmetry(size, s int) []game.Point { return symmetry(size, s) }
