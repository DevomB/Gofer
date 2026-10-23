package game

import (
	"math/bits"
	"math/rand"
	"sync"
)

const bitsetWords = (maxPoints + 63) / 64

// bitset is a fixed-size set of board points.
type bitset [bitsetWords]uint64

func (s *bitset) set(i int)      { s[i>>6] |= 1 << (uint(i) & 63) }
func (s *bitset) clear(i int)    { s[i>>6] &^= 1 << (uint(i) & 63) }
func (s *bitset) has(i int) bool { return s[i>>6]&(1<<(uint(i)&63)) != 0 }
func (s *bitset) or(o *bitset) {
	for i := range s {
		s[i] |= o[i]
	}
}

func (s *bitset) isZero() bool {
	for _, w := range s {
		if w != 0 {
			return false
		}
	}
	return true
}

func (s *bitset) count() int {
	n := 0
	for _, w := range s {
		n += bits.OnesCount64(w)
	}
	return n
}

func (s *bitset) forEach(fn func(int)) {
	for wi, w := range s {
		for w != 0 {
			t := bits.TrailingZeros64(w)
			fn(wi<<6 | t)
			w &= w - 1
		}
	}
}

// neighborTable holds precomputed orthogonal neighbors for one board size.
type neighborTable struct {
	start []int32 // start[p]..start[p+1] indexes into list
	list  []Point
}

func (t *neighborTable) of(p Point) []Point { return t.list[t.start[p]:t.start[p+1]] }

var (
	nbOnce  [MaxSize + 1]sync.Once
	nbTable [MaxSize + 1]*neighborTable
)

func neighbors(size int) *neighborTable {
	nbOnce[size].Do(func() {
		n := size * size
		t := &neighborTable{start: make([]int32, n+1)}
		for p := 0; p < n; p++ {
			t.start[p] = int32(len(t.list))
			x, y := p%size, p/size
			if x > 0 {
				t.list = append(t.list, Point(p-1))
			}
			if x+1 < size {
				t.list = append(t.list, Point(p+1))
			}
			if y > 0 {
				t.list = append(t.list, Point(p-size))
			}
			if y+1 < size {
				t.list = append(t.list, Point(p+size))
			}
		}
		t.start[n] = int32(len(t.list))
		nbTable[size] = t
	})
	return nbTable[size]
}

var (
	zobristStone       [maxPoints][3]uint64
	zobristKo          [maxPoints]uint64
	zobristLast        [HistoryLen][maxPoints + 2]uint64 // index m+2 so noMove=-2 and Pass=-1 fit
	zobristWhiteToPlay uint64
)

func init() {
	rng := rand.New(rand.NewSource(0x60eef0bacafe))
	for i := range zobristStone {
		zobristStone[i][Black] = rng.Uint64()
		zobristStone[i][White] = rng.Uint64()
		zobristKo[i] = rng.Uint64()
	}
	for i := range zobristLast {
		for j := range zobristLast[i] {
			zobristLast[i][j] = rng.Uint64()
		}
	}
	zobristWhiteToPlay = rng.Uint64()
}
