package game

import (
	"math/rand"
	"testing"
)

// refBoard is a deliberately naive flood-fill implementation used as an oracle.
type refBoard struct {
	size  int
	cells []Color
	ko    int
	turn  Color
	seen  map[string]bool
}

func newRef(size int) *refBoard {
	r := &refBoard{size: size, cells: make([]Color, size*size), ko: -1, turn: Black, seen: map[string]bool{}}
	r.seen[r.key()] = true
	return r
}

func (r *refBoard) key() string { return string(colorsToBytes(r.cells)) }

func colorsToBytes(c []Color) []byte {
	out := make([]byte, len(c))
	for i, v := range c {
		out[i] = byte(v)
	}
	return out
}

func (r *refBoard) nbrs(p int) []int {
	x, y := p%r.size, p/r.size
	var out []int
	if x > 0 {
		out = append(out, p-1)
	}
	if x+1 < r.size {
		out = append(out, p+1)
	}
	if y > 0 {
		out = append(out, p-r.size)
	}
	if y+1 < r.size {
		out = append(out, p+r.size)
	}
	return out
}

func (r *refBoard) group(cells []Color, p int) (stones []int, libs map[int]bool) {
	c := cells[p]
	libs = map[int]bool{}
	seen := map[int]bool{p: true}
	stack := []int{p}
	for len(stack) > 0 {
		q := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		stones = append(stones, q)
		for _, n := range r.nbrs(q) {
			switch {
			case cells[n] == Empty:
				libs[n] = true
			case cells[n] == c && !seen[n]:
				seen[n] = true
				stack = append(stack, n)
			}
		}
	}
	return stones, libs
}

// try returns the resulting cells, captured stones, ko point and legality (ignoring superko).
func (r *refBoard) try(p int) ([]Color, int, int, bool) {
	if r.cells[p] != Empty || p == r.ko {
		return nil, 0, -1, false
	}
	cells := append([]Color(nil), r.cells...)
	cells[p] = r.turn
	captured, lastCapt := 0, -1
	for _, n := range r.nbrs(p) {
		if cells[n] == r.turn.Opp() {
			stones, libs := r.group(cells, n)
			if len(libs) == 0 {
				for _, s := range stones {
					cells[s] = Empty
				}
				captured += len(stones)
				lastCapt = n
			}
		}
	}
	stones, libs := r.group(cells, p)
	if len(libs) == 0 {
		return nil, 0, -1, false
	}
	ko := -1
	if captured == 1 && len(stones) == 1 && len(libs) == 1 {
		ko = lastCapt
	}
	return cells, captured, ko, true
}

func (r *refBoard) legal(p int, superko bool) bool {
	cells, _, _, ok := r.try(p)
	if !ok {
		return false
	}
	return !superko || !r.seen[string(colorsToBytes(cells))]
}

func (r *refBoard) play(p int) {
	if p < 0 {
		r.ko = -1
	} else {
		cells, _, ko, _ := r.try(p)
		r.cells, r.ko = cells, ko
		r.seen[r.key()] = true
	}
	r.turn = r.turn.Opp()
}

func TestBoardMatchesReference(t *testing.T) {
	for _, size := range []int{5, 9, 13, 19} {
		for _, superko := range []bool{false, true} {
			rng := rand.New(rand.NewSource(int64(size)*7 + 1))
			games := 60
			if size >= 13 {
				games = 6
			}
			if testing.Short() {
				games = 5
			}
			for g := 0; g < games; g++ {
				b := NewBoard(Options{Size: size, Komi: 7, Superko: superko})
				ref := newRef(size)
				for move := 0; move < size*size*3; move++ {
					var legal []Point
					for p := 0; p < size*size; p++ {
						got, want := b.IsLegal(Point(p)), ref.legal(p, superko)
						if got != want {
							t.Fatalf("size %d game %d move %d: IsLegal(%s)=%v want %v\n%s",
								size, g, move, b.FormatPoint(Point(p)), got, want, b)
						}
						if got {
							legal = append(legal, Point(p))
						}
					}
					m := Pass
					if len(legal) > 0 && rng.Intn(20) > 0 {
						m = legal[rng.Intn(len(legal))]
					}
					if err := b.Play(m); err != nil {
						t.Fatal(err)
					}
					ref.play(int(m))
					for p := range ref.cells {
						if b.At(Point(p)) != ref.cells[p] {
							t.Fatalf("size %d game %d move %d: cell %d = %v want %v", size, g, move, p, b.At(Point(p)), ref.cells[p])
						}
					}
					if int(b.Ko()) != ref.ko && !(b.Ko() == Pass && ref.ko == -1) {
						t.Fatalf("ko %d want %d", b.Ko(), ref.ko)
					}
					for p := 0; p < size*size; p++ {
						if b.At(Point(p)) == Empty {
							continue
						}
						_, libs := ref.group(ref.cells, p)
						if b.Libs(Point(p)) != len(libs) {
							t.Fatalf("libs at %d = %d want %d", p, b.Libs(Point(p)), len(libs))
						}
					}
					if b.GameOver() {
						break
					}
				}
			}
		}
	}
}

func setup(t *testing.T, size int, rows ...string) *Board {
	t.Helper()
	b := NewBoard(Options{Size: size, Komi: 0})
	for y, row := range rows {
		for x, ch := range row {
			p := b.P(x, y)
			switch ch {
			case 'X':
				b.placeForTest(p, Black)
			case 'O':
				b.placeForTest(p, White)
			}
		}
	}
	return b
}

// placeForTest drops a stone without captures by playing it as the right color.
func (b *Board) placeForTest(p Point, c Color) {
	b.toPlay = c
	b.play(p)
	b.moveNum, b.passes, b.ko = 0, 0, Pass
	for i := range b.last {
		b.last[i] = noMove
	}
}

func TestSimpleKo(t *testing.T) {
	b := setup(t, 5,
		".XO..",
		"X.XO.",
		".XO..",
		".....",
		".....")
	b.SetToPlay(White)
	if err := b.Play(b.P(1, 1)); err != nil { // O captures X at C4
		t.Fatal(err)
	}
	if b.Ko() != b.P(2, 1) {
		t.Fatalf("ko = %s, want C4", b.FormatPoint(b.Ko()))
	}
	if b.IsLegal(b.P(2, 1)) {
		t.Fatal("immediate ko recapture must be illegal")
	}
	b.Play(Pass)
	b.Play(Pass) // white passes back; ko lifts after any move
	if !b.IsLegal(b.P(2, 1)) {
		t.Fatal("ko recapture should be legal after intervening moves")
	}
}

func TestSnapbackIsNotKo(t *testing.T) {
	// Capturing one stone with a stone that is part of a larger group is not ko.
	b := setup(t, 5,
		"OXX..",
		".O...",
		".....",
		".....",
		".....")
	b.SetToPlay(Black)
	if err := b.Play(b.P(0, 1)); err != nil { // X at A4 captures O at A5
		t.Fatal(err)
	}
	if b.Ko() != Pass {
		t.Fatalf("ko set to %s after capture by a connected stone", b.FormatPoint(b.Ko()))
	}
}

func TestSuicideIllegal(t *testing.T) {
	b := setup(t, 5,
		".X...",
		"X....",
		".....",
		".....",
		".....")
	b.SetToPlay(White)
	if b.IsLegal(b.P(0, 0)) {
		t.Fatal("suicide must be illegal")
	}
}

func TestScoreAndOwnership(t *testing.T) {
	b := setup(t, 5,
		".X.O.",
		".X.O.",
		"XX.OO",
		".....",
		".....")
	b.SetKomi(0.5)
	bl, wl := b.Score()
	// Black: 4 stones + 2 territory (A5, A4); white: 4 stones + 2 territory (E5, E4) + komi.
	if bl != 6 || wl != 6.5 {
		t.Fatalf("score = %v,%v want 6,6.5", bl, wl)
	}
	if b.Winner() != White {
		t.Fatal("white should win")
	}
}

func TestKeyDistinguishesSideToMove(t *testing.T) {
	b := NewBoard(Options{Size: 9})
	k0 := b.Key()
	b.Play(Pass)
	if b.Key() == k0 {
		t.Fatal("pass must change the NN cache key")
	}
	if b.StoneHash() != NewBoard(Options{Size: 9}).StoneHash() {
		t.Fatal("pass must not change the stone hash")
	}
}

func TestParseFormatPoint(t *testing.T) {
	for _, size := range []int{9, 19} {
		for p := Point(0); int(p) < size*size; p++ {
			s := FormatPoint(p, size)
			q, err := ParsePoint(s, size)
			if err != nil || q != p {
				t.Fatalf("round trip %d -> %s -> %d (%v)", p, s, q, err)
			}
		}
	}
	if _, err := ParsePoint("I5", 9); err == nil {
		t.Fatal("column I does not exist")
	}
}

func BenchmarkRandomGame9x9(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	buf := make([]Point, 0, 82)
	for i := 0; i < b.N; i++ {
		bd := NewBoard(Options{Size: 9, Komi: 7})
		for move := 0; move < 200 && !bd.GameOver(); move++ {
			buf = bd.LegalMoves(buf[:0])
			bd.PlayUnchecked(buf[rng.Intn(len(buf))])
		}
	}
}

func BenchmarkClone9x9(b *testing.B) {
	bd := NewBoard(Options{Size: 9, Komi: 7})
	dst := bd.Clone()
	for i := 0; i < b.N; i++ {
		dst.CopyFrom(bd)
	}
}
