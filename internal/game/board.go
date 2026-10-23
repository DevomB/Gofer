// Package game implements Go rules: a board with incremental chain and
// liberty tracking, legality (suicide, simple ko, optional positional
// superko), and Tromp-Taylor area scoring.
package game

import (
	"fmt"
	"strings"
)

// Color is a stone color or Empty.
type Color uint8

const (
	Empty Color = iota
	Black
	White
)

// Opp returns the opposing color (Empty stays Empty).
func (c Color) Opp() Color {
	switch c {
	case Black:
		return White
	case White:
		return Black
	}
	return Empty
}

func (c Color) String() string {
	switch c {
	case Black:
		return "B"
	case White:
		return "W"
	}
	return "."
}

// Point is a linear board index y*size+x (y=0 is the top row). Pass is -1.
type Point int16

// Pass is the pass move.
const Pass Point = -1

// MaxSize is the largest supported board.
const MaxSize = 19

const maxPoints = MaxSize * MaxSize

// HistoryLen is how many recent moves the board remembers for NN features.
const HistoryLen = 5

type chain struct {
	stones bitset
	libs   bitset
	hash   uint64 // xor of the chain's stone Zobrist keys
	size   int16
}

// Board is a Go position plus the minimal history needed for rules and features.
// Copy with Clone; the zero value is not usable.
type Board struct {
	size    int
	komi    float64
	cells   []Color
	chainOf []int16 // head point of the chain at each stone, -1 when empty
	chains  []chain // indexed by head point
	nb      *neighborTable

	toPlay  Color
	ko      Point // point forbidden by simple ko, or Pass
	passes  int   // consecutive passes ending the game at 2
	moveNum int
	stones  uint64 // Zobrist hash of stone placement only
	last    [HistoryLen]Point
	capt    [3]int // stones captured by color index

	superko bool
	seen    []uint64 // stone hashes of every earlier position, for positional superko
}

// Options configures rules for a new board.
type Options struct {
	Size    int
	Komi    float64
	Superko bool // forbid recreating any earlier stone configuration
}

// NewBoard returns an empty board with Black to play.
func NewBoard(o Options) *Board {
	if o.Size < 2 || o.Size > MaxSize {
		panic(fmt.Sprintf("game: board size %d out of range", o.Size))
	}
	n := o.Size * o.Size
	b := &Board{
		size:    o.Size,
		komi:    o.Komi,
		cells:   make([]Color, n),
		chainOf: make([]int16, n),
		chains:  make([]chain, n),
		nb:      neighbors(o.Size),
		toPlay:  Black,
		ko:      Pass,
		superko: o.Superko,
	}
	for i := range b.chainOf {
		b.chainOf[i] = -1
	}
	for i := range b.last {
		b.last[i] = noMove
	}
	if b.superko {
		b.seen = append(b.seen, b.stones)
	}
	return b
}

// noMove marks an empty history slot (distinct from Pass).
const noMove Point = -2

// Clone returns an independent deep copy.
func (b *Board) Clone() *Board {
	c := *b
	c.cells = append([]Color(nil), b.cells...)
	c.chainOf = append([]int16(nil), b.chainOf...)
	c.chains = append([]chain(nil), b.chains...)
	if b.superko {
		c.seen = append(make([]uint64, 0, len(b.seen)+8), b.seen...)
	}
	return &c
}

// CopyFrom overwrites b with src without allocating when sizes match.
func (b *Board) CopyFrom(src *Board) {
	if b.size != src.size || b.cells == nil {
		*b = *src.Clone()
		return
	}
	cells, chainOf, chains, seen := b.cells, b.chainOf, b.chains, b.seen
	*b = *src
	b.cells = append(cells[:0], src.cells...)
	b.chainOf = append(chainOf[:0], src.chainOf...)
	b.chains = append(chains[:0], src.chains...)
	if src.superko {
		b.seen = append(seen[:0], src.seen...)
	}
}

func (b *Board) Size() int        { return b.size }
func (b *Board) Komi() float64    { return b.komi }
func (b *Board) ToPlay() Color    { return b.toPlay }
func (b *Board) Ko() Point        { return b.ko }
func (b *Board) Passes() int      { return b.passes }
func (b *Board) MoveNum() int     { return b.moveNum }
func (b *Board) Superko() bool    { return b.superko }
func (b *Board) At(p Point) Color { return b.cells[p] }
func (b *Board) NumPoints() int   { return b.size * b.size }

// GameOver reports whether both players passed consecutively.
func (b *Board) GameOver() bool { return b.passes >= 2 }

// Captures returns the number of stones captured by color c.
func (b *Board) Captures(c Color) int { return b.capt[c] }

// SetToPlay overrides the side to move (SGF/GTP setup).
func (b *Board) SetToPlay(c Color) { b.toPlay = c }

// SetKomi changes komi.
func (b *Board) SetKomi(k float64) { b.komi = k }

// LastMove returns the move played i plies ago (i=0 is the most recent), and
// false when the game is shorter than that.
func (b *Board) LastMove(i int) (Point, bool) {
	m := b.last[i]
	return m, m != noMove
}

// StoneHash is the Zobrist hash of stone placement only (used for superko).
func (b *Board) StoneHash() uint64 { return b.stones }

// Key hashes everything the NN sees: stones, side to move, ko, and recent moves.
func (b *Board) Key() uint64 {
	h := b.stones
	if b.toPlay == White {
		h ^= zobristWhiteToPlay
	}
	if b.ko != Pass {
		h ^= zobristKo[b.ko]
	}
	for i, m := range b.last {
		h ^= zobristLast[i][int(m)+2]
	}
	return h
}

// Libs returns the liberty count of the chain at p (0 for empty points).
func (b *Board) Libs(p Point) int {
	h := b.chainOf[p]
	if h < 0 {
		return 0
	}
	return b.chains[h].libs.count()
}

// Neighbors returns the on-board neighbors of p.
func (b *Board) Neighbors(p Point) []Point {
	return b.nb.of(p)
}

// XY converts a point to coordinates.
func (b *Board) XY(p Point) (x, y int) { return int(p) % b.size, int(p) / b.size }

// P converts coordinates to a point.
func (b *Board) P(x, y int) Point { return Point(y*b.size + x) }

// IsLegal reports whether the side to move may play m.
func (b *Board) IsLegal(m Point) bool {
	if m == Pass {
		return true
	}
	if !b.isLegalSimple(m) {
		return false
	}
	if b.superko {
		return !b.repeatsPosition(m)
	}
	return true
}

// isLegalSimple checks occupancy, simple ko and suicide using exact liberties.
func (b *Board) isLegalSimple(m Point) bool {
	if m < 0 || int(m) >= len(b.cells) || b.cells[m] != Empty || m == b.ko {
		return false
	}
	me := b.toPlay
	for _, q := range b.nb.of(m) {
		c := b.cells[q]
		if c == Empty {
			return true
		}
		libs := b.chains[b.chainOf[q]].libs.count()
		if c == me && libs > 1 {
			return true
		}
		if c != me && libs == 1 {
			return true // captures
		}
	}
	return false
}

// repeatsPosition reports whether playing m (already known to be legal under
// simple rules) recreates an earlier stone configuration. The resulting hash
// is derived from per-chain hashes, so no trial move is played.
func (b *Board) repeatsPosition(m Point) bool {
	me, opp := b.toPlay, b.toPlay.Opp()
	h := b.stones ^ zobristStone[m][me]
	var done [4]int16
	nd := 0
outer:
	for _, q := range b.nb.of(m) {
		if b.cells[q] != opp {
			continue
		}
		hq := b.chainOf[q]
		for _, d := range done[:nd] {
			if d == hq {
				continue outer
			}
		}
		done[nd] = hq
		nd++
		if b.chains[hq].libs.count() == 1 {
			h ^= b.chains[hq].hash
		}
	}
	for _, s := range b.seen {
		if s == h {
			return true
		}
	}
	return false
}

// Play applies a move for the side to move. It returns an error (and leaves
// the board unchanged) if the move is illegal.
func (b *Board) Play(m Point) error {
	if !b.IsLegal(m) {
		return fmt.Errorf("illegal move %s for %s", b.FormatPoint(m), b.toPlay)
	}
	b.play(m)
	if b.superko && m != Pass {
		b.seen = append(b.seen, b.stones)
	}
	return nil
}

// PlayUnchecked applies a move the caller has already verified with IsLegal.
func (b *Board) PlayUnchecked(m Point) {
	b.play(m)
	if b.superko && m != Pass {
		b.seen = append(b.seen, b.stones)
	}
}

func (b *Board) pushHistory(m Point) {
	copy(b.last[1:], b.last[:HistoryLen-1])
	b.last[0] = m
	b.moveNum++
	b.toPlay = b.toPlay.Opp()
}

func (b *Board) play(m Point) {
	if m == Pass {
		b.ko = Pass
		b.passes++
		b.pushHistory(Pass)
		return
	}
	me, opp := b.toPlay, b.toPlay.Opp()
	b.passes = 0
	b.cells[m] = me
	b.stones ^= zobristStone[m][me]

	// The new stone starts as its own chain with its empty neighbors as liberties.
	head := m
	ch := chain{size: 1, hash: zobristStone[m][me]}
	ch.stones.set(int(m))
	nbrs := b.nb.of(m)
	for _, q := range nbrs {
		if b.cells[q] == Empty {
			ch.libs.set(int(q))
		}
	}
	b.chains[head] = ch
	b.chainOf[m] = int16(head)

	// m is no longer a liberty of any adjacent chain; merge friendly chains.
	for _, q := range nbrs {
		if b.cells[q] == Empty {
			continue
		}
		hq := Point(b.chainOf[q])
		b.chains[hq].libs.clear(int(m))
		if b.cells[q] == me && hq != head {
			head = b.merge(head, hq)
		}
	}

	// Capture adjacent enemy chains left without liberties.
	captured, lastCapt := 0, Pass
	for _, q := range nbrs {
		if b.cells[q] != opp {
			continue
		}
		hq := Point(b.chainOf[q])
		if b.chains[hq].libs.isZero() {
			lastCapt = q
			captured += b.remove(hq)
		}
	}
	b.capt[me] += captured

	b.ko = Pass
	own := &b.chains[head]
	if captured == 1 && own.size == 1 && own.libs.count() == 1 {
		b.ko = lastCapt
	}
	b.pushHistory(m)
}

// merge joins chains a and b and returns the surviving head.
func (b *Board) merge(a, c Point) Point {
	ca, cc := &b.chains[a], &b.chains[c]
	if ca.size < cc.size {
		a, c = c, a
		ca, cc = cc, ca
	}
	cc.stones.forEach(func(s int) { b.chainOf[s] = int16(a) })
	ca.stones.or(&cc.stones)
	ca.libs.or(&cc.libs)
	ca.hash ^= cc.hash
	ca.size += cc.size
	return a
}

// remove takes a chain off the board and returns the stone count.
func (b *Board) remove(h Point) int {
	ch := b.chains[h]
	color := b.cells[h]
	ch.stones.forEach(func(s int) {
		b.cells[s] = Empty
		b.chainOf[s] = -1
		b.stones ^= zobristStone[s][color]
	})
	// Freed points become liberties of neighboring chains.
	ch.stones.forEach(func(s int) {
		for _, q := range b.nb.of(Point(s)) {
			if hq := b.chainOf[q]; hq >= 0 {
				b.chains[hq].libs.set(s)
			}
		}
	})
	return int(ch.size)
}

// LegalMoves appends every legal move (including Pass, last) to dst.
func (b *Board) LegalMoves(dst []Point) []Point {
	for p := Point(0); int(p) < len(b.cells); p++ {
		if b.IsLegal(p) {
			dst = append(dst, p)
		}
	}
	return append(dst, Pass)
}

// FormatPoint renders a point in GTP notation (A1 bottom-left, no I column).
func (b *Board) FormatPoint(p Point) string { return FormatPoint(p, b.size) }

// FormatPoint renders a point in GTP notation for the given board size.
func FormatPoint(p Point, size int) string {
	if p == Pass {
		return "pass"
	}
	x, y := int(p)%size, int(p)/size
	col := 'A' + rune(x)
	if col >= 'I' {
		col++
	}
	return fmt.Sprintf("%c%d", col, size-y)
}

// ParsePoint parses GTP notation ("D4", "pass").
func ParsePoint(s string, size int) (Point, error) {
	s = strings.ToUpper(strings.TrimSpace(s))
	if s == "PASS" {
		return Pass, nil
	}
	if len(s) < 2 || s[0] < 'A' || s[0] > 'Z' || s[0] == 'I' {
		return Pass, fmt.Errorf("invalid vertex %q", s)
	}
	x := int(s[0] - 'A')
	if s[0] > 'I' {
		x--
	}
	row := 0
	for _, r := range s[1:] {
		if r < '0' || r > '9' {
			return Pass, fmt.Errorf("invalid vertex %q", s)
		}
		row = row*10 + int(r-'0')
	}
	if x >= size || row < 1 || row > size {
		return Pass, fmt.Errorf("vertex %q off %dx%d board", s, size, size)
	}
	return Point((size-row)*size + x), nil
}

// String renders the board as text (X black, O white, last move in parens).
func (b *Board) String() string {
	var sb strings.Builder
	sb.WriteString("   ")
	for x := 0; x < b.size; x++ {
		col := 'A' + rune(x)
		if col >= 'I' {
			col++
		}
		sb.WriteRune(col)
		sb.WriteByte(' ')
	}
	sb.WriteByte('\n')
	last, _ := b.LastMove(0)
	for y := 0; y < b.size; y++ {
		fmt.Fprintf(&sb, "%2d ", b.size-y)
		for x := 0; x < b.size; x++ {
			p := b.P(x, y)
			ch := byte('.')
			switch b.cells[p] {
			case Black:
				ch = 'X'
			case White:
				ch = 'O'
			}
			sb.WriteByte(ch)
			if p == last {
				sb.WriteByte(')')
			} else {
				sb.WriteByte(' ')
			}
		}
		fmt.Fprintf(&sb, "%2d\n", b.size-y)
	}
	return strings.TrimRight(sb.String(), "\n")
}

// Setup places stones for an SGF/GTP setup position (AB/AW) and rebuilds
// chain state. It fails if any resulting chain has no liberties.
func (b *Board) Setup(black, white []Point) error {
	for _, p := range black {
		b.cells[p] = Black
	}
	for _, p := range white {
		b.cells[p] = White
	}
	b.rebuild()
	for p, c := range b.cells {
		if c != Empty && b.Libs(Point(p)) == 0 {
			return fmt.Errorf("setup leaves %s without liberties", b.FormatPoint(Point(p)))
		}
	}
	if b.superko {
		b.seen = append(b.seen[:0], b.stones)
	}
	return nil
}

// rebuild recomputes chains, liberties and the stone hash from cells.
func (b *Board) rebuild() {
	b.stones = 0
	for i := range b.chainOf {
		b.chainOf[i] = -1
	}
	for p, c := range b.cells {
		if c == Empty {
			continue
		}
		b.stones ^= zobristStone[p][c]
		if b.chainOf[p] >= 0 {
			continue
		}
		var ch chain
		stack := []Point{Point(p)}
		b.chainOf[p] = int16(p)
		for len(stack) > 0 {
			q := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			ch.stones.set(int(q))
			ch.hash ^= zobristStone[q][c]
			ch.size++
			for _, r := range b.nb.of(q) {
				switch b.cells[r] {
				case Empty:
					ch.libs.set(int(r))
				case c:
					if b.chainOf[r] < 0 {
						b.chainOf[r] = int16(p)
						stack = append(stack, r)
					}
				}
			}
		}
		b.chains[p] = ch
	}
	b.ko = Pass
}

// WithoutStones returns a copy with the given stones removed (for scoring
// after dead-stone agreement). Move history and side to move are kept.
func (b *Board) WithoutStones(pts []Point) *Board {
	c := b.Clone()
	for _, p := range pts {
		c.cells[p] = Empty
	}
	c.rebuild()
	return c
}
