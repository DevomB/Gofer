package main

// visitMark tracks flood-fill visits without per-call map allocations: a point
// is visited when its generation matches the current one, so starting a new
// fill is an increment rather than a clear. The slice is only cleared when the
// counter wraps.
type visitMark struct {
	gen []uint32
	cur uint32
}

func (v *visitMark) ensure(n int) {
	if len(v.gen) < n {
		v.gen = make([]uint32, n)
		v.cur = 1
		return
	}
	v.bump()
}

func (v *visitMark) bump() {
	v.cur++
	if v.cur == 0 {
		clear(v.gen)
		v.cur = 1
	}
}

func (v *visitMark) mark(i int)      { v.gen[i] = v.cur }
func (v *visitMark) seen(i int) bool { return v.gen[i] == v.cur }

// legalityScratch holds reusable buffers for LegalMoves hot path.
type legalityScratch struct {
	trial    *Board
	snap     trialSnap
	mark     visitMark
	groupBuf []int
	stackBuf []int
}

func (s *legalityScratch) prepare(b *Board) {
	if s.trial == nil || s.trial.Size() != b.Size() {
		s.trial = b.cloneTrial()
	} else {
		s.trial.restoreTrialSnap(s.snap)
	}
	s.snap = b.captureTrialSnap()
	n := b.Size() * b.Size()
	if cap(s.groupBuf) < n {
		s.groupBuf = make([]int, 0, n)
		s.stackBuf = make([]int, 0, n)
	}
	s.groupBuf = s.groupBuf[:0]
	s.stackBuf = s.stackBuf[:0]
	s.mark.ensure(n)
}

func collectGroup(b *Board, start int, color Color) []int {
	var mark visitMark
	return collectGroupMark(b, start, color, &mark, nil, nil)
}

func collectGroupMark(b *Board, start int, color Color, mark *visitMark, groupBuf, stackBuf *[]int) []int {
	if b.AtIndex(start) != color {
		return nil
	}
	n := b.Size() * b.Size()
	mark.ensure(n)
	if groupBuf != nil {
		*groupBuf = (*groupBuf)[:0]
		*stackBuf = (*stackBuf)[:0]
		*groupBuf = append(*groupBuf, start)
		*stackBuf = append(*stackBuf, start)
	} else {
		groupBuf = new([]int)
		stackBuf = new([]int)
		*groupBuf = append(*groupBuf, start)
		*stackBuf = append(*stackBuf, start)
	}
	mark.mark(start)
	for len(*stackBuf) > 0 {
		i := (*stackBuf)[len(*stackBuf)-1]
		*stackBuf = (*stackBuf)[:len(*stackBuf)-1]
		b.forEachNeighbor(i, func(nb int) {
			if b.AtIndex(nb) != color || mark.seen(nb) {
				return
			}
			mark.mark(nb)
			*groupBuf = append(*groupBuf, nb)
			*stackBuf = append(*stackBuf, nb)
		})
	}
	return *groupBuf
}

func libertyCount(b *Board, start int, color Color) int {
	var mark visitMark
	return libertyCountMark(b, start, color, &mark, nil, nil)
}

func libertyCountMark(b *Board, start int, color Color, mark *visitMark, groupBuf, stackBuf *[]int) int {
	group := collectGroupMark(b, start, color, mark, groupBuf, stackBuf)
	mark.bump()
	libs := 0
	for _, g := range group {
		b.forEachNeighbor(g, func(nb int) {
			if b.AtIndex(nb) != Empty || mark.seen(nb) {
				return
			}
			mark.mark(nb)
			libs++
		})
	}
	return libs
}

// groupLabels numbers each stone group once, with its size and liberties, so a
// question about any point's neighbours is a lookup rather than a flood.
type groupLabels struct {
	id   []int // group of each stone; -1 on empty points
	size []int
	libs []int
}

func labelGroups(b *Board) groupLabels {
	n := b.Size() * b.Size()
	g := groupLabels{id: make([]int, n)}
	for i := range g.id {
		g.id[i] = -1
	}
	var mark visitMark
	for i := 0; i < n; i++ {
		c := b.AtIndex(i)
		if c == Empty || g.id[i] >= 0 {
			continue
		}
		stones := collectGroupMark(b, i, c, &mark, nil, nil)
		mark.bump()
		libs := 0
		for _, s := range stones {
			g.id[s] = len(g.size)
			b.forEachNeighbor(s, func(nb int) {
				if b.AtIndex(nb) == Empty && !mark.seen(nb) {
					mark.mark(nb)
					libs++
				}
			})
		}
		g.size = append(g.size, len(stones))
		g.libs = append(g.libs, libs)
	}
	return g
}

// libertiesAfter counts the liberties of player's group at idx once a stone is
// played there and whatever it captures is removed.
func libertiesAfter(b *Board, idx int, player Color) int {
	trial := b.cloneTrial()
	trial.SetStoneIndex(idx, player)
	removeDeadGroups(trial, player.Opposite())
	return libertyCount(trial, idx, player)
}

func removeDeadGroups(b *Board, color Color) []int {
	var mark visitMark
	return removeDeadGroupsMark(b, color, &mark, nil, nil)
}

func removeDeadGroupsMark(b *Board, color Color, mark *visitMark, groupBuf, stackBuf *[]int) []int {
	n := b.Size() * b.Size()
	var captured []int
	for i := 0; i < n; i++ {
		if b.AtIndex(i) != color {
			continue
		}
		if libertyCountMark(b, i, color, mark, groupBuf, stackBuf) == 0 {
			for _, g := range collectGroupMark(b, i, color, mark, groupBuf, stackBuf) {
				b.SetStoneIndex(g, Empty)
				captured = append(captured, g)
			}
		}
	}
	return captured
}

func floodEmpty(b *Board, start int, seen []bool) (territory int, touchBlack, touchWhite bool) {
	// Only empty points are marked: seen is shared across every region on the
	// board, and a stone bordering two regions has to count for both.
	seen[start] = true
	stack := []int{start}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		territory++
		b.forEachNeighbor(i, func(nb int) {
			switch b.AtIndex(nb) {
			case Empty:
				if !seen[nb] {
					seen[nb] = true
					stack = append(stack, nb)
				}
			case Black:
				touchBlack = true
			case White:
				touchWhite = true
			}
		})
	}
	return territory, touchBlack, touchWhite
}

// ownEye reports whether idx is an obvious eye of player: every orthogonal
// neighbour is player's, and the opponent holds fewer than two diagonals,
// counting the edge as one. Filling such a point never captures and only takes
// a liberty away. A point failing the diagonal test is a false eye, where
// filling may be the connection that saves a group, so it does not count.
func ownEye(b *Board, idx int, player Color) bool {
	surrounded := true
	b.forEachNeighbor(idx, func(nb int) {
		if b.AtIndex(nb) != player {
			surrounded = false
		}
	})
	if !surrounded {
		return false
	}
	size := b.Size()
	x, y := idx%size, idx/size
	opp, edge := 0, false
	for _, d := range [4][2]int{{-1, -1}, {1, -1}, {-1, 1}, {1, 1}} {
		dx, dy := x+d[0], y+d[1]
		if dx < 0 || dy < 0 || dx >= size || dy >= size {
			edge = true
		} else if b.AtIndex(dy*size+dx) == player.Opposite() {
			opp++
		}
	}
	if edge {
		opp++
	}
	return opp < 2
}
