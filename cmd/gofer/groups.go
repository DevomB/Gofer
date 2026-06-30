package main

// visitMark tracks flood-fill visits without per-call map allocations.
// visitMark gen overflow clears the slice (rare).
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
	v.cur++
	if v.cur == 0 {
		clear(v.gen)
		v.cur = 1
	}
}

func (v *visitMark) bump() {
	v.cur++
	if v.cur == 0 {
		clear(v.gen)
		v.cur = 1
	}
}

func (v *visitMark) mark(i int)   { v.gen[i] = v.cur }
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
	stack := []int{start}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[i] {
			continue
		}
		seen[i] = true
		switch b.AtIndex(i) {
		case Empty:
			territory++
			for _, nb := range b.Neighbors(i) {
				if !seen[nb] {
					stack = append(stack, nb)
				}
			}
		case Black:
			touchBlack = true
		case White:
			touchWhite = true
		}
	}
	return territory, touchBlack, touchWhite
}
