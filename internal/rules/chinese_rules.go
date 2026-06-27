package rules

import "github.com/DevomB/gofer/internal/board"

// chineseRules implements Chinese rules: area scoring, suicide if capturing, simple ko.
type chineseRules struct{}

// LegalMoves returns all legal moves including pass.
func (r *chineseRules) LegalMoves(b *board.Board) []board.Move {
	size := b.Size()
	n := size * size
	ko := b.Ko()
	moves := make([]board.Move, 0, n+1)
	for idx := 0; idx < n; idx++ {
		if idx == ko || b.AtIndex(idx) != board.Empty {
			continue
		}
		if r.wouldBeLegal(b, idx) {
			moves = append(moves, board.StoneMove(board.PointFromIdx(size, idx)))
		}
	}
	moves = append(moves, board.PassMove)
	return moves
}

// Play applies m for the current player. Returns false if illegal.
func (r *chineseRules) Play(b *board.Board, m board.Move) bool {
	player := b.Player()
	if m.Pass {
		b.StartPlay(m, nil, -1, board.Empty)
		b.FinishTurn(-1)
		return true
	}
	idx := m.Point.Idx(b.Size())
	if idx < 0 || b.AtIndex(idx) != board.Empty || idx == b.Ko() {
		return false
	}
	if !r.wouldBeLegal(b, idx) {
		return false
	}
	prev := b.AtIndex(idx)
	trial := b.Clone()
	trial.SetStoneIndex(idx, player)
	captured := r.removeDeadGroups(trial, player.Opposite())
	if r.groupLibertyCount(trial, idx, player) == 0 {
		return false
	}
	b.StartPlay(m, captured, idx, prev)
	b.SetStoneIndex(idx, player)
	for _, cidx := range captured {
		b.SetStoneIndex(cidx, board.Empty)
	}
	newKo := -1
	// ponytail: ban immediate recapture when exactly one stone was removed.
	if len(captured) == 1 {
		newKo = captured[0]
	}
	b.FinishTurn(newKo)
	return true
}

// Score returns area scores; komi added to white.
// ponytail: seki neutral; no dead-stone removal pass.
// Ceiling: tournament Chinese may differ.
// Upgrade: two-pass scoring (M2).
func (r *chineseRules) Score(b *board.Board) (black, white float64) {
	size := b.Size()
	n := size * size
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case board.Black:
			black++
		case board.White:
			white++
		}
	}
	seen := make([]bool, n)
	for i := 0; i < n; i++ {
		if seen[i] || b.AtIndex(i) != board.Empty {
			continue
		}
		t, tb, tw := r.floodEmpty(b, i, seen)
		if tb && !tw {
			black += float64(t)
		} else if tw && !tb {
			white += float64(t)
		}
	}
	white += b.Komi()
	return black, white
}

func (r *chineseRules) wouldBeLegal(b *board.Board, idx int) bool {
	if idx == b.Ko() {
		return false
	}
	trial := b.Clone()
	player := trial.Player()
	trial.SetStoneIndex(idx, player)
	r.removeDeadGroups(trial, player.Opposite())
	return r.groupLibertyCount(trial, idx, player) > 0
}

func (r *chineseRules) removeDeadGroups(b *board.Board, color board.Color) []int {
	n := b.Size() * b.Size()
	var captured []int
	for i := 0; i < n; i++ {
		if b.AtIndex(i) != color {
			continue
		}
		if r.groupLibertyCount(b, i, color) == 0 {
			for _, g := range r.collectGroup(b, i, color) {
				b.SetStoneIndex(g, board.Empty)
				captured = append(captured, g)
			}
		}
	}
	return captured
}

func (r *chineseRules) groupLibertyCount(b *board.Board, start int, color board.Color) int {
	group := r.collectGroup(b, start, color)
	seen := make(map[int]struct{})
	libs := 0
	for _, g := range group {
		for _, nb := range b.Neighbors(g) {
			if b.AtIndex(nb) != board.Empty {
				continue
			}
			if _, ok := seen[nb]; ok {
				continue
			}
			seen[nb] = struct{}{}
			libs++
		}
	}
	return libs
}

func (r *chineseRules) collectGroup(b *board.Board, start int, color board.Color) []int {
	if b.AtIndex(start) != color {
		return nil
	}
	out := []int{start}
	stack := []int{start}
	seen := map[int]struct{}{start: {}}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, nb := range b.Neighbors(i) {
			if b.AtIndex(nb) != color {
				continue
			}
			if _, ok := seen[nb]; ok {
				continue
			}
			seen[nb] = struct{}{}
			out = append(out, nb)
			stack = append(stack, nb)
		}
	}
	return out
}

func (r *chineseRules) floodEmpty(b *board.Board, start int, seen []bool) (territory int, touchBlack, touchWhite bool) {
	stack := []int{start}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[i] {
			continue
		}
		seen[i] = true
		switch b.AtIndex(i) {
		case board.Empty:
			territory++
			for _, nb := range b.Neighbors(i) {
				if !seen[nb] {
					stack = append(stack, nb)
				}
			}
		case board.Black:
			touchBlack = true
		case board.White:
			touchWhite = true
		}
	}
	return territory, touchBlack, touchWhite
}
