package main

type trompRules struct {
	seen map[uint64]struct{}
}

func newTrompRules() *trompRules {
	return &trompRules{seen: map[uint64]struct{}{}}
}

func (r *trompRules) LegalMoves(b *Board) []Move {
	r.ensureStart(b)
	size := b.Size()
	n := size * size
	moves := make([]Move, 0, n+1)
	for idx := 0; idx < n; idx++ {
		if b.AtIndex(idx) != Empty {
			continue
		}
		if r.wouldBeLegal(b, idx) {
			moves = append(moves, StoneMove(PointFromIdx(size, idx)))
		}
	}
	moves = append(moves, PassMove)
	return moves
}

func (r *trompRules) Play(b *Board, m Move) bool {
	r.ensureStart(b)
	if m.Pass {
		b.StartPlay(m, nil, -1, Empty)
		b.FinishTurn(-1)
		pos := trompPositionHash(b)
		if r.repeats(pos) {
			b.Undo()
			return false
		}
		r.record(pos)
		return true
	}
	idx := m.Point.Idx(b.Size())
	if idx < 0 || b.AtIndex(idx) != Empty {
		return false
	}
	player := b.Player()
	prev := b.AtIndex(idx)
	trial := b.Clone()
	trial.SetStoneIndex(idx, player)
	captured := removeDeadGroups(trial, player.Opposite())
	captured = append(captured, removeDeadGroups(trial, player)...)
	trial.FinishTurn(-1)
	pos := trompPositionHash(trial)
	if r.repeats(pos) {
		return false
	}
	b.StartPlay(m, captured, idx, prev)
	b.SetStoneIndex(idx, player)
	for _, cidx := range captured {
		b.SetStoneIndex(cidx, Empty)
	}
	b.FinishTurn(-1)
	r.record(pos)
	return true
}

// Score returns Tromp-Taylor area scores (komi to white).
// ponytail: all on-board stones alive; territory via empty-region flood.
// Ceiling: no Benson pass-alive removal.
// Upgrade: Benson pass-alive (M3 backlog-core-engine).
func (r *trompRules) Score(b *Board) (black, white float64) {
	size := b.Size()
	n := size * size
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case Black:
			black++
		case White:
			white++
		}
	}
	seen := make([]bool, n)
	for i := 0; i < n; i++ {
		if seen[i] || b.AtIndex(i) != Empty {
			continue
		}
		t, tb, tw := floodEmpty(b, i, seen)
		if tb && !tw {
			black += float64(t)
		} else if tw && !tb {
			white += float64(t)
		}
	}
	white += b.Komi()
	return black, white
}

func (r *trompRules) ensureStart(b *Board) {
	if len(r.seen) == 0 {
		r.record(trompPositionHash(b))
	}
}

func (r *trompRules) wouldBeLegal(b *Board, idx int) bool {
	player := b.Player()
	trial := b.Clone()
	trial.SetStoneIndex(idx, player)
	removeDeadGroups(trial, player.Opposite())
	removeDeadGroups(trial, player)
	trial.FinishTurn(-1)
	return !r.repeats(trompPositionHash(trial))
}

func (r *trompRules) repeats(h uint64) bool {
	_, ok := r.seen[h]
	return ok
}

func (r *trompRules) record(h uint64) {
	r.seen[h] = struct{}{}
}

func trompPositionHash(b *Board) uint64 {
	h := b.Hash()
	if b.Player() == White {
		h ^= 0x9e3779b97f4a7c15
	}
	return h
}
