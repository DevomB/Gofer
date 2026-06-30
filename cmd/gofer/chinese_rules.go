package main

// chineseRules implements Chinese rules: area scoring, suicide if capturing, simple ko.
type chineseRules struct{}

// LegalMoves returns all legal moves including pass.
func (r *chineseRules) LegalMoves(b *Board) []Move {
	size := b.Size()
	n := size * size
	ko := b.Ko()
	var scratch legalityScratch
	scratch.prepare(b)
	moves := make([]Move, 0, n+1)
	for idx := 0; idx < n; idx++ {
		if idx == ko || b.AtIndex(idx) != Empty {
			continue
		}
		scratch.trial.restoreTrialSnap(scratch.snap)
		scratch.mark.ensure(n)
		if r.wouldBeLegalTrialScratch(scratch.trial, idx, &scratch) {
			moves = append(moves, StoneMove(PointFromIdx(size, idx)))
		}
	}
	moves = append(moves, PassMove)
	return moves
}

// Play applies m for the current player. Returns false if illegal.
func (r *chineseRules) Play(b *Board, m Move) bool {
	player := b.Player()
	if m.Pass {
		b.StartPlay(m, nil, -1, Empty)
		b.FinishTurn(-1)
		return true
	}
	idx := m.Point.Idx(b.Size())
	if idx < 0 || b.AtIndex(idx) != Empty || idx == b.Ko() {
		return false
	}
	if !r.wouldBeLegal(b, idx) {
		return false
	}
	prev := b.AtIndex(idx)
	trial := b.Clone()
	trial.SetStoneIndex(idx, player)
	captured := removeDeadGroups(trial, player.Opposite())
	if libertyCount(trial, idx, player) == 0 {
		return false
	}
	b.StartPlay(m, captured, idx, prev)
	b.SetStoneIndex(idx, player)
	for _, cidx := range captured {
		b.SetStoneIndex(cidx, Empty)
	}
	newKo := -1
	if len(captured) == 1 {
		newKo = captured[0]
	}
	b.FinishTurn(newKo)
	return true
}

// Score returns area scores; komi added to white.
// Seki neutral; no dead-stone removal pass.
// Ceiling: tournament Chinese may differ.
// Upgrade: two-pass dead-stone scoring.
func (r *chineseRules) Score(b *Board) (black, white float64) {
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

func (r *chineseRules) wouldBeLegal(b *Board, idx int) bool {
	if idx == b.Ko() {
		return false
	}
	trial := b.Clone()
	return r.wouldBeLegalTrial(trial, idx)
}

func (r *chineseRules) wouldBeLegalTrial(trial *Board, idx int) bool {
	var scratch legalityScratch
	n := trial.Size() * trial.Size()
	scratch.trial = trial
	if cap(scratch.groupBuf) < n {
		scratch.groupBuf = make([]int, 0, n)
		scratch.stackBuf = make([]int, 0, n)
	}
	scratch.mark.ensure(n)
	return r.wouldBeLegalTrialScratch(trial, idx, &scratch)
}

func (r *chineseRules) wouldBeLegalTrialScratch(trial *Board, idx int, scratch *legalityScratch) bool {
	player := trial.Player()
	trial.SetStoneIndex(idx, player)
	removeDeadGroupsMark(trial, player.Opposite(), &scratch.mark, &scratch.groupBuf, &scratch.stackBuf)
	return libertyCountMark(trial, idx, player, &scratch.mark, &scratch.groupBuf, &scratch.stackBuf) > 0
}
