package main

// WithSuperko wraps r with positional superko using board Zobrist hash + side to move.
func WithSuperko(r Ruleset) Ruleset {
	return &superkoRules{inner: r, seen: map[uint64]struct{}{}}
}

type superkoRules struct {
	inner Ruleset
	seen  map[uint64]struct{}
}

func (s *superkoRules) LegalMoves(b *Board) []Move {
	s.ensureStart(b)
	all := s.inner.LegalMoves(b)
	out := make([]Move, 0, len(all))
	for _, m := range all {
		if m.Pass {
			out = append(out, m)
			continue
		}
		if s.trialLegal(b, m) {
			out = append(out, m)
		}
	}
	return out
}

func (s *superkoRules) Play(b *Board, m Move) bool {
	s.ensureStart(b)
	if m.Pass {
		if s.repeats(superkoHash(b)) {
			return false
		}
		if !s.inner.Play(b, m) {
			return false
		}
		s.record(superkoHash(b))
		return true
	}
	if !s.trialLegal(b, m) {
		return false
	}
	if !s.inner.Play(b, m) {
		return false
	}
	s.record(superkoHash(b))
	return true
}

func (s *superkoRules) Score(b *Board) (black, white float64) {
	return s.inner.Score(b)
}

func (s *superkoRules) trialLegal(b *Board, m Move) bool {
	trial := b.Clone()
	if !s.inner.Play(trial, m) {
		return false
	}
	return !s.repeats(superkoHash(trial))
}

func (s *superkoRules) repeats(h uint64) bool {
	_, ok := s.seen[h]
	return ok
}

func (s *superkoRules) record(h uint64) {
	s.seen[h] = struct{}{}
}

func superkoHash(b *Board) uint64 {
	h := b.Hash()
	if b.Player() == White {
		h ^= 0x9e3779b97f4a7c15
	}
	return h
}

func (s *superkoRules) ensureStart(b *Board) {
	if len(s.seen) == 0 {
		s.record(superkoHash(b))
	}
}
