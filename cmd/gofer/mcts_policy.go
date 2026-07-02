package main

import "sort"

func (e *Engine) analyzeLocked(topN, playouts int) Analysis {
	out := Analysis{Playouts: playouts}
	if e.arena == nil || len(e.arena.nodes) == 0 {
		out.Best = PassMove
		return out
	}
	root := e.arena.Get(e.root)
	out.RootValue = root.Mean()
	out.Candidates = rootCandidates(e.arena, e.root, topN)
	if len(out.Candidates) > 0 {
		out.Best = out.Candidates[0].Move
	} else {
		out.Best = PassMove
	}
	out.PV = e.pvLocked(8)
	return out
}

func rootCandidates(a *Arena, root, topN int) []MoveCandidate {
	n := a.Get(root)
	if len(n.Children) == 0 {
		return nil
	}
	total := rootVisitTotal(a, n)
	cands := sortedRootCandidates(a, n)
	applyCandidateShares(cands, total)
	if topN <= 0 {
		topN = 5
	}
	if len(cands) > topN {
		cands = cands[:topN]
	}
	return cands
}

func rootVisitTotal(a *Arena, n *Node) uint32 {
	var total uint32
	for _, cidx := range n.Children {
		total += a.Get(cidx).Visits
	}
	return total
}

func sortedRootCandidates(a *Arena, n *Node) []MoveCandidate {
	cands := make([]MoveCandidate, 0, len(n.Children))
	for _, cidx := range n.Children {
		c := a.Get(cidx)
		cands = append(cands, MoveCandidate{Move: c.Move, Visits: c.Visits, WinRate: c.Mean()})
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Visits != cands[j].Visits {
			return cands[i].Visits > cands[j].Visits
		}
		return cands[i].WinRate > cands[j].WinRate
	})
	return cands
}

func applyCandidateShares(cands []MoveCandidate, total uint32) {
	if total == 0 {
		return
	}
	inv := 1 / float64(total)
	for i := range cands {
		cands[i].Share = float64(cands[i].Visits) * inv
	}
}

func (e *Engine) pvLocked(maxLen int) []Move {
	if e.arena == nil {
		return nil
	}
	pv := make([]Move, 0, maxLen)
	node := e.root
	for len(pv) < maxLen {
		next, ok := e.bestPVChild(node)
		if !ok {
			break
		}
		pv = append(pv, e.arena.Get(next).Move)
		node = next
	}
	return pv
}

func (e *Engine) bestPVChild(node int) (int, bool) {
	n := e.arena.Get(node)
	if len(n.Children) == 0 {
		return 0, false
	}
	best := n.Children[0]
	maxV := e.arena.Get(best).Visits
	for _, cidx := range n.Children[1:] {
		if visits := e.arena.Get(cidx).Visits; visits > maxV {
			maxV = visits
			best = cidx
		}
	}
	return best, true
}

// RootPolicy returns visit-weighted policy over legal moves.
func (e *Engine) RootPolicy(legal []Move) []float32 {
	pi := make([]float32, len(legal))
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.arena == nil || len(e.arena.nodes) == 0 {
		return uniformPolicy32(len(legal))
	}
	root := e.arena.Get(e.root)
	total := rootVisitTotal(e.arena, root)
	if total == 0 {
		return uniformPolicy32(len(legal))
	}
	fillLegalPolicy(pi, legal, root, e.arena, total)
	return pi
}

func fillLegalPolicy(pi []float32, legal []Move, root *Node, arena *Arena, total uint32) {
	inv := 1 / float32(total)
	for _, cidx := range root.Children {
		c := arena.Get(cidx)
		for i, m := range legal {
			if movesEqual(c.Move, m) {
				pi[i] = float32(c.Visits) * inv
				break
			}
		}
	}
}

// RootPolicyPruned returns visit-weighted policy with low-visit moves zeroed (paper SE-4.3).
func (e *Engine) RootPolicyPruned(legal []Move) []float32 {
	pi := e.RootPolicy(legal)
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.arena == nil {
		return pi
	}
	root := e.arena.Get(e.root)
	if !rootHasStrongChild(e.arena, root) {
		return pi
	}
	return prunePolicy(pi, legal, root, e)
}

func rootHasStrongChild(a *Arena, root *Node) bool {
	for _, cidx := range root.Children {
		if a.Get(cidx).Visits >= minPolicyVisits {
			return true
		}
	}
	return false
}

func prunePolicy(pi []float32, legal []Move, root *Node, e *Engine) []float32 {
	sum := float32(0)
	for i, m := range legal {
		if e.rootChildVisitsLocked(root, m) < minPolicyVisits {
			pi[i] = 0
		}
		sum += pi[i]
	}
	if sum == 0 {
		return uniformPolicy32(len(legal))
	}
	inv := 1 / sum
	for i := range pi {
		pi[i] *= inv
	}
	return pi
}

// RootPolicyBoard returns visit-weighted policy over board indices (size² + pass).
func (e *Engine) RootPolicyBoard(b *Board, legal []Move) []float32 {
	legalPi := e.RootPolicyPruned(legal)
	n := b.Size()*b.Size() + 1
	out := make([]float32, n)
	for i, m := range legal {
		idx := policyIndex(m, b.Size())
		if idx >= 0 && idx < n {
			out[idx] = legalPi[i]
		}
	}
	return out
}

func policyIndex(m Move, size int) int {
	if m.Pass {
		return size * size
	}
	return m.Point.Idx(size)
}

func (e *Engine) rootChildVisitsLocked(root *Node, m Move) uint32 {
	for _, cidx := range root.Children {
		c := e.arena.Get(cidx)
		if movesEqual(c.Move, m) {
			return c.Visits
		}
	}
	return 0
}
