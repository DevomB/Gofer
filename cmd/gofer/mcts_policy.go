package main

import (
	"math"
	"sort"
)

func (e *Engine) analyzeLocked(topN, playouts int) Analysis {
	out := Analysis{Playouts: playouts}
	if e.arena == nil || e.arena.Len() == 0 {
		out.Best = PassMove
		return out
	}
	root := e.arena.Get(e.root)
	out.RootValue = root.Mean()
	out.Candidates = rootCandidates(e.arena, e.root, topN, e.cfg)
	if len(out.Candidates) > 0 {
		out.Best = out.Candidates[0].Move
	} else {
		out.Best = PassMove
	}
	out.PV = e.pvLocked(8)
	return out
}

func rootCandidates(a *Arena, root, topN int, cfg SearchConfig) []MoveCandidate {
	n := a.Get(root)
	if len(n.Children) == 0 {
		return nil
	}
	total := rootVisitTotal(a, n)
	cands := sortedRootCandidates(a, n, cfg)
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

// sortedRootCandidates reports values from the root player's perspective.
func sortedRootCandidates(a *Arena, n *Node, cfg SearchConfig) []MoveCandidate {
	parentVisits := float64(n.Visits)
	fpu := fpuFor(a, n, true, cfg)
	cands := make([]MoveCandidate, 0, len(n.Children))
	for _, cidx := range n.Children {
		c := a.Get(cidx)
		cands = append(cands, MoveCandidate{
			Move:   c.Move,
			Visits: c.Visits,
			Value:  -c.Mean(),
			Prior:  c.Prior,
			PUCT:   puctScore(c, parentVisits, fpu, cfg),
		})
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Visits != cands[j].Visits {
			return cands[i].Visits > cands[j].Visits
		}
		return cands[i].Value > cands[j].Value
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
	if e.arena == nil || e.arena.Len() == 0 {
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

// RootPolicyPruned returns the root visit distribution with forced exploration
// taken back out (Wu 2020 §3.2), which is what the policy is trained on.
func (e *Engine) RootPolicyPruned(legal []Move) []float32 {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.arena == nil || e.arena.Len() == 0 {
		return uniformPolicy32(len(legal))
	}
	root := e.arena.Get(e.root)
	visits := prunedRootVisits(e.arena, root, e.cfg)
	pi := make([]float32, len(legal))
	sum := 0.0
	for i, cidx := range root.Children {
		m := e.arena.Get(cidx).Move
		for j, l := range legal {
			if movesEqual(m, l) {
				pi[j] = float32(visits[i])
				sum += visits[i]
				break
			}
		}
	}
	if sum == 0 {
		return uniformPolicy32(len(legal))
	}
	for j := range pi {
		pi[j] = float32(float64(pi[j]) / sum)
	}
	return pi
}

// prunedRootVisits takes up to its forced playouts from every child but the most
// visited, stopping before the child's PUCT would reach the best child's: visits
// the child earned on its value stay. Children left with one visit are dropped.
func prunedRootVisits(a *Arena, root *Node, cfg SearchConfig) []float64 {
	visits := make([]float64, len(root.Children))
	best := 0
	for i, cidx := range root.Children {
		visits[i] = float64(a.Get(cidx).Visits)
		if visits[i] > visits[best] {
			best = i
		}
	}
	if len(visits) == 0 || visits[best] == 0 {
		return visits
	}
	total := float64(rootVisitTotal(a, root))
	sqrtParent := math.Sqrt(float64(root.Visits))
	bestPUCT := puctScore(a.Get(root.Children[best]), float64(root.Visits), 0, cfg)
	for i, cidx := range root.Children {
		if i == best || visits[i] == 0 {
			continue
		}
		c := a.Get(cidx)
		if gap := bestPUCT + c.Mean(); cfg.ForcedRootPlayouts > 0 && gap > 0 {
			// PUCT = Q + CPUCT*P*sqrt(N)/(1+n) stays at or below bestPUCT down to this n.
			floor := cfg.CPUCT*c.Prior*sqrtParent/gap - 1
			reduced := visits[i] - forcedPlayouts(cfg.ForcedRootPlayouts, c.Prior, total)
			visits[i] = math.Min(visits[i], math.Max(reduced, floor))
		}
		if visits[i] <= 1 {
			visits[i] = 0
		}
	}
	return visits
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
