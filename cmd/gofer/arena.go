package main

import "math"

// Node is an index-based MCTS tree node.
type Node struct {
	Parent   int
	Move     Move
	Children []int
	Visits   uint32
	ValueSum float64
	Prior    float64
	Expanded bool
}

// Arena stores nodes in a contiguous slice; doubles capacity when full.
type Arena struct {
	nodes []Node
}

// NewArena creates an empty arena.
func NewArena() *Arena {
	return &Arena{nodes: make([]Node, 0, 64)}
}

// Root allocates the root node and returns its index.
func (a *Arena) Root() int {
	if len(a.nodes) == 0 {
		a.nodes = append(a.nodes, Node{Parent: -1})
	}
	return 0
}

// Get returns node at index.
func (a *Arena) Get(i int) *Node {
	return &a.nodes[i]
}

// AddChild appends a child node and returns its index.
func (a *Arena) AddChild(parent int, m Move, prior float64) int {
	idx := len(a.nodes)
	a.nodes = append(a.nodes, Node{
		Parent: parent,
		Move:   m,
		Prior:  prior,
	})
	a.nodes[parent].Children = append(a.nodes[parent].Children, idx)
	return idx
}

// Mean returns average value for node visits.
func (n *Node) Mean() float64 {
	if n.Visits == 0 {
		return 0
	}
	return n.ValueSum / float64(n.Visits)
}

func puctScore(c *Node, parentVisits float64, isRoot bool, cfg SearchConfig) float64 {
	q := c.Mean()
	if c.Visits == 0 {
		q = -cfg.FPU
	}
	u := cfg.CPUCT * c.Prior * math.Sqrt(parentVisits) / (1 + float64(c.Visits))
	if isRoot && cfg.RootTemperature != 1 && c.Visits > 0 {
		q /= cfg.RootTemperature
	}
	return q + u
}

func (a *Arena) bestRootMove(root int) Move {
	n := a.Get(root)
	if len(n.Children) == 0 {
		return PassMove
	}
	best := n.Children[0]
	maxV := uint32(0)
	for _, cidx := range n.Children {
		c := a.Get(cidx)
		if c.Visits > maxV {
			maxV = c.Visits
			best = cidx
		}
	}
	return a.Get(best).Move
}
