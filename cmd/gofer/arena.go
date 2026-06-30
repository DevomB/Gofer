package main

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

// Arena stores nodes in a contiguous slice (ponytail: doubling growth).
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
