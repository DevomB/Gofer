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

// arenaChunk is the number of nodes per allocation. Chunks are never resized, so
// a *Node handed out by Get stays valid for the life of the arena.
const arenaChunk = 1024

// Arena stores nodes in fixed-size chunks.
//
// It used to be one slice that AddChild appended to, which meant Get returned a
// pointer into memory that the next expansion could reallocate. Any pointer held
// across an AddChild was then silently writing into a discarded array. That cost
// us: the root's "expanded" flag was lost on every 9x9 search (82 children
// against a capacity of 64), the search never descended, and no test caught it
// because every test board has fewer children than the initial capacity. Chunked
// storage makes the whole class of mistake impossible instead of moving the
// threshold at which it appears.
type Arena struct {
	chunks [][]Node
	n      int
}

// NewArena creates an empty arena.
func NewArena() *Arena {
	return &Arena{}
}

// Root allocates the root node and returns its index.
func (a *Arena) Root() int {
	if a.n == 0 {
		a.alloc(Node{Parent: -1})
	}
	return 0
}

// Len returns the number of allocated nodes.
func (a *Arena) Len() int { return a.n }

// Get returns the node at index i. The pointer stays valid across later AddChild
// calls, because chunks are allocated once and never grown.
func (a *Arena) Get(i int) *Node {
	return &a.chunks[i/arenaChunk][i%arenaChunk]
}

func (a *Arena) alloc(n Node) int {
	idx := a.n
	if idx/arenaChunk == len(a.chunks) {
		a.chunks = append(a.chunks, make([]Node, arenaChunk))
	}
	*a.Get(idx) = n
	a.n++
	return idx
}

// AddChild appends a child node and returns its index.
func (a *Arena) AddChild(parent int, m Move, prior float64) int {
	idx := a.alloc(Node{
		Parent: parent,
		Move:   m,
		Prior:  prior,
	})
	p := a.Get(parent)
	p.Children = append(p.Children, idx)
	return idx
}

// Mean returns average value for node visits.
func (n *Node) Mean() float64 {
	if n.Visits == 0 {
		return 0
	}
	return n.ValueSum / float64(n.Visits)
}

func puctScore(c *Node, parentVisits float64, isRoot bool, fpu float64, cfg SearchConfig) float64 {
	// Node means are stored from that node's own side to move (backupLocked flips
	// the sign at every level), so a child's mean is the OPPONENT's view. Negate
	// it to score the move from the parent's side: without this the search picks
	// the move that is best for the opponent, and more playouts make it weaker.
	q := -c.Mean()
	if c.Visits == 0 {
		q = fpu // first-play urgency, in the parent's frame (see selectChildLocked)
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
