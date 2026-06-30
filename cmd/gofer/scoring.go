package main

// OwnershipLabel returns per-intersection training labels: +1 black, -1 white, 0 neutral/seki.
// Area-flood ownership labels; full Benson pass-alive not implemented.
// Upgrade: Benson pass-alive stone marking before territory flood.
func OwnershipLabel(b *Board) []float32 {
	n := b.Size() * b.Size()
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case Black:
			out[i] = 1
		case White:
			out[i] = -1
		}
	}
	seen := make([]bool, n)
	for i := 0; i < n; i++ {
		if b.AtIndex(i) != Empty || seen[i] {
			continue
		}
		indices := emptyRegionIndices(b, i)
		touchB, touchW := false, false
		for _, idx := range indices {
			seen[idx] = true
			b.forEachNeighbor(idx, func(nb int) {
				switch b.AtIndex(nb) {
				case Black:
					touchB = true
				case White:
					touchW = true
				}
			})
		}
		val := float32(0)
		if touchB && !touchW {
			val = 1
		} else if touchW && !touchB {
			val = -1
		}
		for _, idx := range indices {
			out[idx] = val
		}
	}
	return out
}

func emptyRegionIndices(b *Board, start int) []int {
	n := b.Size() * b.Size()
	seen := make([]bool, n)
	var out []int
	stack := []int{start}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[i] || b.AtIndex(i) != Empty {
			continue
		}
		seen[i] = true
		out = append(out, i)
		b.forEachNeighbor(i, func(nb int) {
			if b.AtIndex(nb) == Empty && !seen[nb] {
				stack = append(stack, nb)
			}
		})
	}
	return out
}
