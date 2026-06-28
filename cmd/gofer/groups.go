package main

func collectGroup(b *Board, start int, color Color) []int {
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

func libertyCount(b *Board, start int, color Color) int {
	group := collectGroup(b, start, color)
	seen := make(map[int]struct{})
	libs := 0
	for _, g := range group {
		for _, nb := range b.Neighbors(g) {
			if b.AtIndex(nb) != Empty {
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

func removeDeadGroups(b *Board, color Color) []int {
	n := b.Size() * b.Size()
	var captured []int
	for i := 0; i < n; i++ {
		if b.AtIndex(i) != color {
			continue
		}
		if libertyCount(b, i, color) == 0 {
			for _, g := range collectGroup(b, i, color) {
				b.SetStoneIndex(g, Empty)
				captured = append(captured, g)
			}
		}
	}
	return captured
}

func floodEmpty(b *Board, start int, seen []bool) (territory int, touchBlack, touchWhite bool) {
	stack := []int{start}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[i] {
			continue
		}
		seen[i] = true
		switch b.AtIndex(i) {
		case Empty:
			territory++
			for _, nb := range b.Neighbors(i) {
				if !seen[nb] {
					stack = append(stack, nb)
				}
			}
		case Black:
			touchBlack = true
		case White:
			touchWhite = true
		}
	}
	return territory, touchBlack, touchWhite
}
