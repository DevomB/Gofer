package game

// Score returns Tromp-Taylor area scores: stones on the board plus empty
// regions that reach only one color. Komi is added to White. Stones are all
// treated as alive, so games must be played out (both sides capture dead
// stones before passing) for the result to be meaningful.
func (b *Board) Score() (black, white float64) {
	own := b.Ownership()
	for _, o := range own {
		switch o {
		case Black:
			black++
		case White:
			white++
		}
	}
	return black, white + b.komi
}

// Margin returns black's area score minus white's (komi included).
func (b *Board) Margin() float64 {
	bl, wl := b.Score()
	return bl - wl
}

// Winner returns the winning color under area scoring, or Empty for a draw.
func (b *Board) Winner() Color {
	switch m := b.Margin(); {
	case m > 0:
		return Black
	case m < 0:
		return White
	}
	return Empty
}

// Ownership returns, for every point, the color that owns it under area
// scoring (a stone, or territory bordered by one color) or Empty (dame/seki).
func (b *Board) Ownership() []Color {
	n := len(b.cells)
	own := make([]Color, n)
	seen := make([]bool, n)
	stack := make([]Point, 0, n)
	region := make([]Point, 0, n)
	for p := 0; p < n; p++ {
		if b.cells[p] != Empty {
			own[p] = b.cells[p]
			continue
		}
		if seen[p] {
			continue
		}
		region = region[:0]
		stack = append(stack[:0], Point(p))
		seen[p] = true
		touch := Empty
		mixed := false
		for len(stack) > 0 {
			q := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			region = append(region, q)
			for _, r := range b.nb.of(q) {
				switch c := b.cells[r]; c {
				case Empty:
					if !seen[r] {
						seen[r] = true
						stack = append(stack, r)
					}
				default:
					if touch == Empty {
						touch = c
					} else if touch != c {
						mixed = true
					}
				}
			}
		}
		if !mixed && touch != Empty {
			for _, q := range region {
				own[q] = touch
			}
		}
	}
	return own
}
