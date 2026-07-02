package main

// Gating constants — keep in sync with scripts/gating.env.
const (
	// PromoteMin is the head-to-head win rate a challenger must clear to be
	// flagged promoted (candidate vs current champion). The loop additionally
	// requires the Wilson CI lower bound above 0.5; there is no win-target stop.
	PromoteMin = 0.55
)
