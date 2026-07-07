package main

import "testing"

// TestHeuristicFairKomiSweep documents stone-color win rate vs komi for equal heuristic
// engines (same eval name — swap does not alternate stone color). Fair komi for
// stone-color balance at 50 playouts is near 0..-1, not tournament 6.5.
func TestHeuristicFairKomiSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	old := minGamesBeforePromote
	minGamesBeforePromote = 201
	t.Cleanup(func() { minGamesBeforePromote = old })

	const games = 100
	for _, komi := range []float64{-2, -1, 0, 3.5, 6.5} {
		res := RunMatch(MatchConfig{
			Games: games, Size: 9, Komi: komi, Playouts: 50,
			BlackEval: "heuristic", WhiteEval: "heuristic",
			Seed: 99, SwapColors: true, OpeningMoves: 8, OpeningTemp: 1.0, Parallel: 4,
		})
		t.Logf("komi=%.1f stone_black=%d stone_white=%d draws=%d", komi, res.WinsBlack, res.WinsWhite, res.Draws)
	}
}
