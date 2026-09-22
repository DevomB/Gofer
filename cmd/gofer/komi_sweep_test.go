package main

import (
	"os"
	"testing"
)

// TestHeuristicFairKomiSweep logs stone-colour win rate against komi for equal
// heuristic engines, for recalibrating fair komi after engine changes. It asserts
// nothing and plays 500 games, so it runs only when asked;
// TestIdenticalEvalColorBalance checks the calibrated value.
func TestHeuristicFairKomiSweep(t *testing.T) {
	if os.Getenv("GOFER_KOMI_SWEEP") == "" {
		t.Skip("a measurement, not a check: set GOFER_KOMI_SWEEP=1 to run it")
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
