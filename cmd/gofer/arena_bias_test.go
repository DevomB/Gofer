package main

import "testing"

// TestArenaIdenticalNetsNoSystematicRoleBias verifies swap-colors balances role
// wins for equal strength. Stone colour is expected to skew here: this match uses
// tournament komi 6.5, which at 50 playouts favours White (Black wins about 29%);
// fair komi for these engines is near 0.5, and TestIdenticalEvalColorBalance
// asserts colour balance there. Before the ADR 0008 search fix this test logged
// stone_black=9 of 188 and the skew was the bug, not komi.
func TestArenaIdenticalNetsNoSystematicRoleBias(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	old := minGamesBeforePromote
	minGamesBeforePromote = 201
	t.Cleanup(func() { minGamesBeforePromote = old })

	const games = 200
	res := RunMatch(MatchConfig{
		Games: games, Size: 9, Komi: 6.5, Playouts: 50,
		BlackEval: "heuristic", WhiteEval: "heuristic2",
		Seed: 99, SwapColors: true, OpeningMoves: 8, OpeningTemp: 1.0, Parallel: 4,
	})
	decisive := res.WinsChallenger + res.WinsBaseline
	t.Logf("komi=6.5 challenger=%d baseline=%d draws=%d stone_black=%d stone_white=%d",
		res.WinsChallenger, res.WinsBaseline, res.Draws, res.WinsBlack, res.WinsWhite)
	if decisive == 0 {
		t.Fatal("no decisive games")
	}
	assertWinsNearExpected(t, res.WinsChallenger, decisive, 0.5, 3.0)
}
