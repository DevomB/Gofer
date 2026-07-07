package main

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestWilsonCI(t *testing.T) {
	low, high := WilsonCI(55, 100, 1.96)
	if low > 0.55 || high < 0.45 {
		t.Fatalf("unexpected CI [%f,%f] for 55/100", low, high)
	}
	low0, high0 := WilsonCI(0, 0, 1.96)
	if low0 != 0 || high0 != 1 {
		t.Fatalf("empty CI want [0,1] got [%f,%f]", low0, high0)
	}
}

func TestPromotionGateDecided(t *testing.T) {
	accept, reject := promotionGateDecided(120, 200, 200, PromoteMin)
	if !accept || reject {
		t.Fatalf("120/200 should accept got accept=%v reject=%v", accept, reject)
	}
	accept, reject = promotionGateDecided(70, 200, 200, PromoteMin)
	if accept || !reject {
		t.Fatalf("70/200 should reject got accept=%v reject=%v", accept, reject)
	}
	accept, reject = promotionGateDecided(55, 100, 200, PromoteMin)
	if accept || reject {
		t.Fatalf("55/100 mid-match should continue got accept=%v reject=%v", accept, reject)
	}
	// Best-case finish still below promote bar.
	accept, reject = promotionGateDecided(50, 160, 200, PromoteMin)
	if accept || !reject {
		t.Fatalf("50/160 capped at 90/200 should reject got accept=%v reject=%v", accept, reject)
	}
	accept, reject = promotionGateDecided(1, 1, 2, PromoteMin)
	if accept || reject {
		t.Fatalf("micro arena should not early-stop got accept=%v reject=%v", accept, reject)
	}
}

func TestNoEarlyPromoteBeforeMinGames(t *testing.T) {
	accept, reject := promotionGateDecided(30, 46, 200, PromoteMin)
	if accept {
		t.Fatalf("30/46 hot streak must not accept before %d games", minGamesBeforePromote)
	}
	if reject {
		t.Fatalf("30/46 hot streak must not reject early")
	}
	// Below accept floor: strong rate but not enough games.
	accept, reject = promotionGateDecided(70, 99, 200, PromoteMin)
	if accept {
		t.Fatalf("70/99 must not accept before %d games", minGamesBeforePromote)
	}
	// Strong candidate still accepts once the floor is met.
	accept, reject = promotionGateDecided(120, 200, 200, PromoteMin)
	if !accept || reject {
		t.Fatalf("120/200 should accept after floor got accept=%v reject=%v", accept, reject)
	}
}

func TestPromotablePathRunsFull200(t *testing.T) {
	const maxGames = 200
	bestCasePromotes := func(wins, played int) bool {
		maxWins := wins + (maxGames - played)
		maxRate := float64(maxWins) / float64(maxGames)
		maxCiLow, _ := WilsonCI(maxWins, maxGames, 1.96)
		return maxRate >= PromoteMin && maxCiLow > promoteCILow
	}
	// 114/200 is the lowest integer final in the 0.53–0.57 band that clears both gates.
	for _, finalWins := range []int{114, 115, 116} {
		t.Run(fmt.Sprintf("final_%d", finalWins), func(t *testing.T) {
			for played := minGamesBeforeStop; played < maxGames; played++ {
				wins := played * finalWins / maxGames
				if !bestCasePromotes(wins, played) {
					continue
				}
				accept, reject := promotionGateDecided(wins, played, maxGames, PromoteMin)
				if reject {
					maxWins := wins + (maxGames - played)
					maxRate := float64(maxWins) / float64(maxGames)
					rate := float64(wins) / float64(played)
					ciLow, ciHigh := WilsonCI(wins, played, 1.96)
					t.Fatalf(
						"played=%d wins=%d rate=%.3f maxRate=%.3f ci=[%.3f,%.3f] false reject (accept=%v)",
						played, wins, rate, maxRate, ciLow, ciHigh, accept,
					)
				}
			}
			finalRate := float64(finalWins) / float64(maxGames)
			ciLow, ciHigh := WilsonCI(finalWins, maxGames, 1.96)
			accept, reject := promotionGateDecided(finalWins, maxGames, maxGames, PromoteMin)
			t.Logf(
				"final wins=%d rate=%.3f ci=[%.3f,%.3f] accept=%v reject=%v",
				finalWins, finalRate, ciLow, ciHigh, accept, reject,
			)
		})
	}
}

func TestBorderlinePathPlaysFull200(t *testing.T) {
	const maxGames = 200
	for _, finalWins := range []int{110, 111, 112, 113} {
		t.Run(fmt.Sprintf("final_%d", finalWins), func(t *testing.T) {
			for played := minGamesBeforeStop; played < maxGames; played++ {
				wins := played * finalWins / maxGames
				accept, reject := promotionGateDecided(wins, played, maxGames, PromoteMin)
				if accept {
					t.Fatalf("played=%d wins=%d false accept before 200", played, wins)
				}
				if reject {
					maxWins := wins + (maxGames - played)
					t.Fatalf("played=%d wins=%d maxWins=%d false reject before 200", played, wins, maxWins)
				}
			}
			rate := float64(finalWins) / float64(maxGames)
			ciLow, ciHigh := WilsonCI(finalWins, maxGames, 1.96)
			promote := rate >= PromoteMin && ciLow > promoteCILow
			if promote {
				t.Fatalf("final %d should not promote", finalWins)
			}
			t.Logf("final %d: rate=%.3f ci=[%.3f,%.3f] reject (full 200)", finalWins, rate, ciLow, ciHigh)
		})
	}
}

func TestPromotionGateFinalOutcomes(t *testing.T) {
	const maxGames = 200
	cases := []struct {
		wins    int
		promote bool
	}{
		{106, false}, // 0.53
		{108, false}, // 0.54
		{110, false}, // 0.55 — rate clears bar but Wilson low does not
		{112, false}, // 0.56
		{114, true},  // 0.57 — clears both gates
	}
	for _, tc := range cases {
		rate := float64(tc.wins) / float64(maxGames)
		ciLow, ciHigh := WilsonCI(tc.wins, maxGames, 1.96)
		accept, _ := promotionGateDecided(tc.wins, maxGames, maxGames, PromoteMin)
		promote := rate >= PromoteMin && ciLow > promoteCILow
		if promote != tc.promote {
			t.Fatalf(
				"wins=%d rate=%.3f ci=[%.3f,%.3f] promote=%v want %v",
				tc.wins, rate, ciLow, ciHigh, promote, tc.promote,
			)
		}
		if accept != tc.promote {
			t.Fatalf("wins=%d accept=%v want %v", tc.wins, accept, tc.promote)
		}
	}
}

func TestEarlyRejectWhenMaxRateTooLow(t *testing.T) {
	const maxGames = 200
	for _, finalPct := range []int{53, 54} {
		finalWins := maxGames * finalPct / 100
		rejectAt := 0
		for played := minGamesBeforeStop; played < maxGames; played++ {
			wins := played * finalWins / maxGames
			_, reject := promotionGateDecided(wins, played, maxGames, PromoteMin)
			if reject {
				rejectAt = played
				break
			}
		}
		if rejectAt == 0 {
			t.Fatalf("%d%% trajectory never rejected early", finalPct)
		}
		if rejectAt >= maxGames {
			t.Fatalf("%d%% rejected at or after cap", finalPct)
		}
		t.Logf("%d%% linear path: first reject at played=%d (before 200)", finalPct, rejectAt)
	}
}

func TestNoStopBeforeMinGames(t *testing.T) {
	for played := 1; played < minGamesBeforeStop; played++ {
		accept, reject := promotionGateDecided(played, played, 200, PromoteMin)
		if accept || reject {
			t.Fatalf("played=%d should not early-stop before floor", played)
		}
	}
}

func TestCountRoleWinIdenticalEval(t *testing.T) {
	cfg := MatchConfig{BlackEval: "heuristic", WhiteEval: "heuristic"}
	bw, cw := countRoleWin(cfg, GameSummary{WhiteWins: true, WhiteEval: "heuristic"}, 0, 0)
	if bw != 0 || cw != 0 {
		t.Fatalf("same eval names should not attribute wins got baseline=%d challenger=%d", bw, cw)
	}
}

func TestIdenticalEvalColorBalance(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	oldAccept := minGamesBeforePromote
	minGamesBeforePromote = 201
	t.Cleanup(func() { minGamesBeforePromote = oldAccept })

	const games = 200
	res := RunMatch(MatchConfig{
		Games:         games,
		Size:          9,
		Komi:          -1.0, // heuristic fair komi at test playouts; not the arena remap
		Playouts:      50,
		BlackEval:     "heuristic",
		WhiteEval:     "heuristic",
		Seed:          99,
		SwapColors:    true,
		OpeningMoves:  8,
		OpeningTemp:   1.0,
		Parallel:      4,
	})
	if res.Games != games {
		t.Fatalf("want %d games got %d", games, res.Games)
	}
	assertWinsNearExpected(t, res.WinsBlack, res.Games, 0.5, 3.0)
	t.Logf("black=%d white=%d draws=%d komi=%.1f",
		res.WinsBlack, res.WinsWhite, res.Draws, -1.0)
}


func assertWinsNearExpected(t *testing.T, wins, n int, p, sigmaN float64) {
	t.Helper()
	mean := float64(n) * p
	sigma := math.Sqrt(float64(n) * p * (1 - p))
	low := int(math.Floor(mean - sigmaN*sigma))
	high := int(math.Ceil(mean + sigmaN*sigma))
	if wins < low || wins > high {
		t.Fatalf("wins %d/%d outside [%d,%d] (%.1f sigma)", wins, n, low, high, sigmaN)
	}
}

func TestRunMatchSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	result := RunMatch(MatchConfig{
		Games:     2,
		Size:      5,
		Komi:      6.5,
		Playouts:  8,
		BlackEval: "heuristic",
		WhiteEval: "uniform",
		Seed:      42,
	})
	if result.Games != 2 {
		t.Fatalf("games %d", result.Games)
	}
	if result.ConfigHash == "" {
		t.Fatal("missing config hash")
	}
	if len(result.GameSummaries) != 2 {
		t.Fatalf("summaries %d", len(result.GameSummaries))
	}
}

func TestCapRandomizeDistribution(t *testing.T) {
	cfg := DefaultSelfplayConfig()
	cfg.CapRandomizeP = 0.25
	rng := rand.New(rand.NewSource(99))
	full := 0
	const n = 4000
	for i := 0; i < n; i++ {
		if _, fs := selfplayMovePlayouts(cfg, rng); fs {
			full++
		}
	}
	rate := float64(full) / float64(n)
	if rate < 0.18 || rate > 0.32 {
		t.Fatalf("full-search rate %v want ~0.25", rate)
	}
}

func TestFullOnlyExportFilters(t *testing.T) {
	cfg := DefaultSelfplayConfig()
	cfg.Games = 1
	cfg.Playouts = 10
	cfg.BoardSize = 5
	cfg.CapRandomizeP = 0.5
	cfg.Seed = 7
	cfg.FullOnlyExport = true
	all, _ := RunSelfplayWithLogs(SelfplayConfig{
		Games: cfg.Games, BoardSize: cfg.BoardSize, Komi: cfg.Komi,
		Playouts: cfg.Playouts, CapRandomizeP: 0, Seed: cfg.Seed, FullOnlyExport: false,
	})
	filtered := RunSelfplay(cfg)
	if len(filtered) > len(all) {
		t.Fatalf("filtered %d > all %d", len(filtered), len(all))
	}
	for _, s := range filtered {
		if !s.FullSearch {
			t.Fatal("non-full sample in filtered export")
		}
	}
}
