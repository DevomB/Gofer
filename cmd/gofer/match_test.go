package main

import (
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
