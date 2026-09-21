package main

import (
	"fmt"
	"sync/atomic"
)

// evalHealth records model failures that fall back to the heuristic.
var evalHealth struct {
	requests  atomic.Uint64
	fallbacks atomic.Uint64
}

func recordEvalRequests(n int) { evalHealth.requests.Add(uint64(n)) }

func recordEvalFallbacks(n int) { evalHealth.fallbacks.Add(uint64(n)) }

func resetEvalHealth() {
	evalHealth.requests.Store(0)
	evalHealth.fallbacks.Store(0)
}

// evalFallbackRate returns the heuristic fallback share and its counts.
func evalFallbackRate() (rate float64, requests, fallbacks uint64) {
	requests = evalHealth.requests.Load()
	fallbacks = evalHealth.fallbacks.Load()
	if requests == 0 {
		return 0, 0, 0
	}
	return float64(fallbacks) / float64(requests), requests, fallbacks
}

// maxEvalFallbackRate tolerates occasional transient model failures.
const maxEvalFallbackRate = 0.02

// passShareOpeningPlies limits the pass-collapse check to opening moves.
const passShareOpeningPlies = 20

// openingPassShare returns opening pass mass from full-search training targets.
func openingPassShare(samples []Sample) (share float64, rows int) {
	var total float64
	for _, s := range samples {
		if !s.FullSearch || s.MoveNum >= passShareOpeningPlies || len(s.Policy) == 0 {
			continue
		}
		total += float64(s.Policy[len(s.Policy)-1]) // pass is the last policy entry
		rows++
	}
	if rows == 0 {
		return 0, 0
	}
	return total / float64(rows), rows
}

// checkSelfplayPassCollapse rejects pass-heavy opening policy targets when enabled.
func checkSelfplayPassCollapse(share float64, rows int, max float64) error {
	if max <= 0 || rows == 0 || share <= max {
		return nil
	}
	return fmt.Errorf("self-play policy targets put %.1f%% of their mass on pass over the first %d plies "+
		"(%d rows), above the %.1f%% bar set by -selfplay-max-pass-share: the policy is collapsing toward "+
		"passing and these rows would train the next network to do more of it",
		share*100, passShareOpeningPlies, rows, max*100)
}
