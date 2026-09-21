package main

import (
	"fmt"
	"sync/atomic"
)

// Both ONNX paths answer with the heuristic when the model cannot: the sidecar
// on a transport error or timeout, the in-process backend on a session error or
// a malformed policy. That keeps a game moving, but it means a match can be
// decided entirely by the heuristic while reporting the model's name in
// black_eval and white_eval. A promotion gate that does this is not measuring
// the candidate at all, and it returns a plausible number rather than an error
// (ADR 0008). So the counts are process-wide, reported alongside the result, and
// the arena refuses a result that leaned on the fallback.
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

// evalFallbackRate is the share of evaluations the heuristic served instead of
// the model, with the totals it came from. A zero request count means no model
// backend ran, which is not the same as a healthy zero rate.
func evalFallbackRate() (rate float64, requests, fallbacks uint64) {
	requests = evalHealth.requests.Load()
	fallbacks = evalHealth.fallbacks.Load()
	if requests == 0 {
		return 0, 0, 0
	}
	return float64(fallbacks) / float64(requests), requests, fallbacks
}

// maxEvalFallbackRate is what an arena naming an ONNX evaluator will tolerate
// before refusing to report. Not zero: a handful of timeouts across a 600-game
// gate is noise, and failing on one would make gating flaky for no gain.
const maxEvalFallbackRate = 0.02

// passShareOpeningPlies bounds the moves the pass-collapse check looks at.
// Passing is a real move in a finished position, so a whole-game average mixes
// the pathology with correct endgame play. Inside the opening it is never right.
const passShareOpeningPlies = 20

// Observed opening pass shares, three arms of one live run (2026-09-21):
//
//	heuristic-generated shards        0.006 - 0.007
//	net-generated, strength falling   0.032 - 0.087
//
// Over the same shards the networks went +53 -> -35 -> -168 Elo against the
// heuristic, and by the last one the network's best move on an empty board was
// pass with 43.5% of the visits -- while the search valued pass at 0.216
// against 0.463 for the best real move, so the learned prior was overriding the
// search rather than the search finding pass good.
//
// That is enough to separate heuristic self-play from collapsing self-play. It
// is NOT enough to set a bar, because this run never produced healthy
// net-generated self-play: every net shard in it is on the way down, so there
// is no upper end of the healthy band to sit above. A threshold picked from
// these numbers alone would either fire on all net self-play or, as the first
// version of this check did at 0.12, on none of it.
//
// So the share is always measured and always recorded in ShardMeta, and the
// refusal is opt-in until someone can calibrate it against a run that works.
// The project has already paid once for a threshold shipped on argument rather
// than measurement (maxEvalFallbackRate, chosen at 2% and never checked against
// a healthy rate, which turned out to be 0).

// openingPassShare is the mean probability the full-search policy targets put on
// pass over the opening plies. Fast-search rows are excluded because their
// visit distributions are not training targets for the policy head.
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

// checkSelfplayPassCollapse refuses to write a shard whose opening policy
// targets have collapsed toward passing.
//
// This guards the same thing checkSelfplayEvalHealth does -- training data,
// which outlives the cycle that produced it -- against a failure that is
// quieter. A collapsed shard has the right row count, the right provenance and
// an eval fallback rate of zero; every gate that follows it rejects correctly
// while the replay window fills with games decided by mutual passing.
// A max of 0 disables the refusal, which is the default: see the note above on
// why this cannot be calibrated from a run that never worked.
func checkSelfplayPassCollapse(share float64, rows int, max float64) error {
	if max <= 0 || rows == 0 || share <= max {
		return nil
	}
	return fmt.Errorf("self-play policy targets put %.1f%% of their mass on pass over the first %d plies "+
		"(%d rows), above the %.1f%% bar set by -selfplay-max-pass-share: the policy is collapsing toward "+
		"passing and these rows would train the next network to do more of it",
		share*100, passShareOpeningPlies, rows, max*100)
}
