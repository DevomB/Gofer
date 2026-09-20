package main

import "sync/atomic"

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
