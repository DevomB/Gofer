package main

import "testing"

// The reported failure: run an arena with both ONNX endpoints closed and it
// exits cleanly with a normal-looking onnx vs onnx2 result. Both backends
// answer with the heuristic rather than failing, so every game completes and
// nothing in the report says the model never ran. A promotion gate would score
// the heuristic against itself and call it a candidate evaluation.
func TestArenaRefusesResultDecidedByFallback(t *testing.T) {
	cfg := MatchConfig{BlackEval: "onnx", WhiteEval: "onnx2"}

	// Every evaluation fell back: the endpoints were closed.
	allFallback := MatchResult{Games: 40, EvalRequests: 5000, EvalFallbacks: 5000, EvalFallbackRate: 1.0}
	if err := checkEvalHealth(cfg, allFallback); err == nil {
		t.Fatal("accepted a result where 100% of evaluations were the heuristic")
	}

	// The subtler case: the backend never ran at all, so there is nothing to
	// take a rate over and a naive rate check reads 0.0 as healthy.
	neverRan := MatchResult{Games: 40}
	if err := checkEvalHealth(cfg, neverRan); err == nil {
		t.Fatal("accepted a result where no model evaluation was attempted")
	}

	// A handful of timeouts across a long gate is noise, not a broken model.
	occasional := MatchResult{Games: 600, EvalRequests: 100000, EvalFallbacks: 500, EvalFallbackRate: 0.005}
	if err := checkEvalHealth(cfg, occasional); err != nil {
		t.Errorf("rejected a healthy run over 0.5%% fallbacks: %v", err)
	}
}

// A heuristic-only arena has no model to fall back from, so the check must not
// fire: the reproducible strength baseline is exactly this shape.
func TestEvalHealthIgnoresHeuristicOnlyArenas(t *testing.T) {
	cfg := MatchConfig{BlackEval: "heuristic", WhiteEval: "heuristic"}
	if err := checkEvalHealth(cfg, MatchResult{Games: 200}); err != nil {
		t.Errorf("heuristic-only arena rejected: %v", err)
	}
}

func TestNamesONNX(t *testing.T) {
	for _, name := range []string{"onnx", "onnx2", "onnx-batch", "ONNX"} {
		if !namesONNX(name) {
			t.Errorf("namesONNX(%q) = false", name)
		}
	}
	for _, name := range []string{"heuristic", "uniform", "batched", "mock-batch", ""} {
		if namesONNX(name) {
			t.Errorf("namesONNX(%q) = true", name)
		}
	}
}

// The counters are process-wide, so a second match must not inherit the first
// one's fallbacks. RunMatch resets them; this pins that contract.
func TestEvalHealthResets(t *testing.T) {
	resetEvalHealth()
	recordEvalRequests(100)
	recordEvalFallbacks(40)
	rate, req, fb := evalFallbackRate()
	if rate != 0.4 || req != 100 || fb != 40 {
		t.Fatalf("got rate=%v requests=%d fallbacks=%d, want 0.4/100/40", rate, req, fb)
	}
	resetEvalHealth()
	if rate, req, fb := evalFallbackRate(); rate != 0 || req != 0 || fb != 0 {
		t.Errorf("after reset got rate=%v requests=%d fallbacks=%d", rate, req, fb)
	}
}
