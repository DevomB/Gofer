package main

import "testing"

// The reported failure: materially different opening settings produced the same
// config hash, so a resumed gate could reuse arena evidence generated under a
// different configuration. Each field below changes what the match measures.
func TestMatchConfigHashDistinguishesWhatChangesTheResult(t *testing.T) {
	base := MatchConfig{
		Games: 200, Size: 9, Komi: 6.5, Playouts: 400,
		BlackEval: "heuristic", WhiteEval: "heuristic", Seed: 42,
		OpeningMoves: 8, OpeningTemp: 1.0,
	}
	h := matchConfigHash(base)

	for _, tc := range []struct {
		name string
		mut  func(*MatchConfig)
	}{
		{"opening moves", func(c *MatchConfig) { c.OpeningMoves = 16 }},
		{"opening temperature", func(c *MatchConfig) { c.OpeningTemp = 0.5 }},
		{"play-all-games", func(c *MatchConfig) { c.PlayAllGames = true }},
		{"komi", func(c *MatchConfig) { c.Komi = 0.5 }},
		{"playouts", func(c *MatchConfig) { c.Playouts = 200 }},
		{"seed", func(c *MatchConfig) { c.Seed = 43 }},
		{"evaluator", func(c *MatchConfig) { c.WhiteEval = "onnx" }},
	} {
		cfg := base
		tc.mut(&cfg)
		if got := matchConfigHash(cfg); got == h {
			t.Errorf("%s: hash unchanged (%s); a cached report would be reused across this difference", tc.name, got)
		}
	}
}

// Concurrency must NOT change the hash. Games are a pure function of (config,
// game index), so a report from a 4-way run is valid evidence for a 32-way one;
// hashing Parallel would throw away reusable matches for no reason.
func TestMatchConfigHashIgnoresParallelism(t *testing.T) {
	base := MatchConfig{Games: 200, Size: 9, Komi: 6.5, Playouts: 400, Seed: 42, Parallel: 4}
	wide := base
	wide.Parallel = 32
	if matchConfigHash(base) != matchConfigHash(wide) {
		t.Error("parallelism changed the hash; it cannot change a result")
	}
}

// The backend and the model identities are outside MatchConfig, in evalConfig,
// so the hash has to reach for them. A candidate is a different network every
// cycle behind an unchanging path.
func TestMatchConfigHashCoversBackendAndModels(t *testing.T) {
	saved := evalConfig
	t.Cleanup(func() { evalConfig = saved })
	cfg := MatchConfig{Games: 40, Size: 9, Komi: 6.5, Playouts: 400, BlackEval: "onnx", WhiteEval: "onnx2"}

	evalConfig = EvalConfig{Backend: "inprocess", ModelPath: "a.onnx", ModelPath2: "b.onnx"}
	inproc := matchConfigHash(cfg)

	evalConfig.Backend = "sidecar"
	if matchConfigHash(cfg) == inproc {
		t.Error("backend change did not move the hash")
	}

	evalConfig = EvalConfig{Backend: "sidecar", ONNXURL: "http://a:8080", ONNXURL2: "http://b:8081"}
	twoSidecars := matchConfigHash(cfg)
	evalConfig.ONNXURL2 = "http://a:8080" // challenger pointed at the champion
	if matchConfigHash(cfg) == twoSidecars {
		t.Error("collapsing both sidecars onto one URL did not move the hash")
	}
}

// An absent model and an unreadable one are different failures and must not
// hash alike: the first is an arena with no model, the second is a broken run.
func TestFileDigestSeparatesAbsentFromUnreadable(t *testing.T) {
	if fileDigest("") != "none" {
		t.Errorf("empty path = %q, want none", fileDigest(""))
	}
	missing := fileDigest("does-not-exist.onnx")
	if missing == "none" || missing == "" {
		t.Errorf("missing file = %q, want an unreadable marker", missing)
	}
	real := fileDigest("testdata/features_golden_v2.json")
	if real == missing || len(real) != 16 {
		t.Errorf("readable file digest = %q", real)
	}
}
