package main

import "time"

// EvalConfig holds CLI options for evaluator construction.
type EvalConfig struct {
	ModelPath   string
	ModelPath2  string // second model for eval name "onnx2" (champion-vs-challenger)
	ONNXURL     string
	ONNXURL2    string // second sidecar (eval name "onnx2") for champion-vs-challenger arenas
	Backend     string // inprocess (default) or sidecar
	BatchSize   int
	EvalTimeout time.Duration
	MaxWait     time.Duration
	// ORT threads per inference call. One dispatch goroutine per model runs
	// EvalBatch, so this is what decides how much of the machine an in-process
	// run can use: at 1, a 32-core box evaluates on one core per model.
	// 0 leaves ORT to pick; 1 is the historical value and keeps results bit-exact
	// against the parity reference, which a multi-threaded reduction does not.
	ORTIntraThreads int
	// Concurrent in-flight inferences per model. 1 is the historical shape and
	// caps the engine at one evaluation at a time however many games run.
	EvalDispatchers int
}

var evalConfig = EvalConfig{
	Backend:         "inprocess",
	BatchSize:       8,
	ORTIntraThreads: 1,
	EvalDispatchers: 1,
	EvalTimeout:     500 * time.Millisecond,
	MaxWait:         2 * time.Millisecond,
}

// SetEvalConfig updates package-level evaluator options (called from flag parse).
func SetEvalConfig(c EvalConfig) {
	if c.BatchSize < 1 {
		c.BatchSize = 8
	}
	if c.EvalTimeout <= 0 {
		c.EvalTimeout = 8 * time.Millisecond
	}
	if c.MaxWait <= 0 {
		c.MaxWait = 2 * time.Millisecond
	}
	if c.Backend == "" {
		c.Backend = "inprocess"
	}
	if c.EvalDispatchers < 1 {
		c.EvalDispatchers = 1
	}
	if c.ORTIntraThreads < 0 {
		c.ORTIntraThreads = 1
	}
	evalConfig = c
}
