package main

import "time"

// EvalConfig holds CLI options for evaluator construction.
type EvalConfig struct {
	ModelPath   string
	ModelPath2  string // second model for eval name "onnx2" (champion-vs-challenger)
	ONNXURL     string
	ONNXURL2    string // second sidecar (eval name "onnx2") for champion-vs-challenger arenas
	Backend     string // sidecar (default) or inprocess
	BatchSize   int
	EvalTimeout time.Duration
	MaxWait     time.Duration
}

var evalConfig = EvalConfig{
	Backend:   "sidecar",
	BatchSize: 8,
	EvalTimeout: 500 * time.Millisecond,
	MaxWait:     2 * time.Millisecond,
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
		c.Backend = "sidecar"
	}
	evalConfig = c
}
