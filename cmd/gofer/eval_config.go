package main

import "time"

// EvalConfig holds CLI options for evaluator construction.
type EvalConfig struct {
	ModelPath   string
	ONNXURL     string
	BatchSize   int
	EvalTimeout time.Duration
	MaxWait     time.Duration
}

var evalConfig = EvalConfig{
	BatchSize:   8,
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
	evalConfig = c
}
