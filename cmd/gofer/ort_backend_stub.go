//go:build !onnx

package main

import "fmt"

var errInprocessUnavailable = fmt.Errorf("rebuild with -tags=onnx and CGO_ENABLED=1 for in-process ONNX")

func newORTBackend(modelPath string, fallback Evaluator) (*ORTBackend, error) {
	return nil, errInprocessUnavailable
}

// ORTBackend is a stub when built without the onnx tag.
type ORTBackend struct{}

func (o *ORTBackend) Close() {}

func (o ORTBackend) EvalBatch(boards []*Board) []Result {
	out := make([]Result, len(boards))
	return out
}
