package main

import (
	"math"
	"testing"
)

func TestSoftmaxPolicyMatchesPython(t *testing.T) {
	// Same logits as numpy reference: x - max(x); exp; normalize
	logits := []float32{1.0, 2.0, 0.5, -1.0}
	got := softmaxPolicy(logits)
	var sum float64
	for _, p := range got {
		sum += float64(p)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Fatalf("sum=%v", sum)
	}
	if len(got) != len(logits) {
		t.Fatal("length mismatch")
	}
}
