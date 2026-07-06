package main

import "math"

// softmaxPolicy mirrors training/inference_server.py softmax on policy logits.
func softmaxPolicy(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}
	maxv := logits[0]
	for _, v := range logits[1:] {
		if v > maxv {
			maxv = v
		}
	}
	out := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - maxv))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}
