package main

// Evaluator and search-engine construction from CLI names and flags.

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

func evalBackendInprocess() bool {
	return strings.EqualFold(evalConfig.Backend, "inprocess")
}

func parseEvaluator(name string) Evaluator {
	switch {
	case strings.EqualFold(name, "uniform"):
		return Uniform{}
	case strings.EqualFold(name, "batched"), strings.EqualFold(name, "mock-batch"):
		return NewBatchedEvaluatorWithTimeout(
			Inference{MockValue: 0, Latency: 500 * time.Microsecond},
			Heuristic{},
			evalConfig.BatchSize,
			evalConfig.MaxWait,
			evalConfig.EvalTimeout,
		)
	case strings.EqualFold(name, "onnx"), strings.EqualFold(name, "onnx-batch"):
		return newONNXEvaluator(champion, evalConfig.BatchSize)
	case strings.EqualFold(name, "onnx2"):
		return newONNXEvaluator(challenger, evalConfig.BatchSize)
	case strings.EqualFold(name, "heuristic2"):
		return Heuristic{}
	default:
		return Heuristic{}
	}
}

// onnxSlot selects which of the two configured ONNX evaluators to build. Arena
// gating runs the champion in one slot and the challenger in the other.
type onnxSlot int

const (
	champion onnxSlot = iota
	challenger
)

// resolveONNXSlot returns the model path and sidecar URL configured for a slot.
// The challenger falls back to the champion's setting for each independently, so
// a single-model run can still name onnx2 without configuring a second of
// everything.
func resolveONNXSlot(slot onnxSlot) (model, url string) {
	model, url = evalConfig.ModelPath, evalConfig.ONNXURL
	if slot == challenger {
		if evalConfig.ModelPath2 != "" {
			model = evalConfig.ModelPath2
		}
		if evalConfig.ONNXURL2 != "" {
			url = evalConfig.ONNXURL2
		}
	}
	return model, url
}

// newONNXEvaluator builds a batched evaluator for one slot, backed either by
// in-process ONNX Runtime or by an HTTP sidecar according to evalConfig.Backend.
// The heuristic is the fallback on both paths.
func newONNXEvaluator(slot onnxSlot, minBatch int) Evaluator {
	if minBatch < 1 {
		minBatch = evalConfig.BatchSize
	}
	model, url := resolveONNXSlot(slot)

	var backend EvalBackend
	if evalBackendInprocess() {
		ort, err := newORTBackend(model, Heuristic{}, evalConfig.ORTIntraThreads)
		if err != nil {
			fmt.Fprintf(os.Stderr, "in-process ONNX: %v\n", err)
			os.Exit(1)
		}
		backend = ort
	} else {
		if url == "" {
			url = "http://127.0.0.1:8080"
		}
		backend = SidecarBackend{
			URL:      url,
			Fallback: Heuristic{},
			Client:   &http.Client{Timeout: evalConfig.EvalTimeout},
		}
	}
	return NewBatchedEvaluatorDispatch(
		backend,
		Heuristic{},
		minBatch,
		evalConfig.MaxWait,
		evalConfig.EvalTimeout,
		evalConfig.EvalDispatchers,
	)
}

func newSearchEngine(r Ruleset, playouts int, think time.Duration, evalName string) *Engine {
	return newSearchEngineSeed(r, playouts, think, evalName, DefaultConfig().Seed)
}

func newSearchEngineSeed(r Ruleset, playouts int, think time.Duration, evalName string, seed int64) *Engine {
	cfg := DefaultConfig()
	cfg.Playouts = playouts
	cfg.ThinkTime = think
	cfg.Seed = seed
	return NewEngine(r, parseEvaluator(evalName), cfg)
}

func defaultPlayoutsForSize(size int) int {
	switch {
	case size <= 9:
		return 400
	case size <= 13:
		return 800
	default:
		return 1600
	}
}
