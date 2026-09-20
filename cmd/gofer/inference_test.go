package main

import (
	"sync"
	"testing"
	"time"
)

func TestBatchedEvaluator(t *testing.T) {
	// minBatch=2 with a single request means only the maxWait timer can flush the
	// batch, which is the path worth exercising. The request timeout has to be far
	// larger than maxWait, though: NewBatchedEvaluator derives it as maxWait*4, and
	// a 4ms deadline expires before the worker is even scheduled on a loaded
	// machine. Evaluate then silently returns fallback.Evaluate(board) — Heuristic{}
	// on an empty board, i.e. 0 — and the failure looks like a wrong value rather
	// than a missed deadline. That is exactly how this test flaked in a full run.
	ev := NewBatchedEvaluatorWithTimeout(Inference{MockValue: 0.5}, Heuristic{}, 2, time.Millisecond, 10*time.Second)
	defer ev.Close()
	b := NewBoard(5, 6.5)
	r := ev.Evaluate(b)
	if r.Value != 0.5 {
		t.Fatalf("value %v: the result should come from the backend, not the fallback", r.Value)
	}
}

func TestBatchedEvaluatorParallel(t *testing.T) {
	// The request timeout is deliberately longer than the deadlock guard below.
	// With the maxWait*4 default a stalled worker would be masked: every Evaluate
	// would quietly fall back and return long before the guard could fire, so the
	// test would pass without the batch path ever running.
	ev := NewBatchedEvaluatorWithTimeout(Inference{Latency: 100 * time.Microsecond}, Heuristic{}, 4, 2*time.Millisecond, 30*time.Second)
	defer ev.Close()
	b := NewBoard(9, 6.5)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = ev.Evaluate(b)
		}()
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("deadlock in parallel batched eval")
	}
}

func TestBatchedSearch(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 20
	cfg.Workers = 4
	e := NewEngine(r, NewBatchedEvaluator(Inference{}, Heuristic{}, 4, 2*time.Millisecond), cfg)
	defer func() {
		if c, ok := e.Eval.(*BatchedEvaluator); ok {
			c.Close()
		}
	}()
	b := NewBoard(5, 6.5)
	_ = e.BestMove(b)
}
