package main

import (
	"sync"
	"testing"
	"time"
)

func TestBatchedEvaluator(t *testing.T) {
	ev := NewBatchedEvaluator(Inference{MockValue: 0.5}, Heuristic{}, 2, time.Millisecond)
	defer ev.Close()
	b := NewBoard(5, 6.5)
	r := ev.Evaluate(b)
	if r.Value != 0.5 {
		t.Fatalf("value %v", r.Value)
	}
}

func TestBatchedEvaluatorParallel(t *testing.T) {
	ev := NewBatchedEvaluator(Inference{Latency: 100 * time.Microsecond}, Heuristic{}, 4, 2*time.Millisecond)
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
