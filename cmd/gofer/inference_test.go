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

// concurrencyProbe records the most EvalBatch calls ever in flight at once.
type concurrencyProbe struct {
	mu       sync.Mutex
	inFlight int
	peak     int
	delay    time.Duration
}

func (p *concurrencyProbe) EvalBatch(boards []*Board) []Result {
	p.mu.Lock()
	p.inFlight++
	if p.inFlight > p.peak {
		p.peak = p.inFlight
	}
	p.mu.Unlock()
	time.Sleep(p.delay)
	p.mu.Lock()
	p.inFlight--
	p.mu.Unlock()
	return make([]Result, len(boards))
}

func (p *concurrencyProbe) peakInFlight() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.peak
}

// The failure mode this pins is only slowness: with one dispatcher every
// evaluation waits for the previous one to return, however many games are
// running, and nothing errors or logs. On a 32-core box that showed up as
// sixteen parallel games driving 1.15 cores.
func TestDispatchersDecideInFlightInferences(t *testing.T) {
	for _, tc := range []struct{ dispatchers, wantPeak int }{{1, 1}, {4, 4}} {
		probe := &concurrencyProbe{delay: 50 * time.Millisecond}
		// minBatch 1: every request dispatches at once, so the peak reflects the
		// number of workers rather than how fast requests happened to arrive.
		ev := NewBatchedEvaluatorDispatch(probe, Heuristic{}, 1, time.Millisecond, 30*time.Second, tc.dispatchers)
		b := NewBoard(9, 6.5)
		var wg sync.WaitGroup
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = ev.Evaluate(b)
			}()
		}
		wg.Wait()
		ev.Close()

		if got := probe.peakInFlight(); got != tc.wantPeak {
			t.Errorf("dispatchers=%d: peak in-flight = %d, want %d", tc.dispatchers, got, tc.wantPeak)
		}
		if ev.Dispatchers() != tc.dispatchers {
			t.Errorf("Dispatchers() = %d, want %d", ev.Dispatchers(), tc.dispatchers)
		}
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
