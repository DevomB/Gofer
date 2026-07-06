package main

import (
	"sync"
	"time"
)

// EvalBackend evaluates a batch of boards (ONNX/mock worker).
// Implementations: SidecarBackend (HTTP), ORTBackend (in-process, -tags=onnx).
type EvalBackend interface {
	EvalBatch(boards []*Board) []Result
}

// Inference is a mock neural-network evaluator hook for external runtimes.
type Inference struct {
	MockValue  float64
	MockPolicy []float32
	Latency    time.Duration // simulated inference delay per batch
}

// EvalBatch implements EvalBackend.
func (inf Inference) EvalBatch(boards []*Board) []Result {
	if inf.Latency > 0 {
		time.Sleep(inf.Latency)
	}
	out := make([]Result, len(boards))
	for i, b := range boards {
		out[i] = inf.Evaluate(b)
	}
	return out
}

// Evaluate implements Evaluator (single-position).
func (inf Inference) Evaluate(b *Board) Result {
	n := b.Size()*b.Size() + 1
	p := inf.MockPolicy
	if len(p) != n {
		p = make([]float32, n)
		for i := range p {
			p[i] = 1
		}
	}
	return Result{Value: inf.MockValue, Policy: p, HasValue: true}
}

type batchReq struct {
	b    *Board
	resp chan Result
}

// BatchedEvaluator queues positions and evaluates in batches.
type BatchedEvaluator struct {
	backend    EvalBackend
	fallback   Evaluator
	minBatch   int
	maxWait    time.Duration
	reqTimeout time.Duration
	reqCh      chan batchReq
	stopCh     chan struct{}
	wg         sync.WaitGroup
	once       sync.Once
}

// NewBatchedEvaluator starts the batch worker.
func NewBatchedEvaluator(backend EvalBackend, fallback Evaluator, minBatch int, maxWait time.Duration) *BatchedEvaluator {
	return NewBatchedEvaluatorWithTimeout(backend, fallback, minBatch, maxWait, maxWait*4)
}

// NewBatchedEvaluatorWithTimeout starts the batch worker with an explicit request timeout.
func NewBatchedEvaluatorWithTimeout(backend EvalBackend, fallback Evaluator, minBatch int, maxWait, reqTimeout time.Duration) *BatchedEvaluator {
	if minBatch < 1 {
		minBatch = 8
	}
	if maxWait <= 0 {
		maxWait = 2 * time.Millisecond
	}
	if reqTimeout <= 0 {
		reqTimeout = maxWait * 4
	}
	b := &BatchedEvaluator{
		backend:    backend,
		fallback:   fallback,
		minBatch:   minBatch,
		maxWait:    maxWait,
		reqTimeout: reqTimeout,
		reqCh:      make(chan batchReq, 256),
		stopCh:     make(chan struct{}),
	}
	b.wg.Add(1)
	go b.worker()
	return b
}

// QueueDepth returns pending evaluate requests (measurement only).
func (b *BatchedEvaluator) QueueDepth() int {
	return len(b.reqCh)
}

// Close stops the batch worker.
func (b *BatchedEvaluator) Close() {
	b.once.Do(func() {
		close(b.stopCh)
		b.wg.Wait()
	})
}

// Evaluate submits one position and waits for batched result.
func (b *BatchedEvaluator) Evaluate(board *Board) Result {
	resp := make(chan Result, 1)
	req := batchReq{b: board, resp: resp}
	select {
	case b.reqCh <- req:
	case <-time.After(b.reqTimeout):
		return b.fallback.Evaluate(board)
	}
	select {
	case r := <-resp:
		return r
	case <-time.After(b.reqTimeout):
		return b.fallback.Evaluate(board)
	}
}

func (b *BatchedEvaluator) worker() {
	defer b.wg.Done()
	for {
		select {
		case <-b.stopCh:
			return
		case req := <-b.reqCh:
			batch, boards, stopped := b.gatherBatch(req)
			if stopped {
				return
			}
			b.dispatchBatch(batch, boards)
		}
	}
}

func (b *BatchedEvaluator) gatherBatch(first batchReq) ([]batchReq, []*Board, bool) {
	batch := []batchReq{first}
	boards := []*Board{first.b}
	deadline := time.Now().Add(b.maxWait)
	for len(batch) < b.minBatch {
		wait := time.Until(deadline)
		if wait <= 0 {
			break
		}
		select {
		case <-b.stopCh:
			return nil, nil, true
		case r := <-b.reqCh:
			batch = append(batch, r)
			boards = append(boards, r.b)
		case <-time.After(wait):
			return batch, boards, false
		}
	}
	return batch, boards, false
}

func (b *BatchedEvaluator) dispatchBatch(batch []batchReq, boards []*Board) {
	results := b.backend.EvalBatch(boards)
	for i, r := range batch {
		if i < len(results) {
			r.resp <- results[i]
		} else {
			r.resp <- b.fallback.Evaluate(r.b)
		}
	}
}

// GatingHarness compares win rates for model promotion.
type GatingHarness struct {
	Games            int
	MinWinRateMargin float64 // e.g. 0.55 vs 0.45
}

// Pass returns true if candidate beats baseline by the configured margin.
func (g GatingHarness) Pass(baselineWin, candidateWin float64) bool {
	if g.Games <= 0 {
		return true
	}
	margin := g.MinWinRateMargin
	if margin == 0 {
		margin = PromoteMin
	}
	return candidateWin >= margin && candidateWin > baselineWin
}
