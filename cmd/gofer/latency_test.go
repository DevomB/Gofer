package main

import (
	"sort"
	"testing"
	"time"
)

func TestLatencyReport9x9ONNX(t *testing.T) {
	if testing.Short() {
		t.Skip("latency harness skipped in -short")
	}
	url := evalConfig.ONNXURL
	if url == "" {
		url = "http://127.0.0.1:8080"
	}
	SetEvalConfig(EvalConfig{
		Backend:     "sidecar",
		ONNXURL:     url,
		BatchSize:   8,
		EvalTimeout: 50 * time.Millisecond,
		MaxWait:     2 * time.Millisecond,
	})
	client := SidecarBackend{URL: url}.httpClient()
	if err := CheckSidecarHealth(url, client); err != nil {
		t.Skip("sidecar not running:", err)
	}

	r := Chinese()
	// warmup sidecar batch path
	w := newSearchEngine(r, 400, 0, "onnx")
	_ = w.BestMove(NewBoard(9, 6.5))
	w.Close()

	samples := make([]float64, 0, 50)
	for i := 0; i < 50; i++ {
		b := NewBoard(9, 6.5)
		for j := 0; j < i%3; j++ {
			_ = r.Play(b, StoneMove(At((i+j)%9, (i*2+j)%9)))
		}
		eng := newSearchEngine(r, 400, 0, "onnx")
		start := time.Now()
		_ = eng.BestMove(b)
		eng.Close()
		samples = append(samples, float64(time.Since(start).Milliseconds()))
	}
	sort.Float64s(samples)
	p50 := samples[len(samples)/2]
	p99 := samples[int(float64(len(samples)-1)*0.99)]
	t.Logf("9x9@400 playouts onnx: p50=%.0fms p99=%.0fms n=%d", p50, p99, len(samples))
}

func TestLatencyReport19x19Heuristic(t *testing.T) {
	if testing.Short() {
		t.Skip("latency harness skipped in -short")
	}
	r := Chinese()
	b := NewBoard(19, 7.5)
	eng := newSearchEngine(r, 1600, 0, "heuristic")
	start := time.Now()
	a := eng.Analyze(b, 1)
	elapsed := float64(time.Since(start).Milliseconds())
	if a.Playouts != 1600 {
		t.Fatalf("playouts %d want 1600", a.Playouts)
	}
	t.Logf("19x19@1600 heuristic: wall=%.0fms playouts=%d", elapsed, a.Playouts)
}
