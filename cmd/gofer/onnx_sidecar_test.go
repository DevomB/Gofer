package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestSidecarBackendEvalBatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/eval" {
			http.NotFound(w, r)
			return
		}
		var req sidecarEvalReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		results := make([]map[string]any, req.BatchSize)
		for i := 0; i < req.BatchSize; i++ {
			policy := make([]float32, 82)
			for j := range policy {
				policy[j] = 1.0 / 82
			}
			results[i] = map[string]any{"value": 0.25, "policy": policy}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
	}))
	defer srv.Close()

	b := NewBoard(9, 6.5)
	backend := SidecarBackend{URL: srv.URL, Client: srv.Client()}
	out := backend.EvalBatch([]*Board{b})
	if len(out) != 1 {
		t.Fatalf("len %d", len(out))
	}
	if out[0].Value != 0.25 {
		t.Fatalf("value %v", out[0].Value)
	}
	if len(out[0].Policy) != 82 {
		t.Fatalf("policy len %d", len(out[0].Policy))
	}
}

func TestSidecarBackendShapeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{{"value": 0.0, "policy": []float32{1, 2, 3}}},
		})
	}))
	defer srv.Close()

	b := NewBoard(9, 6.5)
	backend := SidecarBackend{URL: srv.URL, Client: srv.Client()}
	out := backend.EvalBatch([]*Board{b})
	if len(out[0].Policy) != 82 {
		// fallback heuristic policy
		h := Heuristic{}.Evaluate(b)
		if len(out[0].Policy) != len(h.Policy) {
			t.Fatalf("expected heuristic fallback policy len %d got %d", len(h.Policy), len(out[0].Policy))
		}
	}
}

func TestSidecarBackendHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "fail", 500)
	}))
	defer srv.Close()

	b := NewBoard(9, 6.5)
	backend := SidecarBackend{URL: srv.URL, Client: srv.Client()}
	out := backend.EvalBatch([]*Board{b})
	if len(out) != 1 || len(out[0].Policy) == 0 {
		t.Fatal("expected heuristic fallback")
	}
}

func TestSidecarBackendTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	b := NewBoard(9, 6.5)
	backend := SidecarBackend{
		URL:    srv.URL,
		Client: &http.Client{Timeout: 5 * time.Millisecond},
	}
	out := backend.EvalBatch([]*Board{b})
	if len(out) != 1 {
		t.Fatalf("len %d", len(out))
	}
}

func TestBatchedEvaluatorSidecarParallel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Microsecond)
		var req sidecarEvalReq
		_ = json.NewDecoder(r.Body).Decode(&req)
		results := make([]map[string]any, req.BatchSize)
		for i := 0; i < req.BatchSize; i++ {
			policy := make([]float32, 82)
			for j := range policy {
				policy[j] = 1.0 / 82
			}
			results[i] = map[string]any{"value": 0.0, "policy": policy}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
	}))
	defer srv.Close()

	ev := NewBatchedEvaluatorWithTimeout(
		SidecarBackend{URL: srv.URL, Client: srv.Client()},
		Heuristic{},
		4,
		2*time.Millisecond,
		20*time.Millisecond,
	)
	defer ev.Close()
	board := NewBoard(9, 6.5)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = ev.Evaluate(board)
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
		t.Fatal("deadlock in parallel sidecar batched eval")
	}
}
