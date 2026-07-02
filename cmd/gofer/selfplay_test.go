package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestSelfplayEvalONNXEmitsSamples(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/eval" {
			http.NotFound(w, r)
			return
		}
		var req sidecarEvalReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := sidecarEvalResp{Results: make([]struct {
			Value  float64   `json:"value"`
			Policy []float32 `json:"policy"`
		}, len(req.Spatial))}
		for i := range resp.Results {
			resp.Results[i].Value = 0.1
			resp.Results[i].Policy = make([]float32, 82)
			resp.Results[i].Policy[0] = 1
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	SetEvalConfig(EvalConfig{
		ONNXURL:     srv.URL,
		BatchSize:   1,
		EvalTimeout: 2 * time.Second,
		MaxWait:     time.Millisecond,
	})

	cfg := DefaultSelfplayConfig()
	cfg.Games = 1
	cfg.Playouts = 8
	cfg.FastPlayouts = 8
	cfg.FullPlayouts = 8
	cfg.CapRandomizeP = 1.0
	cfg.FullOnlyExport = false
	cfg.EvalMode = "onnx"
	cfg.Seed = 99

	samples, _ := RunSelfplayWithLogs(cfg)
	if len(samples) == 0 {
		t.Fatal("expected samples from onnx self-play")
	}
	for _, s := range samples {
		if len(s.FeaturesSpatial) == 0 || len(s.Policy) != 82 {
			t.Fatalf("invalid sample: spatial=%d policy=%d", len(s.FeaturesSpatial), len(s.Policy))
		}
	}
}

func TestSelfplayMixModeAlternates(t *testing.T) {
	var onnxHits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		onnxHits++
		resp := sidecarEvalResp{Results: []struct {
			Value  float64   `json:"value"`
			Policy []float32 `json:"policy"`
		}{{Value: 0, Policy: make([]float32, 82)}}}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	SetEvalConfig(EvalConfig{
		ONNXURL:     srv.URL,
		BatchSize:   1,
		EvalTimeout: 2 * time.Second,
		MaxWait:     time.Millisecond,
	})

	cfg := DefaultSelfplayConfig()
	cfg.Games = 4
	cfg.Playouts = 6
	cfg.FastPlayouts = 6
	cfg.FullPlayouts = 6
	cfg.CapRandomizeP = 1.0
	cfg.FullOnlyExport = false
	cfg.EvalMode = "mix"
	cfg.Seed = 7

	_, _ = RunSelfplayWithLogs(cfg)
	if onnxHits == 0 {
		t.Fatal("mix mode should hit ONNX sidecar on odd game indices")
	}
}

func TestSelfplayEvaluatorSelection(t *testing.T) {
	cfg := DefaultSelfplayConfig()
	cfg.EvalMode = "heuristic"
	if _, ok := selfplayEvaluator(cfg, 0).(Heuristic); !ok {
		t.Fatal("heuristic mode expected Heuristic evaluator")
	}
	cfg.EvalMode = "onnx"
	if _, ok := selfplayEvaluator(cfg, 0).(*BatchedEvaluator); !ok {
		t.Fatal("onnx mode expected BatchedEvaluator")
	}
	cfg.EvalMode = "mix"
	if _, ok := selfplayEvaluator(cfg, 1).(*BatchedEvaluator); !ok {
		t.Fatal("mix odd game expected BatchedEvaluator")
	}
	if _, ok := selfplayEvaluator(cfg, 0).(Heuristic); !ok {
		t.Fatal("mix even game expected Heuristic")
	}
}
