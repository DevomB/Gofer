package main

import (
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestSelfplayEvalONNXEmitsSamples(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(testSidecarEvalHandler(t)))
	defer srv.Close()

	setTestEvalConfig(srv.URL)

	cfg := testSelfplayConfig("onnx", 1)
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

func testSidecarEvalHandler(t *testing.T) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
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
		_ = json.NewEncoder(w).Encode(testSidecarResponse(len(req.Spatial)))
	}
}

func testSidecarResponse(n int) sidecarEvalResp {
	resp := sidecarEvalResp{Results: make([]struct {
		Value  float64   `json:"value"`
		Policy []float32 `json:"policy"`
	}, n)}
	for i := range resp.Results {
		resp.Results[i].Value = 0.1
		resp.Results[i].Policy = make([]float32, 82)
		resp.Results[i].Policy[0] = 1
	}
	return resp
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

	setTestEvalConfig(srv.URL)

	cfg := testSelfplayConfig("mix", 4)
	cfg.Seed = 7

	_, _ = RunSelfplayWithLogs(cfg)
	if onnxHits == 0 {
		t.Fatal("mix mode should hit ONNX sidecar on odd game indices")
	}
}

func setTestEvalConfig(url string) {
	SetEvalConfig(EvalConfig{
		Backend:     "sidecar",
		ONNXURL:     url,
		BatchSize:   1,
		EvalTimeout: 2 * time.Second,
		MaxWait:     time.Millisecond,
	})
}

func testSelfplayConfig(evalMode string, games int) SelfplayConfig {
	cfg := DefaultSelfplayConfig()
	cfg.Games = games
	cfg.Playouts = 6
	cfg.FastPlayouts = 6
	cfg.FullPlayouts = 6
	cfg.CapRandomizeP = 1.0
	cfg.FullOnlyExport = false
	cfg.EvalMode = evalMode
	return cfg
}

func TestSelfplayEvaluatorSelection(t *testing.T) {
	batched := NewBatchedEvaluatorWithTimeout(Inference{}, Heuristic{}, 1, time.Millisecond, time.Second)
	defer batched.Close()
	pool := evalPool{onnx: batched, heuristic: Heuristic{}}

	cfg := DefaultSelfplayConfig()
	cfg.EvalMode = "heuristic"
	if _, ok := selfplayEvaluator(cfg, 0, rand.New(rand.NewSource(1)), pool).(Heuristic); !ok {
		t.Fatal("heuristic mode expected Heuristic evaluator")
	}
	cfg.EvalMode = "onnx"
	if _, ok := selfplayEvaluator(cfg, 0, rand.New(rand.NewSource(1)), pool).(*BatchedEvaluator); !ok {
		t.Fatal("onnx mode expected BatchedEvaluator")
	}
	cfg.EvalMode = "mix"
	cfg.ONNXFraction = 1.0
	if _, ok := selfplayEvaluator(cfg, 0, rand.New(rand.NewSource(1)), pool).(*BatchedEvaluator); !ok {
		t.Fatal("mix with fraction 1.0 expected BatchedEvaluator")
	}
	cfg.ONNXFraction = 0.0
	if _, ok := selfplayEvaluator(cfg, 0, rand.New(rand.NewSource(1)), pool).(Heuristic); !ok {
		t.Fatal("mix with fraction 0.0 expected Heuristic")
	}
}
