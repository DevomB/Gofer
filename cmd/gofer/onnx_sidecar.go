package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
)

const sidecarEvalPath = "/v1/eval"

// SidecarStats counts inference outcomes (for latency reports).
type SidecarStats struct {
	Requests  atomic.Uint64
	Fallbacks atomic.Uint64
	Timeouts  atomic.Uint64
}

// SidecarBackend calls a Python ONNX Runtime HTTP sidecar.
type SidecarBackend struct {
	URL      string
	Client   *http.Client
	Fallback Evaluator
	Stats    *SidecarStats
}

func (s SidecarBackend) fallback() Evaluator {
	if s.Fallback != nil {
		return s.Fallback
	}
	return Heuristic{}
}

func (s SidecarBackend) httpClient() *http.Client {
	if s.Client != nil {
		return s.Client
	}
	return http.DefaultClient
}

func (s SidecarBackend) stats() *SidecarStats {
	if s.Stats != nil {
		return s.Stats
	}
	return &SidecarStats{}
}

type sidecarEvalReq struct {
	SchemaVersion int         `json:"schema_version"`
	BatchSize     int         `json:"batch_size"`
	Spatial       [][]float32 `json:"spatial"`
	Globals       [][]float32 `json:"globals"`
}

type sidecarEvalResp struct {
	Results []struct {
		Value  float64   `json:"value"`
		Policy []float32 `json:"policy"`
	} `json:"results"`
}

// EvalBatch implements EvalBackend.
func (s SidecarBackend) EvalBatch(boards []*Board) []Result {
	out := make([]Result, len(boards))
	fb := s.fallback()
	st := s.stats()
	if len(boards) == 0 {
		return out
	}
	st.Requests.Add(uint64(len(boards)))

	reqBody := sidecarEvalReq{
		SchemaVersion: FeatureSchemaVersion,
		BatchSize:     len(boards),
		Spatial:       make([][]float32, len(boards)),
		Globals:       make([][]float32, len(boards)),
	}
	for i, b := range boards {
		sp, gl := BuildFeaturesV2(b)
		reqBody.Spatial[i] = sp
		reqBody.Globals[i] = gl
	}
	data, err := json.Marshal(reqBody)
	if err != nil {
		st.Fallbacks.Add(uint64(len(boards)))
		for i, b := range boards {
			out[i] = fb.Evaluate(b)
		}
		return out
	}

	url := s.URL + sidecarEvalPath
	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		st.Fallbacks.Add(uint64(len(boards)))
		for i, b := range boards {
			out[i] = fb.Evaluate(b)
		}
		return out
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(httpReq)
	if err != nil {
		st.Fallbacks.Add(uint64(len(boards)))
		for i, b := range boards {
			out[i] = fb.Evaluate(b)
		}
		return out
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode != http.StatusOK {
		st.Fallbacks.Add(uint64(len(boards)))
		for i, b := range boards {
			out[i] = fb.Evaluate(b)
		}
		return out
	}

	var parsed sidecarEvalResp
	if err := json.Unmarshal(body, &parsed); err != nil {
		st.Fallbacks.Add(uint64(len(boards)))
		for i, b := range boards {
			out[i] = fb.Evaluate(b)
		}
		return out
	}

	for i, b := range boards {
		want := b.Size()*b.Size() + 1
		if i >= len(parsed.Results) {
			st.Fallbacks.Add(1)
			out[i] = fb.Evaluate(b)
			continue
		}
		r := parsed.Results[i]
		if len(r.Policy) != want {
			st.Fallbacks.Add(1)
			out[i] = fb.Evaluate(b)
			continue
		}
		out[i] = Result{Value: r.Value, Policy: r.Policy}
	}
	return out
}

// CheckSidecarHealth GETs /health and returns an error if unreachable.
func CheckSidecarHealth(url string, client *http.Client) error {
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Get(url + "/health")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("sidecar health: HTTP %d", resp.StatusCode)
	}
	return nil
}
