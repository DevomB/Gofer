//go:build onnx

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const (
	parityValueTol  = 1e-6
	parityPolicyTol = 1e-5
)

type parityRefRow struct {
	Index     int       `json:"index"`
	BoardHash uint64    `json:"board_hash"`
	Value     float64   `json:"value"`
	Policy    []float32 `json:"policy"`
}

type paritySampleRow struct {
	FeaturesSpatial []float32 `json:"features_spatial"`
	FeaturesGlobal  []float32 `json:"features_global"`
	Type            string    `json:"type"`
}

func repoRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// cmd/gofer -> repo root
	return filepath.Clean(filepath.Join(wd, "..", ".."))
}

func loadParitySamples(t *testing.T, path string, limit int) [][]float32 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var spatials [][]float32
	var globals [][]float32
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var row paritySampleRow
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		if row.Type == "header" {
			continue
		}
		if len(row.FeaturesSpatial) == 0 {
			continue
		}
		spatials = append(spatials, row.FeaturesSpatial)
		globals = append(globals, row.FeaturesGlobal)
		if len(spatials) >= limit {
			break
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	// zip into pairs for evalOne
	out := make([][]float32, 0, len(spatials)*2)
	for i := range spatials {
		out = append(out, spatials[i], globals[i])
	}
	return out
}

func loadParityRef(t *testing.T, path string) []parityRefRow {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows []parityRefRow
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var row parityRefRow
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return rows
}

func TestONNXParityPythonRef(t *testing.T) {
	root := repoRoot(t)
	model := filepath.Join(root, "models", "gofer-9x9-best.onnx")
	if _, err := os.Stat(model); err != nil {
		model = filepath.Join(root, "models", "gofer-9x9-bootstrap.onnx")
	}
	refPath := os.Getenv("GOFER_PARITY_REF")
	if refPath == "" {
		refPath = filepath.Join(root, ".tectonix", "reports", "parity-ref.jsonl")
	}
	samplesPath := os.Getenv("GOFER_PARITY_SAMPLES")
	if samplesPath == "" {
		samplesPath = filepath.Join(root, "training", "data", "samples.jsonl")
	}
	limit := 500
	if v := os.Getenv("GOFER_PARITY_LIMIT"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &limit); err != nil {
			t.Fatalf("bad GOFER_PARITY_LIMIT: %v", err)
		}
	}

	if _, err := os.Stat(refPath); err != nil {
		t.Fatalf("missing reference %s — run: scripts/parity-onnx.sh or python training/parity_harness.py", refPath)
	}
	refs := loadParityRef(t, refPath)
	if len(refs) < 100 {
		t.Fatalf("reference too small: %d rows", len(refs))
	}

	eval, err := newORTEval(model)
	if err != nil {
		t.Fatal(err)
	}
	defer eval.Close()

	pairs := loadParitySamples(t, samplesPath, len(refs))
	if len(pairs) < len(refs)*2 {
		t.Fatalf("samples %d pairs, ref %d rows", len(pairs)/2, len(refs))
	}

	var pass, fail int
	var totalGo time.Duration
	var worstValue, worstPolicy float64
	var worstIdx int
	var worstKind string

	for _, ref := range refs {
		i := ref.Index
		spatial := pairs[i*2]
		globals := pairs[i*2+1]
		t0 := time.Now()
		policy, value, err := eval.evalOne(spatial, globals)
		totalGo += time.Since(t0)
		if err != nil {
			t.Fatalf("position %d: %v", i, err)
		}
		vDiff := math.Abs(float64(value) - ref.Value)
		maxPDiff := float64(0)
		if len(policy) != len(ref.Policy) {
			fail++
			t.Errorf("position %d: policy len %d != ref %d", i, len(policy), len(ref.Policy))
			continue
		}
		for j := range policy {
			d := math.Abs(float64(policy[j] - ref.Policy[j]))
			if d > maxPDiff {
				maxPDiff = d
			}
		}
		ok := vDiff < parityValueTol && maxPDiff < parityPolicyTol
		if ok {
			pass++
		} else {
			fail++
			if vDiff >= parityValueTol && vDiff > worstValue {
				worstValue = vDiff
				worstIdx = i
				worstKind = "value"
			}
			if maxPDiff >= parityPolicyTol && maxPDiff > worstPolicy {
				worstPolicy = maxPDiff
				worstIdx = i
				worstKind = "policy"
			}
			t.Errorf("position %d FAIL value_diff=%.3e (tol %.0e) policy_max_diff=%.3e (tol %.0e)",
				i, vDiff, parityValueTol, maxPDiff, parityPolicyTol)
		}
	}

	msPer := float64(totalGo.Milliseconds()) / float64(len(refs))
	t.Logf("go_ort positions=%d pass=%d fail=%d ms_per_pos=%.4f worst_idx=%d worst_kind=%s worst_value_diff=%.3e worst_policy_diff=%.3e",
		len(refs), pass, fail, msPer, worstIdx, worstKind, worstValue, worstPolicy)
	if fail > 0 {
		t.Fatalf("%d/%d positions failed parity", fail, len(refs))
	}
}
