//go:build onnx_integration

package main

import (
	"net/http"
	"os"
	"testing"
	"time"
)

func TestSidecarLive(t *testing.T) {
	url := os.Getenv("GOFER_ONNX_URL")
	if url == "" {
		url = "http://127.0.0.1:8080"
	}
	client := &http.Client{Timeout: 2 * time.Second}
	if err := CheckSidecarHealth(url, client); err != nil {
		t.Skip("sidecar not running:", err)
	}
	b := NewBoard(9, 6.5)
	backend := SidecarBackend{URL: url, Client: client}
	out := backend.EvalBatch([]*Board{b})
	if len(out) != 1 || len(out[0].Policy) != 82 {
		t.Fatalf("unexpected result: %+v", out)
	}
}
