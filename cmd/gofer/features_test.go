package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestFeatureGolden(t *testing.T) {
	b := NewBoard(9, 6.5)
	r := Chinese()
	if !r.Play(b, StoneMove(At(3, 3))) {
		t.Fatal("setup play")
	}
	f := BuildFeaturesV1(b)
	if len(f) != featurePlanesV1*9*9+2 {
		t.Fatalf("feature len %d", len(f))
	}
	got := FeatureHash(f)
	path := filepath.Join("testdata", "features_golden.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Hash string `json:"hash"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	if got != golden.Hash {
		t.Fatalf("hash %s want %s", got, golden.Hash)
	}
}
