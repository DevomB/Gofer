package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestFeatureGoldenV2(t *testing.T) {
	b := NewBoard(9, 6.5)
	r := Chinese()
	if !r.Play(b, StoneMove(At(3, 3))) {
		t.Fatal("setup play")
	}
	sp, gl := BuildFeaturesV2(b)
	if len(sp) != featurePlanesV2*9*9 {
		t.Fatalf("spatial len %d", len(sp))
	}
	if len(gl) != featureGlobalsV2 {
		t.Fatalf("globals len %d", len(gl))
	}
	got := FeatureHashV2(sp, gl)
	path := filepath.Join("testdata", "features_golden_v2.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		SchemaVersion int    `json:"schema_version"`
		Hash          string `json:"hash"`
		BoardSize     int    `json:"board_size"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	if golden.SchemaVersion != FeatureSchemaVersion {
		t.Fatalf("schema %d", golden.SchemaVersion)
	}
	if got != golden.Hash {
		t.Fatalf("hash %s want %s", got, golden.Hash)
	}
}
