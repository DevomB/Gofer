package main

import (
	"bufio"
	"encoding/json"
	"os"
	"runtime/debug"
)

const SampleSchemaVersion = 1

// Sample is one self-play training record.
type Sample struct {
	BoardHash  uint64    `json:"board_hash"`
	MoveNum    int       `json:"move_num"`
	Policy     []float32 `json:"policy"`
	PolicyOpp  []float32 `json:"policy_opp,omitempty"`
	FeaturesSpatial []float32 `json:"features_spatial,omitempty"`
	FeaturesGlobal  []float32 `json:"features_global,omitempty"`
	ToPlay     Color     `json:"to_play"`
	Value      float32   `json:"value"`
	Komi       float64   `json:"komi,omitempty"`
	Ownership  []float32 `json:"ownership,omitempty"`
	FullSearch bool      `json:"full_search,omitempty"`
	ScorePDF   []float64 `json:"score_pdf,omitempty"`
	ScoreCDF   []float64 `json:"score_cdf,omitempty"`
}

// SampleExport wraps training rows with schema metadata.
type SampleExport struct {
	SchemaVersion int      `json:"schema_version"`
	GitCommit     string   `json:"git_commit,omitempty"`
	Samples       []Sample `json:"samples"`
}

func buildInfoVersion() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		return info.Main.Version
	}
	return ""
}

// WriteSampleJSONL writes one JSON object per line with a header line.
func WriteSampleJSONL(path string, samples []Sample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	header, err := json.Marshal(map[string]any{
		"schema_version": SampleSchemaVersion,
		"git_commit":     buildInfoVersion(),
		"type":           "header",
	})
	if err != nil {
		return err
	}
	if _, err := w.Write(header); err != nil {
		return err
	}
	if _, err := w.WriteString("\n"); err != nil {
		return err
	}
	for _, s := range samples {
		line, err := json.Marshal(s)
		if err != nil {
			return err
		}
		if _, err := w.Write(line); err != nil {
			return err
		}
		if _, err := w.WriteString("\n"); err != nil {
			return err
		}
	}
	return w.Flush()
}

// MarshalSampleExport returns pretty JSON with schema wrapper.
func MarshalSampleExport(samples []Sample) ([]byte, error) {
	return json.MarshalIndent(SampleExport{
		SchemaVersion: SampleSchemaVersion,
		GitCommit:     buildInfoVersion(),
		Samples:       samples,
	}, "", "  ")
}
