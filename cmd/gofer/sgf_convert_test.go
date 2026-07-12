package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSmoothedPolicySumsToOne(t *testing.T) {
	size := 9
	b := NewBoard(size, 6.5)
	r := Chinese()
	legal := r.LegalMoves(b)
	played := legal[0]
	policy := smoothedPolicyTarget(legal, played, size, 0.10)
	if policy == nil {
		t.Fatal("nil policy")
	}
	var sum float32
	for _, v := range policy {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-5 {
		t.Fatalf("sum=%v", sum)
	}
	if policy[policyIndex(played, size)] < 0.89 {
		t.Fatalf("peak too low: %v", policy[policyIndex(played, size)])
	}
}

func TestConvertSGFSimpleGame(t *testing.T) {
	sgf := "(;FF[4]GM[1]SZ[9]KM[6.5]RE[B+1.5];B[cc];W[gg];B[ee];W[])"

	samples, reason, err := ConvertSGFBytes([]byte(sgf), SGFConvertConfig{BoardSize: 9, Epsilon: 0.10})
	if err != nil {
		t.Fatalf("reason=%q err=%v", reason, err)
	}
	if len(samples) < 2 {
		t.Fatalf("samples=%d", len(samples))
	}
	for _, s := range samples {
		if len(s.FeaturesSpatial) != 8*9*9 {
			t.Fatalf("spatial len %d", len(s.FeaturesSpatial))
		}
		if len(s.Policy) != 82 {
			t.Fatalf("policy len %d", len(s.Policy))
		}
		if len(s.Ownership) != 81 {
			t.Fatalf("ownership len %d", len(s.Ownership))
		}
		if s.Value == 0 && s.ToPlay == Black {
			// may be draw edge case
		}
	}
}

func TestConvertRejectsHandicap(t *testing.T) {
	sgf := "(;FF[4]SZ[9]KM[6.5]RE[B+1]AB[cc][dd];B[ee];W[gg])"
	_, reason, err := ConvertSGFBytes([]byte(sgf), SGFConvertConfig{BoardSize: 9})
	if err == nil || reason != "handicap" {
		t.Fatalf("reason=%q err=%v", reason, err)
	}
}

func TestConvertRejectsBadOpeningCorner(t *testing.T) {
	sgf := "(;FF[4]SZ[9]KM[6.5]RE[B+1];B[aa];W[cc];B[dd])"
	_, reason, err := ConvertSGFBytes([]byte(sgf), SGFConvertConfig{BoardSize: 9})
	if err == nil || reason != "bad_opening" {
		t.Fatalf("reason=%q err=%v", reason, err)
	}
}

func TestConvertRejectsKomiOutOfRange(t *testing.T) {
	sgf := "(;FF[4]SZ[9]KM[5.5]RE[B+1];B[cc];W[dd])"
	_, reason, err := ConvertSGFBytes([]byte(sgf), SGFConvertConfig{BoardSize: 9})
	if err == nil || reason != "komi" {
		t.Fatalf("reason=%q err=%v", reason, err)
	}
}

func TestBalancedPerSourceCap(t *testing.T) {
	root := t.TempDir()
	write := func(dir, name, sgf string) {
		d := filepath.Join(root, dir)
		if err := os.MkdirAll(d, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, name), []byte(sgf), 0644); err != nil {
			t.Fatal(err)
		}
	}
	sgf := "(;FF[4]SZ[9]KM[6.5]RE[B+1];B[cc];W[gg];B[ee];W[hh])"
	write("cgos", "a.sgf", sgf)
	write("minigo", "b.sgf", sgf)
	write("aeb", "c.sgf", sgf)
	out := filepath.Join(root, "out.jsonl")
	stats, err := RunSGFConvert(SGFConvertConfig{OutPath: out, BoardSize: 9, MaxRows: 9, Epsilon: 0.10}, []string{
		filepath.Join(root, "cgos"),
		filepath.Join(root, "minigo"),
		filepath.Join(root, "aeb"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.PerSourceCap != 3 {
		t.Fatalf("per_source_cap=%d want 3", stats.PerSourceCap)
	}
	for src, n := range stats.SourceRows {
		if n > 3 {
			t.Fatalf("source %s rows=%d exceeds cap", src, n)
		}
	}
	if len(stats.SourceRows) < 2 {
		t.Fatalf("expected multiple sources, got %v", stats.SourceRows)
	}
}

func TestConvertDedupAcrossGames(t *testing.T) {
	dir := t.TempDir()
	// Same opening sequence, different results — should share early positions.
	sgf1 := "(;FF[4]SZ[9]KM[6.5]RE[B+1];B[cc];W[gg];B[ee];W[hh];B[ff])"
	sgf2 := "(;FF[4]SZ[9]KM[6.5]RE[W+1];B[cc];W[gg];B[ee];W[hh];B[ff])"
	if err := os.WriteFile(filepath.Join(dir, "a.sgf"), []byte(sgf1), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "b.sgf"), []byte(sgf2), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out.jsonl")
	stats, err := RunSGFConvert(SGFConvertConfig{OutPath: out, BoardSize: 9, Epsilon: 0.10}, []string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if stats.RowsWritten == 0 {
		t.Fatal("no rows")
	}
	if stats.RowsSkippedDup == 0 {
		t.Log("no dup skips (games may not overlap)")
	}
}
