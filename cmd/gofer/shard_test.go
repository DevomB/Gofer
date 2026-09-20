package main

import (
	"archive/zip"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func readShardEntries(t *testing.T, path string) map[string][]byte {
	t.Helper()
	zr, err := zip.OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	out := map[string][]byte{}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			t.Fatal(err)
		}
		out[strings.TrimSuffix(f.Name, ".npy")] = b
	}
	return out
}

// npyPayload splits a .npy blob into header dict and data, checking alignment.
func npyPayload(t *testing.T, blob []byte) (string, []byte) {
	t.Helper()
	if string(blob[:6]) != "\x93NUMPY" || blob[6] != 1 {
		t.Fatalf("bad npy magic %q", blob[:8])
	}
	hlen := int(binary.LittleEndian.Uint16(blob[8:10]))
	if (10+hlen)%64 != 0 {
		t.Fatalf("npy header not 64-byte aligned: %d", 10+hlen)
	}
	return string(blob[10 : 10+hlen]), blob[10+hlen:]
}

func TestNPYHeaderShapes(t *testing.T) {
	h := string(npyHeader("<f4", []int{7}))
	if !strings.Contains(h, "'shape': (7,)") || !strings.HasSuffix(h, "\n") {
		t.Fatalf("1-D header wrong: %q", h)
	}
	h = string(npyHeader("|u1", []int{3, 8, 9, 9}))
	if !strings.Contains(h, "'shape': (3, 8, 9, 9)") {
		t.Fatalf("4-D header wrong: %q", h)
	}
}

func TestWriteSampleShardRoundTrip(t *testing.T) {
	cfg := testSelfplayConfig("heuristic", 2)
	cfg.CapRandomizeP = 0.5
	samples, _ := RunSelfplayWithLogs(cfg)
	if len(samples) == 0 {
		t.Fatal("no samples")
	}
	path := filepath.Join(t.TempDir(), "s.npz")
	if err := WriteSampleShard(path, samples, ShardMeta{Komi: cfg.Komi, Seed: cfg.Seed, Model: "heuristic"}); err != nil {
		t.Fatal(err)
	}
	entries := readShardEntries(t, path)
	for _, name := range []string{"spatial", "globals", "policy", "policy_opp", "value", "score", "ownership", "full_search", "game_id", "move_num", "meta"} {
		if _, ok := entries[name]; !ok {
			t.Fatalf("missing array %s", name)
		}
	}
	n := len(samples)

	_, metaRaw := npyPayload(t, entries["meta"])
	var meta ShardMeta
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		t.Fatal(err)
	}
	if meta.Format != ShardFormat || meta.Rows != n || meta.BoardSize != 9 || meta.Games != 2 {
		t.Fatalf("meta mismatch: %+v", meta)
	}

	hdr, spatial := npyPayload(t, entries["spatial"])
	if !strings.Contains(hdr, "'descr': '|u1'") || len(spatial) != n*8*81 {
		t.Fatalf("spatial: %q len=%d", hdr, len(spatial))
	}
	_, own := npyPayload(t, entries["ownership"])
	_, score := npyPayload(t, entries["score"])
	for i, s := range samples {
		if spatial[i*8*81] != byte(s.FeaturesSpatial[0]) {
			t.Fatalf("row %d spatial mismatch", i)
		}
		// Side-to-move frame: sum(ownership) minus/plus komi is exactly the margin.
		sum := 0.0
		for p := 0; p < 81; p++ {
			sum += float64(int8(own[i*81+p]))
		}
		komi := cfg.Komi
		if s.ToPlay == White {
			komi = -komi
		}
		got := math.Float32frombits(binary.LittleEndian.Uint32(score[i*4:]))
		if float32(sum-komi) != got {
			t.Fatalf("row %d: ownership sum %.1f komi %.1f != score %.1f", i, sum, komi, got)
		}
	}
}

func TestLabelGameSamplesPolicyNext(t *testing.T) {
	game := []Sample{
		{ToPlay: Black, Policy: []float32{1, 0}, FullSearch: true},
		{ToPlay: White, Policy: []float32{0, 1}, FullSearch: true},
		{ToPlay: Black, Policy: []float32{1, 0}, FullSearch: false},
	}
	labelGameSamples(game, 10, 5, nil)
	if len(game[0].PolicyNext) != 2 || game[0].PolicyNext[1] != 1 {
		t.Fatalf("row 0 policy_next should be row 1 policy: %v", game[0].PolicyNext)
	}
	if game[1].PolicyNext != nil {
		t.Fatal("row 1 next ply is a fast search; policy_next must be empty")
	}
	if game[0].ScoreMargin != 5 || game[1].ScoreMargin != -5 {
		t.Fatalf("score margin perspective wrong: %v %v", game[0].ScoreMargin, game[1].ScoreMargin)
	}
}

// validShardSample is a minimal row that passes every buildShardArrays check,
// so a test can invalidate exactly one field and know that is what failed.
func validShardSample(size int) Sample {
	points := size * size
	return Sample{
		ToPlay:          Black,
		Policy:          make([]float32, points+1),
		FeaturesSpatial: make([]float32, 8*points),
		FeaturesGlobal:  make([]float32, 4),
		Ownership:       make([]float32, points),
	}
}

func TestWriteSampleShardRejectsMixedSizes(t *testing.T) {
	a, b := validShardSample(9), validShardSample(13)
	err := WriteSampleShard(filepath.Join(t.TempDir(), "x.npz"), []Sample{a, b}, ShardMeta{})
	if err == nil {
		t.Fatal("expected mixed-size error")
	}
	if !strings.Contains(err.Error(), "shape differs") {
		t.Fatalf("want the shape error, got %v", err)
	}
}

// A row without ownership must not be written as zeros: the learner applies its
// ownership loss unmasked, so zeros train the head toward "neutral everywhere".
func TestWriteSampleShardRejectsMissingOwnership(t *testing.T) {
	for _, tc := range []struct {
		name string
		own  []float32
	}{
		{"absent", nil},
		{"empty", []float32{}},
		{"wrong length", make([]float32, 80)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := validShardSample(9)
			s.Ownership = tc.own
			err := WriteSampleShard(filepath.Join(t.TempDir(), "x.npz"), []Sample{s}, ShardMeta{})
			if err == nil {
				t.Fatal("expected an ownership error, shard was written")
			}
			if !strings.Contains(err.Error(), "ownership") {
				t.Fatalf("want an ownership error, got %v", err)
			}
		})
	}
}

func TestWriteSampleShardRejectsNonFiniteFloats(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	for _, tc := range []struct {
		name  string
		field string
		spoil func(*Sample)
	}{
		{"value NaN", "value", func(s *Sample) { s.Value = nan }},
		{"score +Inf", "score", func(s *Sample) { s.ScoreMargin = inf }},
		{"score -Inf", "score", func(s *Sample) { s.ScoreMargin = float32(math.Inf(-1)) }},
		{"policy NaN", "policy", func(s *Sample) { s.Policy[3] = nan }},
		{"globals NaN", "features_global", func(s *Sample) { s.FeaturesGlobal[1] = nan }},
		{"policy_opp NaN", "policy_opp", func(s *Sample) {
			s.PolicyNext = make([]float32, len(s.Policy))
			s.PolicyNext[0] = nan
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := validShardSample(9)
			tc.spoil(&s)
			path := filepath.Join(t.TempDir(), "x.npz")
			err := WriteSampleShard(path, []Sample{s}, ShardMeta{})
			if err == nil {
				t.Fatal("expected a non-finite error, shard was written")
			}
			if !strings.Contains(err.Error(), "non-finite") || !strings.Contains(err.Error(), tc.field) {
				t.Fatalf("want a non-finite %s error, got %v", tc.field, err)
			}
			if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
				t.Fatal("a rejected shard must not be left on disk")
			}
		})
	}
}

// The validation must not reject anything self-play legitimately produces.
func TestWriteSampleShardAcceptsRealSelfplayRows(t *testing.T) {
	cfg := testSelfplayConfig("heuristic", 2)
	samples, _ := RunSelfplayWithLogs(cfg)
	if len(samples) == 0 {
		t.Fatal("no samples")
	}
	if err := WriteSampleShard(filepath.Join(t.TempDir(), "s.npz"), samples, ShardMeta{}); err != nil {
		t.Fatalf("self-play rows must pass validation: %v", err)
	}
}
