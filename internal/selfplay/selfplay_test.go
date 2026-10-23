package selfplay

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"

	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
)

func TestSelfplayWritesShards(t *testing.T) {
	dir := t.TempDir()
	if env := os.Getenv("GOFER_KEEP_SHARD"); env != "" {
		dir = env
	}
	cfg := Config{
		Size: 9, Komi: 7.5, Superko: true,
		Games: 6, Parallel: 3,
		FullVisits: 32, FastVisits: 8, FullProb: 0.25,
		TempMoves: 8,
		Search:    mcts.DefaultConfig(), Seed: 3,
		OutDir: dir, ShardGames: 4, Prefix: "test", SGFDir: filepath.Join(dir, "sgf"),
	}
	st, err := Run(cfg, nn.Uniform{})
	if err != nil {
		t.Fatal(err)
	}
	if st.Games != 6 || st.Rows == 0 || len(st.Shards) != 2 {
		t.Fatalf("stats %+v", st)
	}
	zr, err := zip.OpenReader(st.Shards[0])
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	names := map[string]bool{}
	for _, f := range zr.File {
		names[f.Name] = true
	}
	for _, want := range []string{"spatial.npy", "policy.npy", "value.npy", "ownership.npy"} {
		if !names[want] {
			t.Fatalf("shard missing %s: %v", want, names)
		}
	}
	sgfs, _ := filepath.Glob(filepath.Join(dir, "sgf", "*.sgf"))
	if len(sgfs) != 6 {
		t.Fatalf("%d sgf files", len(sgfs))
	}
}
