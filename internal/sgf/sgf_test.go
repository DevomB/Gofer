package sgf

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/DevomB/gofer/internal/game"
)

func TestReplayTestdata(t *testing.T) {
	files, _ := filepath.Glob("testdata/*.sgf")
	if len(files) == 0 {
		t.Fatal("no testdata")
	}
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		g, err := Parse(string(data))
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if _, _, err := g.Replay(); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	rec := Record{Size: 9, Komi: 7, Black: "a]b", Result: "B+R", Moves: []Move{
		{game.Black, game.Point(40)}, {game.White, game.Pass}, {game.Black, game.Point(0)},
	}}
	g, err := Parse(Write(rec))
	if err != nil {
		t.Fatal(err)
	}
	moves, err := g.MainLine()
	if err != nil {
		t.Fatal(err)
	}
	if g.Size != 9 || g.Komi != 7 || g.Result != "B+R" || g.Root.Get("PB") != "a]b" || len(moves) != 3 {
		t.Fatalf("round trip lost data: %+v %v", g, moves)
	}
	for i := range moves {
		if moves[i] != rec.Moves[i] {
			t.Fatalf("move %d = %+v want %+v", i, moves[i], rec.Moves[i])
		}
	}
}

func TestTTIsPass(t *testing.T) {
	p, err := Coord("tt", 9)
	if err != nil || p != game.Pass {
		t.Fatalf("tt on 9x9 should be pass, got %v %v", p, err)
	}
}
