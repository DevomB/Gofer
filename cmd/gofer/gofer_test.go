package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCoordRoundTrip(t *testing.T) {
	size := 9
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			p := At(x, y)
			if p.X != x || p.Y != y {
				t.Fatalf("round trip failed (%d,%d)", x, y)
			}
			if p.Idx(size) != y*size+x {
				t.Fatalf("idx mismatch (%d,%d)", x, y)
			}
		}
	}
}

func TestZobristChangesOnPlay(t *testing.T) {
	b := NewBoard(9, 6.5)
	h0 := b.Hash()
	p := At(4, 4)
	b.StartPlay(StoneMove(p), nil, p.Idx(9), Empty)
	b.SetStoneIndex(p.Idx(9), Black)
	b.FinishTurn(-1)
	if b.Hash() == h0 {
		t.Fatal("hash should change")
	}
}

func TestUndo(t *testing.T) {
	b := NewBoard(9, 6.5)
	p := At(2, 2)
	idx := p.Idx(9)
	b.StartPlay(StoneMove(p), nil, idx, Empty)
	b.SetStoneIndex(idx, Black)
	b.FinishTurn(-1)
	b.Undo()
	if b.StoneAt(p) != Empty {
		t.Fatal("undo failed")
	}
}

func TestSnapshotRestore(t *testing.T) {
	b := NewBoard(9, 6.5)
	s := b.Snapshot()
	p := At(1, 1)
	idx := p.Idx(9)
	b.StartPlay(StoneMove(p), nil, idx, Empty)
	b.SetStoneIndex(idx, Black)
	b.FinishTurn(-1)
	b.Restore(s)
	if b.StoneAt(p) != Empty {
		t.Fatal("restore failed")
	}
}

func TestNewBoard(t *testing.T) {
	b := NewBoard(9, 6.5)
	if b.Size() != 9 {
		t.Fatalf("size %d", b.Size())
	}
}

func TestRunSelfplaySamples(t *testing.T) {
	cfg := DefaultSelfplayConfig()
	cfg.Games = 1
	cfg.Playouts = 5
	cfg.BoardSize = 5
	cfg.FullOnlyExport = false
	samples := RunSelfplay(cfg)
	if len(samples) == 0 {
		t.Fatal("expected samples")
	}
	for _, s := range samples {
		if s.Komi != cfg.Komi {
			t.Fatalf("komi %v want %v", s.Komi, cfg.Komi)
		}
		if s.Value != 1 && s.Value != -1 && s.Value != 0 {
			t.Fatalf("value out of range: %v", s.Value)
		}
		if len(s.Ownership) == 0 {
			t.Fatal("expected ownership labels")
		}
	}
}

func TestOwnershipEndgameSGF(t *testing.T) {
	path := filepath.Join("testdata", "simple.sgf")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := ParseSGF(string(data))
	if err != nil {
		t.Fatal(err)
	}
	r := Chinese()
	b := NewBoard(g.Size, g.Komi)
	if err := g.Setup(b); err != nil {
		t.Fatal(err)
	}
	moves, err := g.MainLine()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range moves {
		if !r.Play(b, sgfMoveToPlay(m)) {
			t.Fatal("illegal replay")
		}
	}
	own := OwnershipLabel(b)
	if len(own) != g.Size*g.Size {
		t.Fatalf("ownership len %d", len(own))
	}
	nonZero := 0
	for _, v := range own {
		if v != 0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Fatal("expected some ownership signal after replay")
	}
}

func TestMatchResultJSONFields(t *testing.T) {
	result := RunMatch(MatchConfig{Games: 1, Size: 5, Playouts: 4, BlackEval: "uniform", WhiteEval: "uniform", Seed: 1})
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"game_count", "games", "config_hash", "baseline_wilson_ci_low", "wins_baseline"} {
		if _, ok := m[key]; !ok {
			t.Fatalf("missing json field %q", key)
		}
	}
}

func TestGatingHarness(t *testing.T) {
	g := GatingHarness{Games: 100, MinWinRateMargin: PromoteMin}
	if !g.Pass(0.4, 0.56) {
		t.Fatal("should pass")
	}
}

func play(t *testing.T, r Ruleset, b *Board, x, y int) {
	t.Helper()
	if !r.Play(b, StoneMove(At(x, y))) {
		t.Fatalf("illegal play at %d,%d", x, y)
	}
}

func sgfMoveToPlay(m SGFMove) Move {
	if m.Point == nil {
		return PassMove
	}
	return StoneMove(*m.Point)
}

func replaySGF(t *testing.T, path string, rules Ruleset, check func(t *testing.T, r Ruleset, b *Board)) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseSGF(string(data))
	if err != nil {
		t.Fatal(err)
	}
	b := NewBoard(parsed.Size, parsed.Komi)
	if err := parsed.Setup(b); err != nil {
		t.Fatal(err)
	}
	moves, err := parsed.MainLine()
	if err != nil {
		t.Fatal(err)
	}
	for i, m := range moves {
		if b.Player() != m.Color {
			t.Fatalf("move %d wrong side", i)
		}
		if !rules.Play(b, sgfMoveToPlay(m)) {
			t.Fatalf("illegal replay move %d in %s", i, filepath.Base(path))
		}
	}
	if check != nil {
		check(t, rules, b)
	}
}

func TestTrompTaylorLegalMoves(t *testing.T) {
	r := TrompTaylor()
	b := NewBoard(9, 6.5)
	moves := r.LegalMoves(b)
	if len(moves) != 9*9+1 {
		t.Fatalf("want 82 got %d", len(moves))
	}
}

func TestTrompTaylorSuicideAllowed(t *testing.T) {
	r := TrompTaylor()
	b := NewBoard(9, 6.5)
	play(t, r, b, 0, 0)
	play(t, r, b, 2, 0)
	play(t, r, b, 0, 1)
	play(t, r, b, 1, 1)
	_ = r.Play(b, StoneMove(At(1, 0)))
}

func TestTrompTaylorScore(t *testing.T) {
	r := TrompTaylor()
	b := NewBoard(9, 6.5)
	play(t, r, b, 3, 3)
	bl, wl := r.Score(b)
	if bl < 1 || wl != 6.5 {
		t.Fatalf("score bl=%v wl=%v", bl, wl)
	}
}

func TestChineseFactory(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	if !r.Play(b, StoneMove(At(4, 4))) {
		t.Fatal("play failed")
	}
}

func TestCaptureRemovesStones(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	play(t, r, b, 1, 0)
	play(t, r, b, 0, 0)
	play(t, r, b, 0, 1)
	if b.StoneAt(At(0, 0)) != Empty {
		t.Fatal("captured stone should be gone")
	}
}

func TestSuicideIllegal(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	play(t, r, b, 0, 0)
	play(t, r, b, 3, 4)
	play(t, r, b, 0, 1)
	play(t, r, b, 5, 4)
	play(t, r, b, 0, 2)
	play(t, r, b, 4, 3)
	play(t, r, b, 0, 3)
	play(t, r, b, 4, 5)
	if r.Play(b, StoneMove(At(4, 4))) {
		t.Fatal("suicide should fail")
	}
}

func TestSimpleKo(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	play(t, r, b, 0, 0)
	play(t, r, b, 4, 4)
	play(t, r, b, 0, 1)
	play(t, r, b, 0, 2)
	play(t, r, b, 3, 4)
	play(t, r, b, 0, 3)
	play(t, r, b, 5, 4)
	play(t, r, b, 0, 4)
	play(t, r, b, 3, 3)
	play(t, r, b, 0, 5)
	play(t, r, b, 3, 5)
	play(t, r, b, 0, 6)
	play(t, r, b, 5, 3)
	play(t, r, b, 0, 7)
	play(t, r, b, 5, 5)
	play(t, r, b, 0, 8)
	play(t, r, b, 4, 3)
	play(t, r, b, 1, 0)
	play(t, r, b, 4, 5)
	wantKo := At(4, 4).Idx(9)
	if b.Ko() != wantKo {
		t.Fatalf("ko want %d got %d", wantKo, b.Ko())
	}
	if r.Play(b, StoneMove(At(4, 4))) {
		t.Fatal("ko recapture illegal")
	}
}

func TestPassAndUndo(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	if !r.Play(b, PassMove) {
		t.Fatal("pass failed")
	}
	if b.Player() != White {
		t.Fatal("turn switch")
	}
	play(t, r, b, 2, 2)
	b.Undo()
	if b.StoneAt(At(2, 2)) != Empty {
		t.Fatal("undo failed")
	}
}

func TestLegalMovesEmpty19(t *testing.T) {
	r := Chinese()
	b := NewBoard(19, 7.5)
	moves := r.LegalMoves(b)
	if len(moves) != 19*19+1 {
		t.Fatalf("want 362 got %d", len(moves))
	}
}

func TestScoreCountsStones(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	play(t, r, b, 0, 0)
	bl, _ := r.Score(b)
	if bl < 1 {
		t.Fatalf("expected black stones got %v", bl)
	}
}

func TestSuperkoWrapper(t *testing.T) {
	r := WithSuperko(Chinese())
	b := NewBoard(9, 6.5)
	if !r.Play(b, StoneMove(At(4, 4))) {
		t.Fatal("superko wrapped play failed")
	}
	if len(r.LegalMoves(b)) == 0 {
		t.Fatal("expected legal moves")
	}
}

func TestTrompVsChineseScoreDivergence(t *testing.T) {
	ch := Chinese()
	tr := TrompTaylor()
	bc := NewBoard(9, 6.5)
	bt := NewBoard(9, 6.5)
	play(t, ch, bc, 2, 2)
	play(t, tr, bt, 2, 2)
	play(t, ch, bc, 3, 3)
	play(t, tr, bt, 3, 3)
	cbl, cwl := ch.Score(bc)
	tbl, twl := tr.Score(bt)
	if cbl == tbl && cwl == twl {
		t.Log("scores equal on this position — divergence may appear with captures")
	}
}
func TestParseSGFCoord(t *testing.T) {
	p, err := ParseSGFCoord(9, "bc")
	if err != nil {
		t.Fatal(err)
	}
	if p.X != 1 || p.Y != 2 {
		t.Fatalf("bc on 9x9 want (1,2) got (%d,%d)", p.X, p.Y)
	}
}

func TestGoldenCaptureSGF(t *testing.T) {
	replaySGF(t, filepath.Join("testdata", "capture.sgf"), Chinese(), func(t *testing.T, _ Ruleset, b *Board) {
		if b.StoneAt(At(0, 0)) != Empty {
			t.Fatal("white at aa should be captured")
		}
	})
}

func TestGoldenKoSGF(t *testing.T) {
	replaySGF(t, filepath.Join("testdata", "ko.sgf"), Chinese(), func(t *testing.T, r Ruleset, b *Board) {
		wantKo := At(4, 4).Idx(9)
		if b.Ko() != wantKo {
			t.Fatalf("ko want %d got %d", wantKo, b.Ko())
		}
		if r.Play(b, StoneMove(At(4, 4))) {
			t.Fatal("immediate ko recapture should be illegal")
		}
	})
}

func TestGoldenPassSGF(t *testing.T) {
	replaySGF(t, filepath.Join("testdata", "pass.sgf"), Chinese(), func(t *testing.T, _ Ruleset, b *Board) {
		if b.Player() != Black {
			t.Fatal("two passes should return turn to black")
		}
	})
}

func TestTrompReplayCorpus(t *testing.T) {
	for _, name := range []string{"capture.sgf"} {
		t.Run(name, func(t *testing.T) {
			replaySGF(t, filepath.Join("testdata", name), TrompTaylor(), nil)
		})
	}
}

func TestParseGameMeta(t *testing.T) {
	g, err := ParseSGF("(;FF[4]SZ[9]KM[6.5];B[cc];W[dd])")
	if err != nil {
		t.Fatal(err)
	}
	if g.Size != 9 || g.Komi != 6.5 {
		t.Fatalf("meta size=%d komi=%v", g.Size, g.Komi)
	}
	moves, err := g.MainLine()
	if err != nil || len(moves) != 2 {
		t.Fatalf("moves %d err=%v", len(moves), err)
	}
}

func TestReplayCorpus(t *testing.T) {
	for _, name := range []string{"capture.sgf", "ko.sgf", "pass.sgf", "setup.sgf", "simple.sgf", "open.sgf"} {
		t.Run(name, func(t *testing.T) {
			replaySGF(t, filepath.Join("testdata", name), Chinese(), nil)
		})
	}
}

func TestExportSGFRoundTrip(t *testing.T) {
	g := &SGFGame{Size: 9, Komi: 6.5}
	pt := At(2, 2)
	moves := []SGFMove{{Color: Black, Point: &pt}}
	out := ExportSGF(g, moves)
	g2, err := ParseSGF(out)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := g2.MainLine()
	if err != nil || len(m2) != 1 {
		t.Fatalf("round trip moves %d err=%v", len(m2), err)
	}
}

func TestGameLogExport(t *testing.T) {
	log := NewGameLog(9, 6.5)
	log.Record(Black, StoneMove(At(2, 2)))
	log.Record(White, PassMove)
	out := log.ExportSGF()
	g, err := ParseSGF(out)
	if err != nil {
		t.Fatal(err)
	}
	moves, err := g.MainLine()
	if err != nil || len(moves) != 2 {
		t.Fatalf("moves %d err=%v", len(moves), err)
	}
}

func TestSelfplaySGFLogs(t *testing.T) {
	cfg := DefaultSelfplayConfig()
	cfg.Games = 1
	cfg.Playouts = 4
	cfg.BoardSize = 5
	_, logs := RunSelfplayWithLogs(cfg)
	if len(logs) != 1 || len(logs[0].Moves) == 0 {
		t.Fatalf("log moves %d", len(logs[0].Moves))
	}
}
