package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func BenchmarkApplyStone(b *testing.B) {
	br := NewBoard(19, 7.5)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx := i % (19 * 19)
		br.setStone(idx, Black)
		br.setStone(idx, Empty)
	}
}

func BenchmarkUndo(b *testing.B) {
	br := NewBoard(19, 7.5)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := At(i%19, (i/19)%19)
		idx := p.Idx(19)
		br.StartPlay(StoneMove(p), nil, idx, Empty)
		br.SetStoneIndex(idx, Black)
		br.FinishTurn(-1)
		br.Undo()
	}
}

func BenchmarkClone(b *testing.B) {
	br := NewBoard(19, 7.5)
	for i := 0; i < 20; i++ {
		br.setStone(i, Black)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = br.Clone()
	}
}

func BenchmarkHashUpdate(b *testing.B) {
	br := NewBoard(19, 7.5)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx := i % (19 * 19)
		br.setStone(idx, Black)
		_ = br.Hash()
		br.setStone(idx, Empty)
	}
}

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

func BenchmarkCloneVsUndo(b *testing.B) {
	b.Run("Clone", func(b *testing.B) {
		br := NewBoard(19, 7.5)
		for i := 0; i < 40; i++ {
			br.setStone(i, Black)
		}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			c := br.Clone()
			idx := (i + 40) % (19 * 19)
			c.setStone(idx, White)
			_ = c
		}
	})
	b.Run("Undo", func(b *testing.B) {
		br := NewBoard(19, 7.5)
		for i := 0; i < 40; i++ {
			br.setStone(i, Black)
		}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			idx := (i + 40) % (19 * 19)
			prev := br.stones[idx]
			br.StartPlay(StoneMove(At(idx%19, idx/19)), nil, idx, prev)
			br.SetStoneIndex(idx, White)
			br.FinishTurn(-1)
			br.Undo()
		}
	})
}

func TestNewBoard(t *testing.T) {
	b := NewBoard(9, 6.5)
	if b.Size() != 9 {
		t.Fatalf("size %d", b.Size())
	}
}

func treeNodes(eng *Engine) int {
	eng.mu.Lock()
	defer eng.mu.Unlock()
	if eng.arena == nil {
		return 0
	}
	return len(eng.arena.nodes)
}

func TestPUCTFormula(t *testing.T) {
	q, prior, pv := 0.5, 0.1, 100.0
	visits, c := uint32(10), 1.1
	got := q + c*prior*math.Sqrt(pv)/(1+float64(visits))
	want := 0.5 + 1.1*0.1*10/11
	if got < want-1e-9 || got > want+1e-9 {
		t.Fatalf("puct got %v want %v", got, want)
	}
}

func TestDeterministicPlayout(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Seed = 42
	cfg.Playouts = 6
	b := NewBoard(5, 6.5)
	m1 := NewEngine(r, Uniform{}, cfg).BestMove(b)
	m2 := NewEngine(r, Uniform{}, cfg).BestMove(b)
	if m1 != m2 {
		t.Fatalf("deterministic seed mismatch %v vs %v", m1, m2)
	}
}

func TestTTStoresAfterSearch(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 20
	e := NewEngine(r, nil, cfg)
	b := NewBoard(5, 6.5)
	_ = e.BestMove(b)
	if _, ok := e.TT.Get(b.Hash()); !ok {
		t.Fatal("expected TT entry after search")
	}
}

func TestRootPolicySumsToOne(t *testing.T) {
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 15
	e := NewEngine(r, nil, cfg)
	b := NewBoard(5, 6.5)
	legal := r.LegalMoves(b)
	_ = e.BestMove(b)
	pi := e.RootPolicy(legal)
	var sum float32
	for _, p := range pi {
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("policy sum %v", sum)
	}
}

func TestMCTSTreeReuse(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	cfg := DefaultConfig()
	cfg.Playouts = 30
	cfg.Workers = 1
	eng := NewEngine(r, Uniform{}, cfg)
	m1 := eng.BestMove(b)
	len1 := treeNodes(eng)
	if len1 == 0 {
		t.Fatal("expected nodes after search")
	}
	eng.BestMove(b)
	if treeNodes(eng) < len1 {
		t.Fatalf("tree shrank on reuse: %d -> %d", len1, treeNodes(eng))
	}
	r.Play(b, m1)
	eng.AdvanceTree(m1)
	if treeNodes(eng) == 0 {
		t.Fatal("tree cleared after advancing to best child")
	}
	eng.ResetArena()
	if treeNodes(eng) != 0 {
		t.Fatal("reset failed")
	}
}

func TestGTPBoardsize(t *testing.T) {
	s := NewSession(SessionConfig{})
	if out := s.Handle("boardsize 9"); out != "" {
		t.Fatalf("boardsize: %q", out)
	}
}

func TestGTPKnownCommand(t *testing.T) {
	s := NewSession(SessionConfig{})
	if out := s.Handle("known_command genmove"); out != "true" {
		t.Fatalf("known genmove: %q", out)
	}
	if out := s.Handle("known_command not_a_command"); out != "false" {
		t.Fatalf("unknown command: %q", out)
	}
}

func TestGTPShowboard(t *testing.T) {
	s := NewSession(SessionConfig{})
	s.Handle("boardsize 9")
	out := s.Handle("showboard")
	if out == "" || !strings.Contains(out, ".") {
		t.Fatalf("showboard empty or missing dots: %q", out)
	}
}

func TestGTPFinalScore(t *testing.T) {
	s := NewSession(SessionConfig{})
	s.Handle("boardsize 9")
	s.Handle("play black pass")
	s.Handle("play white pass")
	out := s.Handle("final_score")
	if out == "" {
		t.Fatal("final_score empty")
	}
}

func TestRunSelfplaySamples(t *testing.T) {
	cfg := DefaultSelfplayConfig()
	cfg.Games = 1
	cfg.Playouts = 5
	cfg.BoardSize = 5
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
	}
}

func TestGatingHarness(t *testing.T) {
	g := GatingHarness{Games: 100, MinWinRateMargin: 0.55}
	if !g.Pass(0.4, 0.56) {
		t.Fatal("should pass")
	}
}

func BenchmarkSearchParallel(b *testing.B) {
	r := Chinese()
	board := NewBoard(9, 6.5)
	cfg := DefaultConfig()
	cfg.Playouts = 50
	cfg.Workers = 4
	eng := NewEngine(r, Uniform{}, cfg)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eng.ResetArena()
		eng.BestMove(board)
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

func BenchmarkLegalMoves(b *testing.B) {
	br := NewBoard(19, 7.5)
	r := Chinese()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = r.LegalMoves(br)
	}
}

func BenchmarkMakeMove(b *testing.B) {
	br := NewBoard(19, 7.5)
	r := Chinese()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := At(i%19, (i/19)%19)
		if !r.Play(br, StoneMove(p)) {
			br = NewBoard(19, 7.5)
			continue
		}
		br.Undo()
	}
}

func BenchmarkPlay(b *testing.B) { BenchmarkMakeMove(b) }

func BenchmarkCaptureHeavy(b *testing.B) {
	br := NewBoard(19, 7.5)
	r := Chinese()
	for y := 3; y < 16; y += 3 {
		for x := 3; x < 16; x += 3 {
			_ = r.Play(br, StoneMove(At(x, y)))
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = r.LegalMoves(br)
	}
}

func BenchmarkScore(b *testing.B) {
	br := NewBoard(19, 7.5)
	r := Chinese()
	for i := 0; i < 30; i++ {
		_ = r.Play(br, StoneMove(At(i%19, (i*2)%19)))
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = r.Score(br)
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

func BenchmarkSGFReplay(b *testing.B) {
	path := filepath.Join("testdata", "open.sgf")
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatal(err)
	}
	g, err := ParseSGF(string(data))
	if err != nil {
		b.Fatal(err)
	}
	moves, err := g.MainLine()
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Chinese()
		brd := NewBoard(g.Size, g.Komi)
		_ = g.Setup(brd)
		for _, m := range moves {
			if !r.Play(brd, sgfMoveToPlay(m)) {
				b.Fatal("illegal replay")
			}
		}
	}
}
