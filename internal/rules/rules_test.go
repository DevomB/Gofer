package rules

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/DevomB/gofer/internal/board"
	"github.com/DevomB/gofer/internal/sgf"
)

func play(t *testing.T, r Ruleset, b *board.Board, x, y int) {
	t.Helper()
	if !r.Play(b, board.StoneMove(board.At(x, y))) {
		t.Fatalf("illegal play at %d,%d", x, y)
	}
}

func TestNewBoard(t *testing.T) {
	b := NewBoard(9, 6.5)
	if b.Size() != 9 {
		t.Fatalf("size %d", b.Size())
	}
}

func TestTrompTaylorDeferred(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	_ = TrompTaylor().LegalMoves(NewBoard(9, 6.5))
}

func TestChineseFactory(t *testing.T) {
	r := Chinese()
	b := NewBoard(9, 6.5)
	if !r.Play(b, board.StoneMove(board.At(4, 4))) {
		t.Fatal("play failed")
	}
}

func TestCaptureRemovesStones(t *testing.T) {
	r := Chinese()
	b := board.New(9, 6.5)
	play(t, r, b, 1, 0)
	play(t, r, b, 0, 0)
	play(t, r, b, 0, 1)
	if b.StoneAt(board.At(0, 0)) != board.Empty {
		t.Fatal("captured stone should be gone")
	}
}

func TestSuicideIllegal(t *testing.T) {
	r := Chinese()
	b := board.New(9, 6.5)
	play(t, r, b, 0, 0)
	play(t, r, b, 3, 4)
	play(t, r, b, 0, 1)
	play(t, r, b, 5, 4)
	play(t, r, b, 0, 2)
	play(t, r, b, 4, 3)
	play(t, r, b, 0, 3)
	play(t, r, b, 4, 5)
	if r.Play(b, board.StoneMove(board.At(4, 4))) {
		t.Fatal("suicide should fail")
	}
}

func TestSimpleKo(t *testing.T) {
	r := Chinese()
	b := board.New(9, 6.5)
	// Connected ring around (4,4); final capture leaves one group liberty (ko point).
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
	play(t, r, b, 4, 3) // W has one liberty at (4,5)
	play(t, r, b, 1, 0)
	play(t, r, b, 4, 5) // capture; ko at (4,4)
	wantKo := board.At(4, 4).Idx(9)
	if b.Ko() != wantKo {
		t.Fatalf("ko want %d got %d", wantKo, b.Ko())
	}
	if r.Play(b, board.StoneMove(board.At(4, 4))) {
		t.Fatal("ko recapture illegal")
	}
}

func TestPassAndUndo(t *testing.T) {
	r := Chinese()
	b := board.New(9, 6.5)
	if !r.Play(b, board.PassMove) {
		t.Fatal("pass failed")
	}
	if b.Player() != board.White {
		t.Fatal("turn switch")
	}
	play(t, r, b, 2, 2)
	if !b.CanUndo() {
		t.Fatal("undo available")
	}
	b.Undo()
	if b.StoneAt(board.At(2, 2)) != board.Empty {
		t.Fatal("undo failed")
	}
}

func TestLegalMovesEmpty19(t *testing.T) {
	r := Chinese()
	b := board.New(19, 7.5)
	moves := r.LegalMoves(b)
	if len(moves) != 19*19+1 {
		t.Fatalf("want 362 got %d", len(moves))
	}
}

func TestScoreCountsStones(t *testing.T) {
	r := Chinese()
	b := board.New(9, 6.5)
	play(t, r, b, 0, 0)
	bl, _ := r.Score(b)
	if bl < 1 {
		t.Fatalf("expected black stones got %v", bl)
	}
}

func TestParseSGFCoord(t *testing.T) {
	p, err := sgf.ParseCoord(9, "bc")
	if err != nil {
		t.Fatal(err)
	}
	if p.X != 1 || p.Y != 2 {
		t.Fatalf("bc on 9x9 want (1,2) got (%d,%d)", p.X, p.Y)
	}
}

func replaySGF(t *testing.T, path string, check func(t *testing.T, r Ruleset, b *board.Board)) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := sgf.ParseMoves(string(data))
	if err != nil {
		t.Fatal(err)
	}
	r := Chinese()
	b := NewBoard(9, 6.5)
	for i, m := range parsed {
		if b.Player() != m.Color {
			t.Fatalf("move %d wrong side", i)
		}
		var play board.Move
		if m.Point == nil {
			play = board.PassMove
		} else {
			play = board.StoneMove(*m.Point)
		}
		if !r.Play(b, play) {
			t.Fatalf("illegal replay move %d in %s", i, filepath.Base(path))
		}
	}
	if check != nil {
		check(t, r, b)
	}
}

func TestGoldenCaptureSGF(t *testing.T) {
	path := filepath.Join("..", "testdata", "capture.sgf")
	replaySGF(t, path, func(t *testing.T, _ Ruleset, b *board.Board) {
		if b.StoneAt(board.At(0, 0)) != board.Empty {
			t.Fatal("white at aa should be captured")
		}
	})
}

func TestGoldenKoSGF(t *testing.T) {
	path := filepath.Join("..", "testdata", "ko.sgf")
	replaySGF(t, path, func(t *testing.T, r Ruleset, b *board.Board) {
		wantKo := board.At(4, 4).Idx(9)
		if b.Ko() != wantKo {
			t.Fatalf("ko want %d got %d", wantKo, b.Ko())
		}
		if r.Play(b, board.StoneMove(board.At(4, 4))) {
			t.Fatal("immediate ko recapture should be illegal")
		}
	})
}

func TestGoldenPassSGF(t *testing.T) {
	path := filepath.Join("..", "testdata", "pass.sgf")
	replaySGF(t, path, func(t *testing.T, _ Ruleset, b *board.Board) {
		if b.Player() != board.Black {
			t.Fatal("two passes should return turn to black")
		}
	})
}
