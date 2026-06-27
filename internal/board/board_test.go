package board_test

import (
	"testing"

	"github.com/DevomB/gofer/internal/board"
)

func TestCoordRoundTrip(t *testing.T) {
	size := 9
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			p := board.At(x, y)
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
	b := board.New(9, 6.5)
	h0 := b.Hash()
	p := board.At(4, 4)
	b.StartPlay(board.StoneMove(p), nil, p.Idx(9), board.Empty)
	b.SetStoneIndex(p.Idx(9), board.Black)
	b.FinishTurn(-1)
	if b.Hash() == h0 {
		t.Fatal("hash should change")
	}
}

func TestUndo(t *testing.T) {
	b := board.New(9, 6.5)
	p := board.At(2, 2)
	idx := p.Idx(9)
	b.StartPlay(board.StoneMove(p), nil, idx, board.Empty)
	b.SetStoneIndex(idx, board.Black)
	b.FinishTurn(-1)
	if !b.CanUndo() {
		t.Fatal("expected undo")
	}
	b.Undo()
	if b.StoneAt(p) != board.Empty {
		t.Fatal("undo failed")
	}
}

func TestSnapshotRestore(t *testing.T) {
	b := board.New(9, 6.5)
	s := b.Snapshot()
	p := board.At(1, 1)
	idx := p.Idx(9)
	b.StartPlay(board.StoneMove(p), nil, idx, board.Empty)
	b.SetStoneIndex(idx, board.Black)
	b.FinishTurn(-1)
	b.Restore(s)
	if b.StoneAt(p) != board.Empty {
		t.Fatal("restore failed")
	}
}
