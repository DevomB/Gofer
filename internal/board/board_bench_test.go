package board

import "testing"

func BenchmarkApplyStone(b *testing.B) {
	br := New(19, 7.5)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx := i % (19 * 19)
		br.setStone(idx, Black)
		br.setStone(idx, Empty)
	}
}

func BenchmarkUndo(b *testing.B) {
	br := New(19, 7.5)
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
	br := New(19, 7.5)
	for i := 0; i < 20; i++ {
		br.setStone(i, Black)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = br.Clone()
	}
}

func BenchmarkHashUpdate(b *testing.B) {
	br := New(19, 7.5)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx := i % (19 * 19)
		br.setStone(idx, Black)
		_ = br.Hash()
		br.setStone(idx, Empty)
	}
}
