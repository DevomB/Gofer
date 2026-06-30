package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
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

func benchmarkSearchWithWorkers(b *testing.B, workers, playouts int) {
	r := Chinese()
	board := NewBoard(9, 6.5)
	cfg := DefaultConfig()
	cfg.Playouts = playouts
	cfg.Workers = workers
	eng := NewEngine(r, Uniform{}, cfg)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eng.ResetArena()
		eng.BestMove(board)
	}
}

func BenchmarkSearchWorkers1(b *testing.B) { benchmarkSearchWithWorkers(b, 1, 200) }
func BenchmarkSearchWorkers8(b *testing.B) { benchmarkSearchWithWorkers(b, 8, 200) }

func BenchmarkSearchParallel(b *testing.B) {
	benchmarkSearchWithWorkers(b, 4, 50)
}

func BenchmarkBestMove(b *testing.B) { BenchmarkSearchParallel(b) }

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

func BenchmarkEvalBatch(b *testing.B) {
	boards := make([]*Board, 32)
	for i := range boards {
		boards[i] = NewBoard(9, 6.5)
	}
	inf := Inference{Latency: 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = inf.EvalBatch(boards)
	}
}

func BenchmarkBatchedEvaluator(b *testing.B) {
	ev := NewBatchedEvaluator(Inference{}, Heuristic{}, 8, 2*time.Millisecond)
	defer ev.Close()
	board := NewBoard(9, 6.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ev.Evaluate(board)
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
