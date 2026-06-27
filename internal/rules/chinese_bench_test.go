package rules

import (
	"testing"

	"github.com/DevomB/gofer/internal/board"
)

func BenchmarkLegalMoves(b *testing.B) {
	br := board.New(19, 7.5)
	r := Chinese()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = r.LegalMoves(br)
	}
}

// BenchmarkMakeMove is the M1 MakeMove hot path (rules Play + Undo cycle).
func BenchmarkMakeMove(b *testing.B) {
	br := board.New(19, 7.5)
	r := Chinese()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := board.At(i%19, (i/19)%19)
		if !r.Play(br, board.StoneMove(p)) {
			br = board.New(19, 7.5)
			continue
		}
		br.Undo()
	}
}

func BenchmarkPlay(b *testing.B) { BenchmarkMakeMove(b) }

// BenchmarkCaptureHeavy legal move gen on a dense capture-fight position.
func BenchmarkCaptureHeavy(b *testing.B) {
	br := board.New(19, 7.5)
	r := Chinese()
	// ponytail: hand-built fight grid — not a pro game; stresses liberty scans.
	for y := 3; y < 16; y += 3 {
		for x := 3; x < 16; x += 3 {
			_ = r.Play(br, board.StoneMove(board.At(x, y)))
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = r.LegalMoves(br)
	}
}

func BenchmarkScore(b *testing.B) {
	br := board.New(19, 7.5)
	r := Chinese()
	for i := 0; i < 30; i++ {
		_ = r.Play(br, board.StoneMove(board.At(i%19, (i*2)%19)))
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = r.Score(br)
	}
}
