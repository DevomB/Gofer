package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Board holds Go grid state.
type Board struct {
	size   int
	komi   float64
	stones []Stone
	player Color
	ko     int // linear index prohibited by simple ko, or -1
	hash   uint64
	undo   []undoSnap
}

type undoSnap struct {
	move      Move
	player    Color
	ko        int
	hash      uint64
	captured  []int
	stoneIdx  int
	prevStone Stone
}

// NewBoard creates an empty board of the given size and komi.
func NewBoard(size int, komi float64) *Board {
	if size < 2 || size > 19 {
		panic("board size out of range")
	}
	b := &Board{
		size:   size,
		komi:   komi,
		stones: make([]Stone, size*size),
		player: Black,
		ko:     -1,
	}
	b.Rehash()
	return b
}

func (b *Board) Size() int     { return b.size }
func (b *Board) Komi() float64 { return b.komi }
func (b *Board) Player() Color { return b.player }

// SetPlayer sets side to move (SGF setup PL property).
func (b *Board) SetPlayer(c Color) { b.player = c }

func (b *Board) Ko() int      { return b.ko }
func (b *Board) Hash() uint64 { return b.hash }

func (b *Board) StoneAt(p Point) Stone {
	idx := p.Idx(b.size)
	if idx < 0 {
		return Empty
	}
	return b.stones[idx]
}

func (b *Board) AtIndex(idx int) Stone {
	if idx < 0 || idx >= len(b.stones) {
		return Empty
	}
	return b.stones[idx]
}

// StartPlay records undo before rules apply a move.
func (b *Board) StartPlay(m Move, captured []int, stoneIdx int, prev Stone) {
	b.undo = append(b.undo, undoSnap{
		move:      m,
		player:    b.player,
		ko:        b.ko,
		hash:      b.hash,
		captured:  append([]int(nil), captured...),
		stoneIdx:  stoneIdx,
		prevStone: prev,
	})
}

// SetStoneIndex places color at linear index (rules use after legality check).
func (b *Board) SetStoneIndex(idx int, c Stone) {
	b.setStone(idx, c)
}

func (b *Board) setStone(idx int, c Stone) {
	old := b.stones[idx]
	if old == c {
		return
	}
	if old != Empty {
		b.hash ^= zobristTable[idx][old]
	}
	b.stones[idx] = c
	if c != Empty {
		b.hash ^= zobristTable[idx][c]
	}
}

// FinishTurn switches player and sets ko point.
func (b *Board) FinishTurn(newKo int) {
	b.ko = newKo
	if b.player == Black {
		b.player = White
	} else {
		b.player = Black
	}
}

// MoveNum returns completed moves on the board.
func (b *Board) MoveNum() int { return len(b.undo) }

// historyMoves returns up to n most recent moves, oldest first (for feature history planes).
func (b *Board) historyMoves(n int) []undoSnap {
	if n <= 0 || len(b.undo) == 0 {
		return nil
	}
	start := len(b.undo) - n
	if start < 0 {
		start = 0
	}
	out := make([]undoSnap, len(b.undo)-start)
	copy(out, b.undo[start:])
	return out
}

func (b *Board) CanUndo() bool { return len(b.undo) > 0 }

func (b *Board) Undo() bool {
	if len(b.undo) == 0 {
		return false
	}
	s := b.undo[len(b.undo)-1]
	b.undo = b.undo[:len(b.undo)-1]
	b.player = s.player
	b.ko = s.ko
	b.hash = s.hash
	if s.move.Pass {
		return true
	}
	b.stones[s.stoneIdx] = s.prevStone
	for _, idx := range s.captured {
		b.stones[idx] = s.player.Opposite()
	}
	return true
}

// Clone returns a deep copy (benchmarks).
func (b *Board) Clone() *Board {
	c := *b
	c.stones = append([]Stone(nil), b.stones...)
	c.undo = append([]undoSnap(nil), b.undo...)
	for i := range c.undo {
		c.undo[i].captured = append([]int(nil), b.undo[i].captured...)
	}
	return &c
}

// Rehash recomputes Zobrist from stones.
func (b *Board) Rehash() {
	var h uint64
	for i, s := range b.stones {
		if s != Empty {
			h ^= zobristTable[i][s]
		}
	}
	b.hash = h
}

// forEachNeighbor visits in-bounds adjacent indices without allocating.
func (b *Board) forEachNeighbor(idx int, fn func(int)) {
	size := b.size
	x, y := idx%size, idx/size
	if x > 0 {
		fn(idx - 1)
	}
	if x+1 < size {
		fn(idx + 1)
	}
	if y > 0 {
		fn(idx - size)
	}
	if y+1 < size {
		fn(idx + size)
	}
}

// Neighbors returns in-bounds adjacent linear indices.
func (b *Board) Neighbors(idx int) []int {
	size := b.size
	x, y := idx%size, idx/size
	var out []int
	if x > 0 {
		out = append(out, idx-1)
	}
	if x+1 < size {
		out = append(out, idx+1)
	}
	if y > 0 {
		out = append(out, idx-size)
	}
	if y+1 < size {
		out = append(out, idx+size)
	}
	return out
}

// Snapshot captures board state.
type Snapshot struct {
	Stones []Stone
	Player Color
	Komi   float64
	Ko     int
	Hash   uint64
}

func (b *Board) Snapshot() Snapshot {
	return Snapshot{
		Stones: append([]Stone(nil), b.stones...),
		Player: b.player,
		Komi:   b.komi,
		Ko:     b.ko,
		Hash:   b.hash,
	}
}

func (b *Board) Restore(s Snapshot) {
	copy(b.stones, s.Stones)
	b.player = s.Player
	b.komi = s.Komi
	b.ko = s.Ko
	b.hash = s.Hash
}

// cloneTrial returns a board copy without undo history (legality trials only).
func (b *Board) cloneTrial() *Board {
	c := *b
	c.stones = append([]Stone(nil), b.stones...)
	c.undo = nil
	return &c
}

// trialSnap captures state for repeated trial restores without re-cloning.
type trialSnap struct {
	stones []Stone
	player Color
	ko     int
}

func (b *Board) captureTrialSnap() trialSnap {
	return trialSnap{
		stones: append([]Stone(nil), b.stones...),
		player: b.player,
		ko:     b.ko,
	}
}

func (b *Board) restoreTrialSnap(s trialSnap) {
	copy(b.stones, s.stones)
	b.player = s.player
	b.ko = s.ko
}

func formatGTPBoard(b *Board, size int) string {
	var sb strings.Builder
	sb.WriteString("  ")
	for x := 0; x < size; x++ {
		col := rune('A' + x)
		if col >= 'I' {
			col++
		}
		sb.WriteRune(col)
		sb.WriteByte(' ')
	}
	sb.WriteByte('\n')
	for y := 0; y < size; y++ {
		row := size - y
		if row < 10 {
			sb.WriteByte(' ')
		}
		sb.WriteString(strconv.Itoa(row))
		sb.WriteByte(' ')
		for x := 0; x < size; x++ {
			switch b.StoneAt(At(x, y)) {
			case Black:
				sb.WriteByte('X')
			case White:
				sb.WriteByte('O')
			default:
				sb.WriteByte('.')
			}
			sb.WriteByte(' ')
		}
		sb.WriteByte('\n')
	}
	return strings.TrimRight(sb.String(), "\n")
}

func formatGTPScore(b *Board, r Ruleset) string {
	bl, wl := r.Score(b)
	diff := bl - wl
	if diff > 0 {
		return fmt.Sprintf("B+%.1f", diff)
	}
	if diff < 0 {
		return fmt.Sprintf("W+%.1f", -diff)
	}
	return "0"
}
