package gofer

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

// --- from internal/board\color.go ---
// Color is stone color or empty.
type Color uint8

const (
	Empty Color = iota
	Black
	White
)

// Stone is an alias for Color on the grid.
type Stone = Color

func (c Color) Opposite() Color {
	switch c {
	case Black:
		return White
	case White:
		return Black
	default:
		return Empty
	}
}

// --- from internal/board\point.go ---
// Point is a board intersection.
type Point struct {
	X, Y int
}

// Idx returns linear index or -1 if off 
func (p Point) Idx(size int) int {
	if p.X < 0 || p.Y < 0 || p.X >= size || p.Y >= size {
		return -1
	}
	return p.Y*size + p.X
}

// At builds a Point from coordinates.
func At(x, y int) Point {
	return Point{X: x, Y: y}
}

// PointFromIdx converts linear index to Point.
func PointFromIdx(size, idx int) Point {
	return Point{X: idx % size, Y: idx / size}
}

// --- from internal/board\move.go ---
// Move is a stone play or pass.
type Move struct {
	Point Point
	Pass  bool
}

// PassMove is the pass move sentinel.
var PassMove = Move{Pass: true}

// StoneMove returns a move at p.
func StoneMove(p Point) Move {
	return Move{Point: p}
}

// --- from internal/board\zobrist.go ---
const maxPoints = 19 * 19

var zobristTable [maxPoints][3]uint64

func init() {
	rng := rand.New(rand.NewSource(0x60eef0bacafe))
	for i := 0; i < maxPoints; i++ {
		for c := Empty; c <= White; c++ {
			zobristTable[i][c] = rng.Uint64()
		}
	}
}

// --- from internal/board\board.go ---
// Board holds Go grid state. Mutate only through rules packages.
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

// New creates an empty 
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

func (b *Board) Ko() int       { return b.ko }
func (b *Board) Hash() uint64  { return b.hash }

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

// --- from internal/rules\rules.go ---
// Ruleset applies rule-specific legality, scoring, and move application.
type Ruleset interface {
	LegalMoves(b *Board) []Move
	Play(b *Board, m Move) bool
	Score(b *Board) (black, white float64)
}

// Chinese returns the v1 Chinese rules implementation.
func Chinese() Ruleset {
	return &chineseRules{}
}

// TrompTaylor returns Tromp-Taylor rules with positional superko.
func TrompTaylor() Ruleset {
	return newTrompRules()
}

// --- from internal/rules\groups.go ---
func collectGroup(b *Board, start int, color Color) []int {
	if b.AtIndex(start) != color {
		return nil
	}
	out := []int{start}
	stack := []int{start}
	seen := map[int]struct{}{start: {}}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, nb := range b.Neighbors(i) {
			if b.AtIndex(nb) != color {
				continue
			}
			if _, ok := seen[nb]; ok {
				continue
			}
			seen[nb] = struct{}{}
			out = append(out, nb)
			stack = append(stack, nb)
		}
	}
	return out
}

func libertyCount(b *Board, start int, color Color) int {
	group := collectGroup(b, start, color)
	seen := make(map[int]struct{})
	libs := 0
	for _, g := range group {
		for _, nb := range b.Neighbors(g) {
			if b.AtIndex(nb) != Empty {
				continue
			}
			if _, ok := seen[nb]; ok {
				continue
			}
			seen[nb] = struct{}{}
			libs++
		}
	}
	return libs
}

func removeDeadGroups(b *Board, color Color) []int {
	n := b.Size() * b.Size()
	var captured []int
	for i := 0; i < n; i++ {
		if b.AtIndex(i) != color {
			continue
		}
		if libertyCount(b, i, color) == 0 {
			for _, g := range collectGroup(b, i, color) {
				b.SetStoneIndex(g, Empty)
				captured = append(captured, g)
			}
		}
	}
	return captured
}

func floodEmpty(b *Board, start int, seen []bool) (territory int, touchBlack, touchWhite bool) {
	stack := []int{start}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[i] {
			continue
		}
		seen[i] = true
		switch b.AtIndex(i) {
		case Empty:
			territory++
			for _, nb := range b.Neighbors(i) {
				if !seen[nb] {
					stack = append(stack, nb)
				}
			}
		case Black:
			touchBlack = true
		case White:
			touchWhite = true
		}
	}
	return territory, touchBlack, touchWhite
}

// --- from internal/rules\superko.go ---
// WithSuperko wraps r with positional superko using board Zobrist hash + side to move.
func WithSuperko(r Ruleset) Ruleset {
	return &superkoRules{inner: r, seen: map[uint64]struct{}{}}
}

type superkoRules struct {
	inner Ruleset
	seen  map[uint64]struct{}
}

func (s *superkoRules) LegalMoves(b *Board) []Move {
	s.ensureStart(b)
	all := s.inner.LegalMoves(b)
	out := make([]Move, 0, len(all))
	for _, m := range all {
		if m.Pass {
			out = append(out, m)
			continue
		}
		if s.trialLegal(b, m) {
			out = append(out, m)
		}
	}
	return out
}

func (s *superkoRules) Play(b *Board, m Move) bool {
	s.ensureStart(b)
	if m.Pass {
		if s.repeats(superkoHash(b)) {
			return false
		}
		if !s.inner.Play(b, m) {
			return false
		}
		s.record(superkoHash(b))
		return true
	}
	if !s.trialLegal(b, m) {
		return false
	}
	if !s.inner.Play(b, m) {
		return false
	}
	s.record(superkoHash(b))
	return true
}

func (s *superkoRules) Score(b *Board) (black, white float64) {
	return s.inner.Score(b)
}

func (s *superkoRules) trialLegal(b *Board, m Move) bool {
	trial := b.Clone()
	if !s.inner.Play(trial, m) {
		return false
	}
	return !s.repeats(superkoHash(trial))
}

func (s *superkoRules) repeats(h uint64) bool {
	_, ok := s.seen[h]
	return ok
}

func (s *superkoRules) record(h uint64) {
	s.seen[h] = struct{}{}
}

func superkoHash(b *Board) uint64 {
	h := b.Hash()
	if b.Player() == White {
		h ^= 0x9e3779b97f4a7c15
	}
	return h
}

func (s *superkoRules) ensureStart(b *Board) {
	if len(s.seen) == 0 {
		s.record(superkoHash(b))
	}
}

// --- from internal/rules\chinese_rules.go ---
// chineseRules implements Chinese rules: area scoring, suicide if capturing, simple ko.
type chineseRules struct{}

// LegalMoves returns all legal moves including pass.
func (r *chineseRules) LegalMoves(b *Board) []Move {
	size := b.Size()
	n := size * size
	ko := b.Ko()
	snap := b.Snapshot()
	trial := b.Clone()
	moves := make([]Move, 0, n+1)
	for idx := 0; idx < n; idx++ {
		if idx == ko || b.AtIndex(idx) != Empty {
			continue
		}
		trial.Restore(snap)
		if r.wouldBeLegalTrial(trial, idx) {
			moves = append(moves, StoneMove(PointFromIdx(size, idx)))
		}
	}
	moves = append(moves, PassMove)
	return moves
}

// Play applies m for the current player. Returns false if illegal.
func (r *chineseRules) Play(b *Board, m Move) bool {
	player := b.Player()
	if m.Pass {
		b.StartPlay(m, nil, -1, Empty)
		b.FinishTurn(-1)
		return true
	}
	idx := m.Point.Idx(b.Size())
	if idx < 0 || b.AtIndex(idx) != Empty || idx == b.Ko() {
		return false
	}
	if !r.wouldBeLegal(b, idx) {
		return false
	}
	prev := b.AtIndex(idx)
	trial := b.Clone()
	trial.SetStoneIndex(idx, player)
	captured := removeDeadGroups(trial, player.Opposite())
	if libertyCount(trial, idx, player) == 0 {
		return false
	}
	b.StartPlay(m, captured, idx, prev)
	b.SetStoneIndex(idx, player)
	for _, cidx := range captured {
		b.SetStoneIndex(cidx, Empty)
	}
	newKo := -1
	if len(captured) == 1 {
		newKo = captured[0]
	}
	b.FinishTurn(newKo)
	return true
}

// Score returns area scores; komi added to white.
// ponytail: seki neutral; no dead-stone removal pass.
// Ceiling: tournament Chinese may differ.
// Upgrade: two-pass scoring (M2).
func (r *chineseRules) Score(b *Board) (black, white float64) {
	size := b.Size()
	n := size * size
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case Black:
			black++
		case White:
			white++
		}
	}
	seen := make([]bool, n)
	for i := 0; i < n; i++ {
		if seen[i] || b.AtIndex(i) != Empty {
			continue
		}
		t, tb, tw := floodEmpty(b, i, seen)
		if tb && !tw {
			black += float64(t)
		} else if tw && !tb {
			white += float64(t)
		}
	}
	white += b.Komi()
	return black, white
}

func (r *chineseRules) wouldBeLegal(b *Board, idx int) bool {
	if idx == b.Ko() {
		return false
	}
	trial := b.Clone()
	return r.wouldBeLegalTrial(trial, idx)
}

func (r *chineseRules) wouldBeLegalTrial(trial *Board, idx int) bool {
	player := trial.Player()
	trial.SetStoneIndex(idx, player)
	removeDeadGroups(trial, player.Opposite())
	return libertyCount(trial, idx, player) > 0
}

// --- from internal/rules\tromp_rules.go ---
type trompRules struct {
	seen map[uint64]struct{}
}

func newTrompRules() *trompRules {
	return &trompRules{seen: map[uint64]struct{}{}}
}

func (r *trompRules) LegalMoves(b *Board) []Move {
	r.ensureStart(b)
	size := b.Size()
	n := size * size
	moves := make([]Move, 0, n+1)
	for idx := 0; idx < n; idx++ {
		if b.AtIndex(idx) != Empty {
			continue
		}
		if r.wouldBeLegal(b, idx) {
			moves = append(moves, StoneMove(PointFromIdx(size, idx)))
		}
	}
	moves = append(moves, PassMove)
	return moves
}

func (r *trompRules) Play(b *Board, m Move) bool {
	r.ensureStart(b)
	if m.Pass {
		b.StartPlay(m, nil, -1, Empty)
		b.FinishTurn(-1)
		pos := trompPositionHash(b)
		if r.repeats(pos) {
			b.Undo()
			return false
		}
		r.record(pos)
		return true
	}
	idx := m.Point.Idx(b.Size())
	if idx < 0 || b.AtIndex(idx) != Empty {
		return false
	}
	player := b.Player()
	prev := b.AtIndex(idx)
	trial := b.Clone()
	trial.SetStoneIndex(idx, player)
	captured := removeDeadGroups(trial, player.Opposite())
	captured = append(captured, removeDeadGroups(trial, player)...)
	trial.FinishTurn(-1)
	pos := trompPositionHash(trial)
	if r.repeats(pos) {
		return false
	}
	b.StartPlay(m, captured, idx, prev)
	b.SetStoneIndex(idx, player)
	for _, cidx := range captured {
		b.SetStoneIndex(cidx, Empty)
	}
	b.FinishTurn(-1)
	r.record(pos)
	return true
}

// Score returns Tromp-Taylor area scores (komi to white).
// ponytail: all on-board stones alive; territory via empty-region flood.
// Ceiling: no Benson pass-alive removal.
// Upgrade: Benson pass-alive (M3 backlog-core-engine).
func (r *trompRules) Score(b *Board) (black, white float64) {
	size := b.Size()
	n := size * size
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case Black:
			black++
		case White:
			white++
		}
	}
	seen := make([]bool, n)
	for i := 0; i < n; i++ {
		if seen[i] || b.AtIndex(i) != Empty {
			continue
		}
		t, tb, tw := floodEmpty(b, i, seen)
		if tb && !tw {
			black += float64(t)
		} else if tw && !tb {
			white += float64(t)
		}
	}
	white += b.Komi()
	return black, white
}

func (r *trompRules) ensureStart(b *Board) {
	if len(r.seen) == 0 {
		r.record(trompPositionHash(b))
	}
}

func (r *trompRules) wouldBeLegal(b *Board, idx int) bool {
	player := b.Player()
	trial := b.Clone()
	trial.SetStoneIndex(idx, player)
	removeDeadGroups(trial, player.Opposite())
	removeDeadGroups(trial, player)
	trial.FinishTurn(-1)
	return !r.repeats(trompPositionHash(trial))
}

func (r *trompRules) repeats(h uint64) bool {
	_, ok := r.seen[h]
	return ok
}

func (r *trompRules) record(h uint64) {
	r.seen[h] = struct{}{}
}

func trompPositionHash(b *Board) uint64 {
	h := b.Hash()
	if b.Player() == White {
		h ^= 0x9e3779b97f4a7c15
	}
	return h
}

// --- from internal/rules\sgf.go ---
// SGFMove is a color and optional board point from SGF (pass when Point is nil).
type SGFMove struct {
	Color Color
	Point *Point
}

// SGFNode is one SGF node with properties and child variations.
type SGFNode struct {
	Props    map[string][]string
	Children []*SGFNode
}

// SGFGame is a parsed SGF game tree.
type SGFGame struct {
	Size int
	Komi float64
	Root *SGFNode
}

// ParseSGFCoord decodes SGF coords (aa = upper-left).
func ParseSGFCoord(size int, s string) (Point, error) {
	if len(s) != 2 {
		return Point{}, fmt.Errorf("sgf coord %q: want 2 letters", s)
	}
	x := int(s[0] - 'a')
	y := int(s[1] - 'a')
	if x < 0 || y < 0 || x >= size || y >= size {
		return Point{}, fmt.Errorf("sgf coord %q off %dx%d board", s, size, size)
	}
	return At(x, y), nil
}

// ParseSGF parses an SGF file (FF[4] subset: SZ, KM, AB, AW, PL, B, W).
func ParseSGF(data string) (*SGFGame, error) {
	p := newSGFParser(data)
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	if p.peek() != '(' {
		return nil, fmt.Errorf("sgf: expected '(' at start")
	}
	root, err := p.parseSequence()
	if err != nil {
		return nil, err
	}
	g := &SGFGame{Size: 19, Komi: 7.5, Root: root}
	if root != nil {
		g.applyMeta(root)
	}
	return g, nil
}

func (g *SGFGame) applyMeta(n *SGFNode) {
	if sz, ok := n.Props["SZ"]; ok && len(sz) > 0 {
		if v, err := strconv.Atoi(sz[0]); err == nil && v >= 2 {
			g.Size = v
		}
	}
	if km, ok := n.Props["KM"]; ok && len(km) > 0 {
		if v, err := strconv.ParseFloat(km[0], 64); err == nil {
			g.Komi = v
		}
	}
}

// MainLine returns moves along the first child at each variation point.
func (g *SGFGame) MainLine() ([]SGFMove, error) {
	var moves []SGFMove
	if err := g.collectMainLine(g.Root, &moves); err != nil {
		return nil, err
	}
	return moves, nil
}

func (g *SGFGame) collectMainLine(n *SGFNode, moves *[]SGFMove) error {
	if n == nil {
		return nil
	}
	if err := appendSGFMoveProps(g.Size, n.Props, moves); err != nil {
		return err
	}
	if len(n.Children) > 0 {
		return g.collectMainLine(n.Children[0], moves)
	}
	return nil
}

func appendSGFMoveProps(size int, props map[string][]string, moves *[]SGFMove) error {
	for tag, vals := range props {
		if tag != "B" && tag != "W" {
			continue
		}
		c := Black
		if tag == "W" {
			c = White
		}
		for _, v := range vals {
			if v == "" {
				*moves = append(*moves, SGFMove{Color: c})
				continue
			}
			pt, err := ParseSGFCoord(size, v)
			if err != nil {
				return err
			}
			*moves = append(*moves, SGFMove{Color: c, Point: &pt})
		}
	}
	return nil
}

// SetupSGF applies AB/AW/PL from the root node onto b.
func (g *SGFGame) Setup(b *Board) error {
	if g.Root == nil {
		return nil
	}
	n := g.Root
	for _, coord := range n.Props["AB"] {
		pt, err := ParseSGFCoord(g.Size, coord)
		if err != nil {
			return err
		}
		b.SetStoneIndex(pt.Idx(g.Size), Black)
	}
	for _, coord := range n.Props["AW"] {
		pt, err := ParseSGFCoord(g.Size, coord)
		if err != nil {
			return err
		}
		b.SetStoneIndex(pt.Idx(g.Size), White)
	}
	if pl, ok := n.Props["PL"]; ok && len(pl) > 0 && pl[0] == "2" {
		b.SetPlayer(White)
	}
	b.Rehash()
	return nil
}

// ExportSGF writes a minimal SGF for the main line.
func ExportSGF(g *SGFGame, moves []SGFMove) string {
	var b strings.Builder
	b.WriteString("(;FF[4]")
	fmt.Fprintf(&b, "SZ[%d]", g.Size)
	fmt.Fprintf(&b, "KM[%.1f]", g.Komi)
	for _, m := range moves {
		tag := 'B'
		if m.Color == White {
			tag = 'W'
		}
		b.WriteByte(';')
		b.WriteByte(byte(tag))
		b.WriteByte('[')
		if m.Point != nil {
			p := *m.Point
			b.WriteByte('a' + byte(p.X))
			b.WriteByte('a' + byte(p.Y))
		}
		b.WriteByte(']')
	}
	b.WriteString(")")
	return b.String()
}

// --- from internal/rules\sgf_parse.go ---
type sgfParser struct {
	s string
	i int
}

func newSGFParser(data string) *sgfParser {
	return &sgfParser{s: data}
}

func (p *sgfParser) eof() bool { return p.i >= len(p.s) }

func (p *sgfParser) peek() byte {
	if p.eof() {
		return 0
	}
	return p.s[p.i]
}

func (p *sgfParser) skipWS() error {
	for !p.eof() {
		c := p.s[p.i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			p.i++
			continue
		}
		break
	}
	return nil
}

func (p *sgfParser) parseSequence() (*SGFNode, error) {
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	if p.peek() != '(' {
		return nil, fmt.Errorf("sgf: expected '('")
	}
	p.i++
	return p.parseSequenceBody()
}

func (p *sgfParser) parseSequenceBody() (*SGFNode, error) {
	var first, cur *SGFNode
	for {
		if err := p.skipWS(); err != nil {
			return nil, err
		}
		if p.eof() {
			return nil, fmt.Errorf("sgf: unclosed '('")
		}
		switch p.peek() {
		case ';':
			node, err := p.parseNode()
			if err != nil {
				return nil, err
			}
			first, cur = sgfLinkNode(first, cur, node)
		case '(':
			child, err := p.parseSequence()
			if err != nil {
				return nil, err
			}
			first, cur = sgfLinkChild(first, cur, child)
		case ')':
			p.i++
			return first, nil
		default:
			return nil, fmt.Errorf("sgf: unexpected %q at %d", p.peek(), p.i)
		}
	}
}

func sgfLinkNode(first, cur, node *SGFNode) (*SGFNode, *SGFNode) {
	if first == nil {
		return node, node
	}
	if cur != nil {
		cur.Children = append(cur.Children, node)
	}
	return first, node
}

func sgfLinkChild(first, cur, child *SGFNode) (*SGFNode, *SGFNode) {
	if cur != nil {
		cur.Children = append(cur.Children, child)
		return first, cur
	}
	if first == nil {
		return child, nil
	}
	return first, cur
}

func (p *sgfParser) parseNode() (*SGFNode, error) {
	if p.peek() != ';' {
		return nil, fmt.Errorf("sgf: expected ';'")
	}
	p.i++
	n := &SGFNode{Props: map[string][]string{}}
	for p.hasMoreProps() {
		key, err := p.readPropKey()
		if err != nil {
			return nil, err
		}
		vals, err := p.readPropValues()
		if err != nil {
			return nil, err
		}
		if len(vals) > 0 {
			n.Props[key] = append(n.Props[key], vals...)
		}
	}
	return n, nil
}

func (p *sgfParser) hasMoreProps() bool {
	if err := p.skipWS(); err != nil {
		return false
	}
	if p.eof() {
		return false
	}
	c := p.peek()
	return c != '(' && c != ')' && c != ';' && c >= 'A' && c <= 'Z'
}

func (p *sgfParser) readPropKey() (string, error) {
	if err := p.skipWS(); err != nil {
		return "", err
	}
	start := p.i
	for !p.eof() && p.s[p.i] >= 'A' && p.s[p.i] <= 'Z' {
		p.i++
	}
	if start == p.i {
		return "", fmt.Errorf("sgf: expected property key")
	}
	return p.s[start:p.i], nil
}

func (p *sgfParser) readPropValues() ([]string, error) {
	var vals []string
	for {
		if err := p.skipWS(); err != nil {
			return vals, err
		}
		if p.peek() != '[' {
			break
		}
		val, err := p.parseValue()
		if err != nil {
			return nil, err
		}
		vals = append(vals, val)
	}
	return vals, nil
}

func (p *sgfParser) parseValue() (string, error) {
	if p.peek() != '[' {
		return "", fmt.Errorf("sgf: expected '['")
	}
	p.i++
	var b strings.Builder
	for !p.eof() {
		c := p.s[p.i]
		if c == ']' {
			p.i++
			return b.String(), nil
		}
		if c == '\\' && p.i+1 < len(p.s) {
			p.i++
			b.WriteByte(p.s[p.i])
			p.i++
			continue
		}
		b.WriteByte(c)
		p.i++
	}
	return "", fmt.Errorf("sgf: unclosed '['")
}

// --- from internal/engine\tt.go ---
// Entry is a transposition table record.
type Entry struct {
	Depth int
	Value float64
}

// Table is a Zobrist-keyed transposition table (M6).
type Table struct {
	slots []Entry
	mask  uint64
}

// NewTable creates a TT with the given slot count (power of two).
func NewTable(size int) *Table {
	if size < 256 {
		size = 256
	}
	return &Table{
		slots: make([]Entry, size),
		mask:  uint64(size - 1),
	}
}

// Get looks up hash.
func (t *Table) Get(hash uint64) (Entry, bool) {
	e := t.slots[hash&t.mask]
	if e.Depth == 0 {
		return Entry{}, false
	}
	return e, true
}

// Store saves an entry (replace always — ponytail).
// Ceiling: no depth-preferred replacement.
// Upgrade: two-tier TT (M6+).
func (t *Table) Store(hash uint64, e Entry) {
	t.slots[hash&t.mask] = e
}

// --- from internal/engine\arena.go ---
// Node is an index-based MCTS tree node.
type Node struct {
	Parent   int
	Move     Move
	Children []int
	Visits   uint32
	ValueSum float64
	Prior    float64
	Expanded bool
}

// Arena stores nodes in a contiguous slice (ponytail: doubling growth).
type Arena struct {
	nodes []Node
}

// NewArena creates an empty arena.
func NewArena() *Arena {
	return &Arena{nodes: make([]Node, 0, 64)}
}

// Root allocates the root node and returns its index.
func (a *Arena) Root() int {
	if len(a.nodes) == 0 {
		a.nodes = append(a.nodes, Node{Parent: -1})
	}
	return 0
}

// Get returns node at index.
func (a *Arena) Get(i int) *Node {
	return &a.nodes[i]
}

// AddChild appends a child node and returns its index.
func (a *Arena) AddChild(parent int, m Move, prior float64) int {
	idx := len(a.nodes)
	a.nodes = append(a.nodes, Node{
		Parent: parent,
		Move:   m,
		Prior:  prior,
	})
	a.nodes[parent].Children = append(a.nodes[parent].Children, idx)
	return idx
}

// Len returns node count.
func (a *Arena) Len() int { return len(a.nodes) }

// Mean returns average value for node visits.
func (n *Node) Mean() float64 {
	if n.Visits == 0 {
		return 0
	}
	return n.ValueSum / float64(n.Visits)
}

// --- from internal/engine\evaluator.go ---
// Result holds leaf evaluation from an Evaluator.
type Result struct {
	Value  float64   // from current player perspective, in [-1,1]
	Policy []float32 // optional move priors indexed by point (len size*size+1 for pass)
}

// Evaluator scores positions and optional policy priors (M7 boundary).
type Evaluator interface {
	Evaluate(b *Board) Result
}

// Uniform returns equal priors and zero value (M4-M5 placeholder).
type Uniform struct{}

func (Uniform) Evaluate(b *Board) Result {
	n := b.Size()*b.Size() + 1
	p := make([]float32, n)
	for i := range p {
		p[i] = 1
	}
	return Result{Value: 0, Policy: p}
}

// Heuristic uses stone count diff as value (M7 ponytail).
type Heuristic struct{}

func (Heuristic) Evaluate(b *Board) Result {
	bl, wl := 0, 0
	n := b.Size() * b.Size()
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case Black:
			bl++
		case White:
			wl++
		}
	}
	v := float64(bl-wl) / float64(n)
	if b.Player() == White {
		v = -v
	}
	return Result{Value: v, Policy: nil}
}

// Mock returns fixed value/policy for tests.
type Mock struct {
	Value  float64
	Policy []float32
}

func (m Mock) Evaluate(b *Board) Result {
	return Result{Value: m.Value, Policy: m.Policy}
}

// --- from internal/engine\inference.go ---
// Inference is an external NN adapter (M11 — mock/sidecar hook).
type Inference struct {
	// ponytail: no ONNX runtime in-
	// Ceiling: mock weights only.
	// Upgrade: sidecar ONNX/Torch via HTTP or CGO.
	MockValue  float64
	MockPolicy []float32
}

// Evaluate implements Evaluator via mock/sidecar hook.
func (inf Inference) Evaluate(b *Board) Result {
	n := b.Size()*b.Size() + 1
	p := inf.MockPolicy
	if len(p) != n {
		p = make([]float32, n)
		for i := range p {
			p[i] = 1
		}
	}
	return Result{Value: inf.MockValue, Policy: p}
}

// GatingHarness compares win rates for model promotion (M11).
type GatingHarness struct {
	Games            int
	MinWinRateMargin float64 // e.g. 0.55 vs 0.45
}

// Pass returns true if candidate beats baseline by margin over Games.
func (g GatingHarness) Pass(baselineWin, candidateWin float64) bool {
	if g.Games <= 0 {
		return true
	}
	margin := g.MinWinRateMargin
	if margin == 0 {
		margin = 0.55
	}
	return candidateWin >= margin && candidateWin > baselineWin
}

// --- from internal/engine\mcts.go ---
const (
	defaultCPUCT      = 1.1
	defaultFPU        = 0.2
	dirichletAlpha    = 0.03
	dirichletBlend    = 0.25
	rootTemperature   = 1.03
	maxRolloutMoves   = 150
)

// SearchConfig holds MCTS parameters (paper defaults).
type SearchConfig struct {
	CPUCT           float64
	FPU             float64
	Playouts        int
	Seed            int64
	RootNoise       bool
	RootTemperature float64
}

// DefaultConfig returns paper-aligned defaults.
func DefaultConfig() SearchConfig {
	return SearchConfig{
		CPUCT:           defaultCPUCT,
		FPU:             defaultFPU,
		Playouts:        100,
		Seed:            1,
		RootNoise:       false,
		RootTemperature: rootTemperature,
	}
}

// Engine runs MCTS search.
type Engine struct {
	Rules Ruleset
	Eval  Evaluator
	TT    *Table
	cfg   SearchConfig
	rng   *rand.Rand
	arena *Arena
}

// New creates a search 
func NewEngine(r Ruleset, ev Evaluator, cfg SearchConfig) *Engine {
	if cfg.CPUCT == 0 {
		cfg = DefaultConfig()
	}
	if ev == nil {
		ev = Uniform{}
	}
	return &Engine{
		Rules: r,
		Eval:  ev,
		TT:    NewTable(1 << 16),
		cfg:   cfg,
		rng:   rand.New(rand.NewSource(cfg.Seed)),
	}
}

// ResetArena clears the search tree (GTP tree reuse on same board).
func (e *Engine) ResetArena() {
	e.arena = nil
}

// BestMove runs MCTS and returns the most visited root child move.
func (e *Engine) BestMove(b *Board) Move {
	e.arena = NewArena()
	root := e.arena.Root()
	for i := 0; i < e.cfg.Playouts; i++ {
		e.runPlayout(b, root)
	}
	return e.selectBestRoot(root)
}

// RootPolicy returns visit-weighted policy over legal moves (training π).
func (e *Engine) RootPolicy(legal []Move) []float32 {
	pi := make([]float32, len(legal))
	if e.arena == nil || len(e.arena.nodes) == 0 {
		return uniformPolicy32(len(legal))
	}
	root := e.arena.Get(0)
	var total uint32
	for _, cidx := range root.Children {
		total += e.arena.Get(cidx).Visits
	}
	if total == 0 {
		return uniformPolicy32(len(legal))
	}
	for i, m := range legal {
		for _, cidx := range root.Children {
			c := e.arena.Get(cidx)
			if movesEqual(c.Move, m) {
				pi[i] = float32(c.Visits) / float32(total)
				break
			}
		}
	}
	return pi
}

func (e *Engine) runPlayout(b *Board, root int) {
	br := b.Clone()
	path := []int{root}
	node := root

	for {
		n := e.arena.Get(node)
		if !n.Expanded {
			break
		}
		if len(n.Children) == 0 {
			break
		}
		child := e.selectChild(node, node == root)
		path = append(path, child)
		cn := e.arena.Get(child)
		e.applyMove(br, cn.Move)
		node = child
	}

	n := e.arena.Get(node)
	if !n.Expanded {
		if v, ok := e.ttLeafValue(b); ok {
			e.backup(path, v)
			return
		}
		e.expand(node, br)
	}

	value := e.leafValue(br)
	e.backup(path, value)
}

func (e *Engine) backup(path []int, value float64) {
	for i := len(path) - 1; i >= 0; i-- {
		nd := e.arena.Get(path[i])
		nd.Visits++
		nd.ValueSum += value
		value = -value
	}
}

func (e *Engine) applyMove(br *Board, m Move) {
	if m.Pass {
		e.Rules.Play(br, PassMove)
	} else {
		e.Rules.Play(br, StoneMove(m.Point))
	}
}

func (e *Engine) expand(node int, b *Board) {
	n := e.arena.Get(node)
	if n.Expanded {
		return
	}
	if e.isTerminal(b) {
		n.Expanded = true
		return
	}
	moves := e.Rules.LegalMoves(b)
	res := e.Eval.Evaluate(b)
	priors := uniformPriors(len(moves))
	if len(res.Policy) > 0 {
		priors = policyPriors(b, moves, res.Policy)
	}
	if node == 0 && e.cfg.RootNoise {
		priors = blendDirichlet(priors, e.rng)
	}
	for i, m := range moves {
		e.arena.AddChild(node, m, priors[i])
	}
	n.Expanded = true
	e.TT.Store(b.Hash(), Entry{Depth: 1, Value: res.Value})
}

func (e *Engine) selectChild(node int, isRoot bool) int {
	n := e.arena.Get(node)
	parentVisits := float64(n.Visits)
	if parentVisits == 0 {
		parentVisits = 1
	}
	best := -1
	bestScore := math.Inf(-1)
	for _, cidx := range n.Children {
		c := e.arena.Get(cidx)
		score := e.puctScore(c, parentVisits, isRoot)
		if score > bestScore {
			bestScore = score
			best = cidx
		}
	}
	return best
}

func (e *Engine) puctScore(c *Node, parentVisits float64, isRoot bool) float64 {
	q := c.Mean()
	if c.Visits == 0 {
		q = -e.cfg.FPU
	}
	u := e.cfg.CPUCT * c.Prior * math.Sqrt(parentVisits) / (1 + float64(c.Visits))
	if isRoot && e.cfg.RootTemperature != 1 && c.Visits > 0 {
		q /= e.cfg.RootTemperature
	}
	return q + u
}

func (e *Engine) ttLeafValue(b *Board) (float64, bool) {
	entry, ok := e.TT.Get(b.Hash())
	if !ok || entry.Depth == 0 {
		return 0, false
	}
	return entry.Value, true
}

func (e *Engine) leafValue(b *Board) float64 {
	if v, ok := e.ttLeafValue(b); ok {
		return v
	}
	res := e.Eval.Evaluate(b)
	if res.Value != 0 {
		e.TT.Store(b.Hash(), Entry{Depth: 1, Value: res.Value})
		return res.Value
	}
	v := e.randomPlayout(b)
	e.TT.Store(b.Hash(), Entry{Depth: 1, Value: v})
	return v
}

func (e *Engine) randomPlayout(b *Board) float64 {
	br := b.Clone()
	player := br.Player()
	passes := 0
	for move := 0; move < maxRolloutMoves && passes < 2; move++ {
		moves := e.Rules.LegalMoves(br)
		if len(moves) == 0 {
			break
		}
		m := moves[e.rng.Intn(len(moves))]
		e.Rules.Play(br, m)
		if m.Pass {
			passes++
		} else {
			passes = 0
		}
	}
	bl, wl := e.Rules.Score(br)
	diff := bl - wl
	if player == White {
		diff = wl - bl
	}
	if diff > 0 {
		return 1
	}
	if diff < 0 {
		return -1
	}
	return 0
}

func (e *Engine) isTerminal(b *Board) bool {
	for _, m := range e.Rules.LegalMoves(b) {
		if !m.Pass {
			return false
		}
	}
	return true
}

func (e *Engine) selectBestRoot(root int) Move {
	n := e.arena.Get(root)
	if len(n.Children) == 0 {
		return PassMove
	}
	best := n.Children[0]
	maxV := uint32(0)
	for _, cidx := range n.Children {
		c := e.arena.Get(cidx)
		if c.Visits > maxV {
			maxV = c.Visits
			best = cidx
		}
	}
	return e.arena.Get(best).Move
}

func uniformPriors(n int) []float64 {
	if n == 0 {
		return nil
	}
	p := 1.0 / float64(n)
	out := make([]float64, n)
	for i := range out {
		out[i] = p
	}
	return out
}

func uniformPolicy32(n int) []float32 {
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	for i := range out {
		out[i] = 1 / float32(n)
	}
	return out
}

func policyPriors(b *Board, moves []Move, policy []float32) []float64 {
	size := b.Size()
	sum := float64(0)
	raw := make([]float64, len(moves))
	for i, m := range moves {
		idx := size * size
		if !m.Pass {
			idx = m.Point.Idx(size)
		}
		if idx >= 0 && idx < len(policy) {
			raw[i] = float64(policy[idx])
		} else {
			raw[i] = 1
		}
		sum += raw[i]
	}
	if sum == 0 {
		return uniformPriors(len(moves))
	}
	for i := range raw {
		raw[i] /= sum
	}
	return raw
}

func blendDirichlet(priors []float64, rng *rand.Rand) []float64 {
	out := make([]float64, len(priors))
	sum := 0.0
	noise := make([]float64, len(priors))
	for i := range noise {
		noise[i] = math.Pow(rng.Float64(), 1/dirichletAlpha)
		sum += noise[i]
	}
	for i := range out {
		n := noise[i] / sum
		out[i] = (1-dirichletBlend)*priors[i] + dirichletBlend*n
	}
	return out
}

func movesEqual(a, b Move) bool {
	if a.Pass != b.Pass {
		return false
	}
	if a.Pass {
		return true
	}
	return a.Point == b.Point
}

// PUCTScore exposes the formula for unit tests.
func PUCTScore(q, prior, parentVisits float64, visits uint32, cPUCT float64) float64 {
	u := cPUCT * prior * math.Sqrt(parentVisits) / (1 + float64(visits))
	return q + u
}

// TTHitRate returns fraction of lookups that hit (for M6 benchmarks).
func (e *Engine) TTHitRate(b *Board, probes int) float64 {
	if probes <= 0 {
		return 0
	}
	hits := 0
	for i := 0; i < probes; i++ {
		if _, ok := e.TT.Get(b.Hash()); ok {
			hits++
		}
	}
	return float64(hits) / float64(probes)
}

// --- from internal/engine\gtp.go ---
// Session holds GTP engine state.
type Session struct {
	Board  *Board
	Rules  Ruleset
	Search *Engine
	Size   int
	Komi   float64
}

// NewSession creates a default 19x19 session.
func NewSession() *Session {
	size, komi := 19, 7.5
	b := NewBoard(size, komi)
	r := Chinese()
	cfg := DefaultConfig()
	cfg.Playouts = 8
	cfg.Seed = 1
	return &Session{
		Board:  b,
		Rules:  r,
		Search: NewEngine(r, nil, cfg),
		Size:   size,
		Komi:   komi,
	}
}

// Handle processes one GTP command line and returns the response body (without =/? prefix).
func (s *Session) Handle(line string) string {
	line = strings.TrimSpace(line)
	if line == "" {
		return ""
	}
	parts := strings.Fields(line)
	cmd := strings.ToLower(parts[0])
	switch cmd {
	case "protocol_version":
		return "2"
	case "name":
		return "Gofer"
	case "version":
		return "0.1"
	case "known_command":
		return "true"
	case "list_commands":
		return "boardsize clear_board komi play genmove quit"
	case "boardsize":
		if len(parts) < 2 {
			return "boardsize not an integer"
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil || n < 2 || n > 19 {
			return "unacceptable size"
		}
		s.Size = n
		s.Board = NewBoard(n, s.Komi)
		s.Search.ResetArena()
		return ""
	case "clear_board":
		s.Board = NewBoard(s.Size, s.Komi)
		s.Search.ResetArena()
		return ""
	case "komi":
		if len(parts) < 2 {
			return "komi not a float"
		}
		k, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			return "komi not a float"
		}
		s.Komi = k
		s.Board = NewBoard(s.Size, k)
		s.Search.ResetArena()
		return ""
	case "play":
		if len(parts) < 3 {
			return "syntax error"
		}
		color, err := parseColor(parts[1])
		if err != nil {
			return err.Error()
		}
		if s.Board.Player() != color {
			return "wrong color to move"
		}
		m, err := parseVertex(parts[2], s.Size)
		if err != nil {
			return err.Error()
		}
		if !s.Rules.Play(s.Board, m) {
			return "illegal move"
		}
		s.Search.ResetArena()
		return ""
	case "genmove":
		if len(parts) < 2 {
			return "syntax error"
		}
		color, err := parseColor(parts[1])
		if err != nil {
			return err.Error()
		}
		if s.Board.Player() != color {
			return "wrong color to move"
		}
		m := s.Search.BestMove(s.Board)
		if !s.Rules.Play(s.Board, m) {
			return "pass"
		}
		return moveToVertex(m, s.Size)
	case "quit":
		return ""
	default:
		return "? unknown command"
	}
}

func parseColor(s string) (Color, error) {
	switch strings.ToUpper(s) {
	case "B", "BLACK":
		return Black, nil
	case "W", "WHITE":
		return White, nil
	default:
		return Empty, fmt.Errorf("invalid color")
	}
}

func parseVertex(v string, size int) (Move, error) {
	if strings.ToLower(v) == "pass" {
		return PassMove, nil
	}
	v = strings.ToUpper(strings.TrimSpace(v))
	if len(v) < 2 {
		return Move{}, fmt.Errorf("invalid vertex")
	}
	x := int(v[0] - 'A')
	if v[0] >= 'I' {
		x--
	}
	row, err := strconv.Atoi(v[1:])
	if err != nil || row < 1 || row > size {
		return Move{}, fmt.Errorf("invalid vertex")
	}
	y := size - row
	if x < 0 || y < 0 || x >= size || y >= size {
		return Move{}, fmt.Errorf("invalid vertex")
	}
	return StoneMove(At(x, y)), nil
}

func moveToVertex(m Move, size int) string {
	if m.Pass {
		return "pass"
	}
	col := 'A' + m.Point.X
	if col >= 'I' {
		col++
	}
	row := size - m.Point.Y
	return fmt.Sprintf("%c%d", col, row)
}

// --- from internal/engine\sample.go ---
// Sample is one training record (M10/M11 export schema).
type Sample struct {
	BoardHash uint64      `json:"board_hash"`
	MoveNum   int         `json:"move_num"`
	Policy    []float32   `json:"policy"`
	PolicyOpp []float32   `json:"policy_opp,omitempty"`
	ToPlay    Color `json:"to_play"`
	Ownership []float32   `json:"ownership,omitempty"`
	ScorePDF  []float64   `json:"score_pdf,omitempty"`
	ScoreCDF  []float64   `json:"score_cdf,omitempty"`
}

// --- from internal/engine\selfplay.go ---
// SelfplayConfig holds self-play parameters (paper M10 subset).
type SelfplayConfig struct {
	Games          int
	BoardSize      int
	Komi           float64
	Playouts       int
	CapRandomizeP  float64
	Seed           int64
	RulesRandomize bool
}

// DefaultSelfplayConfig returns reasonable defaults.
func DefaultSelfplayConfig() SelfplayConfig {
	return SelfplayConfig{
		Games:          1,
		BoardSize:      9,
		Komi:           6.5,
		Playouts:       30,
		CapRandomizeP:  0.25,
		Seed:           1,
		RulesRandomize: false,
	}
}

// RunSelfplay plays games and returns training samples with visit-weighted π.
func RunSelfplay(cfg SelfplayConfig) []Sample {
	rng := rand.New(rand.NewSource(cfg.Seed))
	var samples []Sample
	for g := 0; g < cfg.Games; g++ {
		rs := Chinese()
		if cfg.RulesRandomize && rng.Float64() < 0.5 {
			rs = TrompTaylor()
		}
		size := cfg.BoardSize
		if cfg.RulesRandomize {
			sizes := []int{9, 13, 19}
			size = sizes[rng.Intn(len(sizes))]
		}
		b := NewBoard(size, cfg.Komi)
		playouts := cfg.Playouts
		if rng.Float64() < cfg.CapRandomizeP {
			playouts = cfg.Playouts * 2
		}
		scfg := DefaultConfig()
		scfg.Playouts = playouts
		scfg.Seed = cfg.Seed + int64(g)
		eng := NewEngine(rs, nil, scfg)
		passes := 0
		for moveNum := 0; moveNum < size*size+2; moveNum++ {
			moves := rs.LegalMoves(b)
			if onlyPass(moves) {
				break
			}
			m := eng.BestMove(b)
			pi := eng.RootPolicy(moves)
			samples = append(samples, Sample{
				BoardHash: b.Hash(),
				MoveNum:   moveNum,
				Policy:    pi,
				ToPlay:    b.Player(),
			})
			rs.Play(b, m)
			if m.Pass {
				passes++
			} else {
				passes = 0
			}
			if passes >= 2 {
				break
			}
		}
	}
	return samples
}

func onlyPass(moves []Move) bool {
	for _, m := range moves {
		if !m.Pass {
			return false
		}
	}
	return true
}
